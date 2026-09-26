import os
import sys
import base64
import io
import json
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Add project root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import UNet, predict_mri_image, load_or_create_model, get_device

app = Flask(__name__)
handler = app # Top-level export for Vercel deployment handler compatibility

# Load model weights on startup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "model_best_checkpoint.pth")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is None:
        device = get_device()
        _cached_model = load_or_create_model(weights_path=WEIGHTS_PATH, device=device)
    return _cached_model

def image_to_base64(img_np_or_pil, format="PNG"):
    if isinstance(img_np_or_pil, np.ndarray):
        if img_np_or_pil.ndim == 2:
            pil_img = Image.fromarray(img_np_or_pil)
        else:
            pil_img = Image.fromarray(img_np_or_pil.astype(np.uint8))
    else:
        pil_img = img_np_or_pil

    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def create_color_overlay(mri_gray, mask, opacity=0.4, colormap_name="Jet"):
    h, w = mri_gray.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    mri_rgb = cv2.cvtColor(mri_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

    import matplotlib.pyplot as plt
    if colormap_name == "Viridis":
        cmap = plt.get_cmap('viridis')
    elif colormap_name == "Crimson":
        cmap = plt.get_cmap('magma')
    elif colormap_name == "Hot":
        cmap = plt.get_cmap('hot')
    else:
        cmap = plt.get_cmap('jet')

    colored_mask = (cmap(mask.astype(np.float32))[:, :, :3] * 255.0).astype(np.float32)
    blend = mri_rgb.copy()
    mask_indices = mask > 0
    blend[mask_indices] = (1.0 - opacity) * mri_rgb[mask_indices] + opacity * colored_mask[mask_indices]
    return np.clip(blend, 0, 255).astype(np.uint8)

def draw_bounding_boxes(image_rgb, boxes):
    img_copy = image_rgb.copy()
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 50, 50), 2)
        cv2.putText(img_copy, "TUMOR", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)
    return img_copy

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Detection & U-Net Segmentation</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0E1117;
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-primary: #FAFAFA;
            --text-secondary: #A0AAB8;
            --accent-cyan: #00C9FF;
            --accent-green: #92FE9D;
            --danger-red: #EF4444;
            --success-green: #10B981;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        .grid-container {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 2rem;
        }

        .sidebar {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);

        }

        .sidebar h2 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: var(--accent-cyan);
        }

        .form-group {
            margin-bottom: 1.2rem;
        }

        .form-group label {
            display: block;
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }

        select, input[type="file"], input[type="range"] {
            width: 100%;
            padding: 0.6rem;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: inherit;
        }

        .btn {
            width: 100%;
            padding: 0.8rem;
            background: linear-gradient(135deg, #00C9FF, #92FE9D);
            border: none;
            border-radius: 8px;
            color: #0E1117;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .btn:hover { opacity: 0.9; }

        .main-content {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .status-banner {
            border-radius: 12px;
            padding: 1.2rem;
            font-weight: 700;
            font-size: 1.2rem;
            display: none;
        }

        .status-danger {
            background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.3));
            border: 1px solid var(--danger-red);
            color: #FCA5A5;
        }

        .status-success {
            background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(4,120,87,0.3));
            border: 1px solid var(--success-green);
            color: #6EE7B7;
        }

        .visual-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }

        .image-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .image-card h3 {
            font-size: 1rem;
            margin-bottom: 0.8rem;
            color: var(--text-secondary);
        }

        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #000;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--card-border);
        }

        .metrics-table th, .metrics-table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }

        .metrics-table th {
            background: rgba(255, 255, 255, 0.05);
            color: var(--accent-cyan);
        }

        .spinner {
            display: none;
            text-align: center;
            padding: 2rem;
            font-size: 1.2rem;
            color: var(--accent-cyan);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Brain Tumor Detection & U-Net Segmentation</h1>
        <p>PyTorch Deep Learning Clinical Decision Support System</p>
    </div>

    <div class="grid-container">
        <div class="sidebar">
            <h2>⚙️ Controls</h2>
            
            <div class="form-group">
                <label>Select Sample Brain MRI:</label>
                <select id="sampleSelect">
                    <option value="">-- Choose Sample --</option>
                    {% for sample in samples %}
                    <option value="{{ sample }}">{{ sample }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label>OR Upload Custom MRI Image:</label>
                <input type="file" id="fileUpload" accept="image/*">
            </div>

            <div class="form-group">
                <label>Probability Threshold: <span id="threshVal">0.35</span></label>
                <input type="range" id="threshold" min="0.10" max="0.90" step="0.05" value="0.35">
            </div>

            <div class="form-group">
                <label>Overlay Opacity: <span id="opacityVal">0.45</span></label>
                <input type="range" id="opacity" min="0.0" max="1.0" step="0.05" value="0.45">
            </div>

            <div class="form-group">
                <label>Color Palette:</label>
                <select id="colormap">
                    <option value="Jet">Jet Thermal</option>
                    <option value="Viridis">Viridis</option>
                    <option value="Crimson">Crimson Magma</option>
                    <option value="Hot">Hot Red</option>
                </select>
            </div>

            <button class="btn" onclick="runDetection()">🔍 Run Tumor Detection</button>
        </div>

        <div class="main-content">
            <div id="statusBanner" class="status-banner"></div>

            <div id="spinner" class="spinner">Analyzing MRI Scan with U-Net Neural Network...</div>

            <div id="resultsGrid" class="visual-grid" style="display:none;">
                <div class="image-card">
                    <h3>1. Preprocessed MRI</h3>
                    <img id="mriImg" src="" alt="MRI">
                </div>
                <div class="image-card">
                    <h3>2. U-Net Tumor Mask</h3>
                    <img id="maskImg" src="" alt="Mask">
                </div>
                <div class="image-card">
                    <h3>3. Tumor Overlay Visualizer</h3>
                    <img id="overlayImg" src="" alt="Overlay">
                </div>
            </div>

            <table id="metricsTable" class="metrics-table" style="display:none;">
                <thead>
                    <tr><th>Metric</th><th>Analysis Value</th></tr>
                </thead>
                <tbody>
                    <tr><td>Tumor Detection Status</td><td id="mStatus">-</td></tr>
                    <tr><td>Detection Confidence</td><td id="mConf">-</td></tr>
                    <tr><td>Tumor Surface Area (%)</td><td id="mArea">-</td></tr>
                    <tr><td>Tumor Surface Area (Pixels)</td><td id="mPixels">-</td></tr>
                    <tr><td>Bounding Box Regions</td><td id="mBoxes">-</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        document.getElementById('threshold').addEventListener('input', (e) => {
            document.getElementById('threshVal').innerText = e.target.value;
        });
        document.getElementById('opacity').addEventListener('input', (e) => {
            document.getElementById('opacityVal').innerText = e.target.value;
        });

        async function runDetection() {
            const sampleSelect = document.getElementById('sampleSelect').value;
            const fileUpload = document.getElementById('fileUpload').files[0];
            const threshold = parseFloat(document.getElementById('threshold').value);
            const opacity = parseFloat(document.getElementById('opacity').value);
            const colormap = document.getElementById('colormap').value;

            if (!sampleSelect && !fileUpload) {
                alert("Please select a sample MRI scan or upload an image first.");
                return;
            }

            document.getElementById('spinner').style.display = 'block';
            document.getElementById('resultsGrid').style.display = 'none';
            document.getElementById('metricsTable').style.display = 'none';
            document.getElementById('statusBanner').style.display = 'none';

            const formData = new FormData();
            formData.append('threshold', threshold);
            formData.append('opacity', opacity);
            formData.append('colormap', colormap);

            if (fileUpload) {
                formData.append('file', fileUpload);
            } else {
                formData.append('sample', sampleSelect);
            }

            try {
                const res = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('spinner').style.display = 'none';

                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }

                // Render Banner
                const banner = document.getElementById('statusBanner');
                banner.style.display = 'block';
                if (data.has_tumor) {
                    banner.className = 'status-banner status-danger';
                    banner.innerHTML = `⚠️ BRAIN TUMOR DETECTED | Confidence: ${(data.confidence * 100).toFixed(1)}% | Estimated Area: ${data.area_percentage.toFixed(2)}% of MRI region`;
                } else {
                    banner.className = 'status-banner status-success';
                    banner.innerHTML = `✅ NO TUMOR DETECTED | Normal Scan Profile | Confidence: ${((1 - data.confidence) * 100).toFixed(1)}%`;
                }

                // Render Images
                document.getElementById('mriImg').src = data.mri_base64;
                document.getElementById('maskImg').src = data.mask_base64;
                document.getElementById('overlayImg').src = data.overlay_base64;
                document.getElementById('resultsGrid').style.display = 'grid';

                // Render Metrics
                document.getElementById('mStatus').innerText = data.has_tumor ? "Detected 🚨" : "Normal Clean ✅";
                document.getElementById('mConf').innerText = (data.confidence * 100).toFixed(2) + "%";
                document.getElementById('mArea').innerText = data.area_percentage.toFixed(2) + "%";
                document.getElementById('mPixels').innerText = data.pixel_count + " px";
                document.getElementById('mBoxes').innerText = data.bounding_boxes.length + " region(s)";
                document.getElementById('metricsTable').style.display = 'table';

            } catch (err) {
                document.getElementById('spinner').style.display = 'none';
                alert("Failed to analyze image: " + err);
            }
        }

        // Auto-run first sample on load
        window.addEventListener('load', () => {
            const select = document.getElementById('sampleSelect');
            if (select.options.length > 1) {
                select.selectedIndex = 1;
                runDetection();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        samples = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.png')])
    return render_template_string(HTML_TEMPLATE, samples=samples)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        model = get_model()
        threshold = float(request.form.get('threshold', 0.35))
        opacity = float(request.form.get('opacity', 0.45))
        colormap = request.form.get('colormap', 'Jet')

        image_input = None
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            img = Image.open(file.stream)
            image_input = np.array(img)
        elif 'sample' in request.form and request.form['sample']:
            sample_name = request.form['sample']
            sample_path = os.path.join(SAMPLES_DIR, sample_name)
            img = Image.open(sample_path)
            arr = np.array(img)
            w = arr.shape[1]
            mri_crop = arr[:, :int(w / 3), :]
            image_input = mri_crop

        if image_input is None:
            return jsonify({'error': 'No image input provided'}), 400

        result = predict_mri_image(image_input, model=model, threshold=threshold)

        has_tumor = result["has_tumor"]
        confidence = result["confidence"]
        area_pct = result["area_percentage"]
        pixel_count = result["pixel_count"]
        mask_128 = result["binary_mask_128"]
        resized_gray = result["resized_gray"]
        boxes = result["bounding_boxes"]

        overlay_img = create_color_overlay(resized_gray, mask_128, opacity=opacity, colormap_name=colormap)
        if has_tumor and boxes:
            overlay_img = draw_bounding_boxes(overlay_img, boxes)

        mri_b64 = image_to_base64(resized_gray)
        mask_b64 = image_to_base64((mask_128 * 255).astype(np.uint8))
        overlay_b64 = image_to_base64(overlay_img)

        return jsonify({
            'has_tumor': bool(has_tumor),
            'confidence': float(confidence),
            'area_percentage': float(area_pct),
            'pixel_count': int(pixel_count),
            'bounding_boxes': boxes,
            'mri_base64': mri_b64,
            'mask_base64': mask_b64,
            'overlay_base64': overlay_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
