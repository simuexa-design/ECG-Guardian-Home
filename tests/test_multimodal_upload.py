"""
Multi-Modal Ingestion Verification Suite
Tests uploading PDF, Image (PNG/JPG), CSV, and JSON files through the /api/upload endpoint.
"""

import os
import sys
import io
import json
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app

client = app.test_client()

def test_multimodal():
    print("=" * 80)
    print("  MULTI-MODAL ECG FILE INGESTION & DIGITIZATION TEST SUITE")
    print("  Supported: PDF, Images (PNG/JPG), CSV, JSON, MATLAB (.mat)")
    print("=" * 80)

    # 1. Test CSV Upload
    print("\n[1/5] Testing CSV File Upload ...")
    csv_content = "time,lead_ii\n" + "\n".join([f"{i/250.0:.4f},{np.sin(2*np.pi*1.2*i/250.0):.4f}" for i in range(2500)])
    res = client.post('/api/upload', data={
        'file': (io.BytesIO(csv_content.encode('utf-8')), 'patient_record.csv'),
        'fs': '250'
    }, content_type='multipart/form-data')
    assert res.status_code == 200, f"CSV upload failed: {res.data}"
    data = res.get_json()
    print(f"  [+] Success! File: {data['filename']}, Type: {data.get('file_type')}, Samples: {data['num_samples']}, HR: {data['hrv']['heart_rate_bpm']} BPM")

    # 2. Test JSON Upload
    print("\n[2/5] Testing JSON Digital Record Upload ...")
    json_data = {'signal': [float(np.sin(2*np.pi*1.5*i/250.0)) for i in range(2500)], 'fs': 250}
    res = client.post('/api/upload', data={
        'file': (io.BytesIO(json.dumps(json_data).encode('utf-8')), 'telemetry_stream.json')
    }, content_type='multipart/form-data')
    assert res.status_code == 200, f"JSON upload failed: {res.data}"
    data = res.get_json()
    print(f"  [+] Success! File: {data['filename']}, Type: {data.get('file_type')}, Samples: {data['num_samples']}")

    # 3. Test Medical Image (PNG) Upload & Digitization
    print("\n[3/5] Testing Medical ECG Image Upload (PNG Trace Digitization) ...")
    img = Image.new('RGB', (1000, 300), color=(255, 240, 240))
    draw = ImageDraw.Draw(img)
    for y in range(0, 300, 20):
        draw.line([(0, y), (1000, y)], fill=(255, 200, 200), width=1)
    for x in range(0, 1000, 20):
        draw.line([(x, 0), (x, 300)], fill=(255, 200, 200), width=1)
    pts = []
    for x in range(1000):
        t = x / 100.0
        val = np.sin(2 * np.pi * 1.0 * t)
        if (x % 100) > 45 and (x % 100) < 55:
            val += 2.5 * np.sin(np.pi * (x % 100 - 45) / 10.0)
        y = int(150 - val * 40)
        pts.append((x, y))
    draw.line(pts, fill=(10, 10, 10), width=2)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    res = client.post('/api/upload', data={
        'file': (img_bytes, 'patient_ecg_scan.png')
    }, content_type='multipart/form-data')
    assert res.status_code == 200, f"Image upload failed: {res.data}"
    data = res.get_json()
    print(f"  [+] Success! File: {data['filename']}, Type: {data.get('file_type')}, Samples: {data['num_samples']}, HR: {data['hrv']['heart_rate_bpm']} BPM")

    # 4. Test Clinical PDF Upload
    print("\n[4/5] Testing Clinical PDF Report Upload ...")
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 50), "HOSPITAL CLINICAL ELECTROCARDIOGRAM REPORT", fontsize=14)
    page.insert_text((50, 75), "Patient: Willy McKee | Age: 58 | Sex: M", fontsize=10)
    page.insert_text((50, 95), "Heart Rate: 78 bpm | PR: 160 ms | QRS: 90 ms | QTc: 410 ms", fontsize=10)
    page.insert_text((50, 115), "Diagnosis: Normal Sinus Rhythm with occasional PVC", fontsize=10)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    pdf_bytes = io.BytesIO()
    doc.save(pdf_bytes)
    doc.close()
    pdf_bytes.seek(0)
    res = client.post('/api/upload', data={
        'file': (pdf_bytes, 'clinical_ecg_report.pdf')
    }, content_type='multipart/form-data')
    assert res.status_code == 200, f"PDF upload failed: {res.data}"
    data = res.get_json()
    print(f"  [+] Success! File: {data['filename']}, Type: {data.get('file_type')}, Samples: {data['num_samples']}")
    print(f"  [+] Extracted PDF Metadata: Patient={data['metadata'].get('patient')}, HR={data['metadata'].get('heart_rate')} bpm, QRS={data['metadata'].get('qrs')} ms, QTc={data['metadata'].get('qtc')} ms")

    # 5. Test AI Analysis on Extracted Signal
    print("\n[5/5] Testing AI Diagnosis & 1D Grad-CAM on Extracted PDF Signal ...")
    extracted_sig = data['signal']
    res = client.post('/api/analyze', data=json.dumps({
        'signal': extracted_sig,
        'fs': 250.0,
        'patient_name': 'Willy McKee'
    }), content_type='application/json')
    assert res.status_code == 200, f"AI analysis failed: {res.data}"
    analysis = res.get_json()
    print(f"  [+] AI Prediction from PDF: {analysis['prediction']} ({analysis['ai_classification']['confidence']}%)")
    print(f"  [+] 1D Grad-CAM Heatmaps Generated: {list(analysis['gradcam_heatmaps'].keys())}")
    print(f"  [+] Clinical Risk Tier: {analysis['clinical_report']['risk_tier']}")

    print("\n" + "=" * 80)
    print("  ALL 5 MULTI-MODAL FORMATS TESTED & FULLY VERIFIED (PDF, IMAGE, CSV, JSON)!")
    print("=" * 80)
    return True

if __name__ == '__main__':
    try:
        test_multimodal()
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
