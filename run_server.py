"""
Server Launcher Script for Explainable Multi-Label ECG Arrhythmia Classification
"""

import os
import sys

# Ensure paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(CURRENT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app

if __name__ == '__main__':
    print("=" * 75)
    print("  EXPLAINABLE DEEP LEARNING FOR MULTI-LABEL ECG ARRHYTHMIA CLASSIFICATION")
    print("  College Academic Project - AI Medical Diagnostic Platform")
    print("=" * 75)
    print("  * Backend Engine: PyTorch 2.15 + Flask")
    print("  * Network Architecture: SE-ResNet-1D (Squeeze-and-Excitation + Attention)")
    print("  * Explainable AI (XAI): 1D Grad-CAM + Integrated Gradients")
    print("  * Signal Processing: Butterworth Bandpass + Pan-Tompkins QRS Telemetry")
    print("=" * 75)
    print("\n  >>> Access the Medical AI Web Interface at: http://127.0.0.1:5000\n")
    print("=" * 75)
    app.run(host='127.0.0.1', port=5000, debug=False)

