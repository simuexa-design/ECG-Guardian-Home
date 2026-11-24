# Explainable Deep Learning for Multi-Label ECG Arrhythmia Classification

> **College Academic Research Project**  
> **Lead Researcher & Cardiologist:** Dr. Sonali Patel  
> An end-to-end, clinically-grounded Medical AI platform combining 1D Deep Residual Neural Networks with Squeeze-and-Excitation channel attention and 1D Gradient-Weighted Class Activation Mapping (Grad-CAM) to deliver transparent, multi-label electrocardiogram diagnosis.

---

## 1. Project Overview

Electrocardiogram (ECG) interpretation frequently involves **multi-label classification**, where a single patient recording exhibits multiple co-occurring cardiac anomalies (e.g., Sinus Tachycardia alongside Atrial Fibrillation or Left Bundle Branch Block complicated by Ventricular Ectopy). Traditional "black-box" deep learning models output classification probabilities without showing the underlying electrophysiological morphology that led to the prediction.

This project delivers:
1. **Multi-Stage ECG Signal Preprocessing**: Butterworth bandpass filtering (0.5–45 Hz), powerline notch filtering (50/60 Hz), double-median baseline wander correction, and Z-score lead standardization.
2. **Pan-Tompkins QRS Complex Detection & HRV Telemetry**: Accurate real-time R-peak localization, Heart Rate (BPM), mean RR intervals, SDNN, RMSSD, pNN50, and Bazett-corrected QTc calculation.
3. **SE-ResNet-1D Deep Neural Network**: Multi-scale 1D residual convolutional network with Squeeze-and-Excitation (SE) channel attention and multi-head temporal attention pooling, trained with multi-label Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
4. **1D Grad-CAM (Explainable AI)**: Mathematical gradient attribution that pinpoints the exact temporal segments (P-wave, QRS complex, ST segment, or fibrillatory baseline) that triggered each specific predicted condition.
5. **Interactive Medical Research Web Application**: A modern, clinical-grade user interface with an authentic 25 mm/s, 10 mm/mV ECG monitor grid, synchronized cursor scrubbing, multi-label probability breakdown, XAI heatmaps, and evidence-based diagnostic reporting.

---

## 2. Mathematical Methodology & Architecture

### A. SE-ResNet-1D Neural Network
Given an input ECG signal $X \in \mathbb{R}^{1 \times L}$ where $L = 2500$ (10 seconds at 250 Hz):

1. **Stem**: Wide 1D convolution ($\text{kernel} = 15, \text{stride} = 2$) captures coarse morphology, followed by Batch Normalization, GELU activation, and Max Pooling.
2. **Residual Blocks with Squeeze-and-Excitation (SE)**:
   $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$
   The SE block recalibrates feature map channel weights:
   $$\mathbf{z}_c = \frac{1}{L'} \sum_{t=1}^{L'} \mathbf{u}_c(t)$$
   $$\mathbf{s} = \sigma\left( W_2 \cdot \text{ReLU}(W_1 \mathbf{z}) \right)$$
   $$\mathbf{\tilde{x}}_c = \mathbf{s}_c \cdot \mathbf{u}_c$$
3. **Temporal Attention Pooling**: Multi-head attention dynamically focuses on pathological segments across the cardiac cycle before the final multi-label linear projection.

### B. Multi-Label Loss Formulation
For $C = 10$ arrhythmia classes, multi-label optimization uses independent sigmoid probabilities:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{C} \sum_{c=1}^C \left[ y_c \log \sigma(\hat{y}_c) + (1 - y_c) \log (1 - \sigma(\hat{y}_c)) \right]$$

### C. 1D Grad-CAM Formulation
To explain prediction for class $c$, the gradient of class logit $y^c$ with respect to feature map $A^k$ of the final convolutional layer is computed:
$$\alpha_k^c = \frac{1}{L'} \sum_{t=1}^{L'} \frac{\partial y^c}{\partial A^k(t)}$$
The class activation map $L_{\text{Grad-CAM}}^c$ is:
$$L_{\text{Grad-CAM}}^c(t) = \text{ReLU}\left( \sum_k \alpha_k^c A^k(t) \right)$$
$L_{\text{Grad-CAM}}^c$ is upsampled via linear interpolation to match the original 2500-sample resolution, normalized to $[0, 1]$, and rendered as a color-coded heatmap overlay.

---

## 3. Diagnostic Arrhythmia Classes

| Code | Arrhythmia Name | Electrophysiological Criteria |
| :--- | :--- | :--- |
| **NORM** | Normal Sinus Rhythm | Regular rate (60–100 bpm), standard P-QRS-T complexes |
| **AFIB** | Atrial Fibrillation | Irregularly irregular RR, absent P-waves, fibrillatory baseline |
| **AFL** | Atrial Flutter | Rapid regular atrial flutter waves in sawtooth pattern |
| **LBBB** | Left Bundle Branch Block | Delayed LV depolarization, wide QRS (>120ms), slurred R |
| **RBBB** | Right Bundle Branch Block | Delayed RV depolarization, classic rSR' "rabbit ears" in V1/V2 |
| **PAC** | Premature Atrial Contraction | Early ectopic P-wave followed by normal narrow QRS |
| **PVC** | Premature Ventricular Contraction | Early wide bizarre QRS complex followed by compensatory pause |
| **STACH**| Sinus Tachycardia | Regular sinus rhythm with Heart Rate > 100 bpm |
| **SBRAD**| Sinus Bradycardia | Regular sinus rhythm with resting Heart Rate < 60 bpm |
| **MI** | Myocardial Infarction / Ischemia| Acute ST-segment elevation (>0.2 mV) or pathological Q-waves |

---

## 4. Directory Structure

```
D:\ecg/
├── backend/
│   ├── app.py                     # Flask REST API (21 endpoints) & PyTorch server
│   ├── models/
│   │   ├── ecg_model.py           # SE-ResNet-1D PyTorch architecture
│   │   └── trainer.py             # Multi-label training pipeline & evaluation
│   ├── preprocessing/
│   │   ├── filters.py             # Butterworth bandpass, notch, baseline removal
│   │   ├── qrs_detector.py        # Pan-Tompkins algorithm & HRV telemetry
│   │   ├── signal_parser.py       # Time-series parsing and resampling
│   │   └── multimodal_parser.py   # Universal parser (PDF reports, PNG/JPG scans, CSV, JSON)
│   ├── xai/
│   │   ├── gradcam1d.py           # 1D Grad-CAM implementation
│   │   ├── saliency.py            # Gradient saliency & Integrated Gradients
│   │   └── interpreter.py         # Clinical diagnostic reasoning engine
│   ├── sample_data/
│   │   └── generator.py           # Biophysical cardiac dipole model & benchmarks
│   └── trained_models/
│       ├── ecg_multilabel_model.pt # Trained SE-ResNet-1D weights (1.99M params)
│       └── training_metrics.json  # Checkpoint validation metrics & history
├── sample_files/                  # Clinical benchmark files (PDF, CSV, 12-lead)
│   ├── case_01_normal_sinus.csv
│   ├── case_02_atrial_fibrillation.csv
│   ├── case_03_lbbb_tachycardia.csv
│   ├── case_04_rbbb_pvc.csv
│   ├── case_05_st_elevation_mi.csv
│   ├── case_06_afib_pvc.csv
│   └── willy_mckee_report_patient_123123_12lead.csv
├── static/
│   ├── css/
│   │   ├── guardian.css           # Modern hospital design system & dark mode
│   │   ├── cardioai.css           # Streamlit healthcare theme
│   │   └── style.css              # Classic canvas styling
│   ├── js/
│   │   ├── guardian.js            # ECG Guardian SPA state machine & Plotly charts
│   │   ├── app.js                 # Classic REST client
│   │   └── ecg_visualizer.js      # Canvas ECG grid & Grad-CAM renderer
│   ├── images/doctors-hero.jpg    # Clinical showcase assets
│   └── doctors-hero.jpg
├── templates/
│   ├── index.html                 # ECG Guardian medical platform (Dr. Sonali Patel)
│   └── index_classic.html         # Classic 12-lead Canvas Diagnostic Lab (/classic)
├── tests/                         # Automated verification test suite
│   ├── run_all_tests.py           # Master test runner
│   ├── test_api_endpoints.py      # Verification of all 21 REST API endpoints
│   ├── test_multimodal_upload.py  # Ingestion tests (PDF, PNG scan, CSV, JSON)
│   └── test_browser_workflow.py   # End-to-end browser DOM and clinical workflow
├── .gitignore                     # Standard Git ignore rules
├── requirements.txt               # Pinned Python package dependencies
├── run_server.py                  # Python server launcher
├── run_server.bat                 # 1-Click Windows execution script
└── README.md                      # Comprehensive academic documentation
```

---

## 5. How to Run the Application

### Option A: 1-Click Launch (Recommended)
Double-click:
```
run_server.bat
```

### Option B: Command Line Launch
```bash
python run_server.py
```

Open your web browser and navigate to:
- **ECG Guardian Medical Platform**: `http://127.0.0.1:5000`
- **Classic 12-Lead Diagnostic Lab**: `http://127.0.0.1:5000/classic`

#### Doctor Login Credentials:
- **1-Click Quick Login**: Click **"Login as Dr. Sonali Patel (Doctor)"**
- **Manual Login**: `dr.sonali@hospital.org` / `Doctor@Hospital2026`

---

## 6. Running the Automated Test Suite

To verify all system endpoints, multi-modal ingestion, and clinical workflows:
```bash
python tests/run_all_tests.py
```

---

## 7. Real Model Verification & Validation

The model is pre-calibrated and saved in `backend/trained_models/ecg_multilabel_model.pt`.
- **Validation ROC-AUC**: `0.9993`
- **Validation Macro F1**: `0.9567`
- **Validation Micro F1**: `0.9622`
- **Subset Accuracy**: `89.29%`
- **Hamming Loss**: `0.0107`
- **Total Parameters**: `1,998,731`

To benchmark or retrain the model at any time:
```bash
python train_model.py
```

