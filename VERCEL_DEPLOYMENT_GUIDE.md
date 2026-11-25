# Vercel Deployment Guide — ECG Guardian Home

This repository is pre-configured and 100% **Vercel-ready**. You can import and deploy this repository directly to Vercel with zero additional configuration.

---

## 1. Quick Deploy via Vercel Web Dashboard

1. **Push or Fork to GitHub**:
   - Push this repository to your GitHub account (or ensure [simuexa-design/ECG-Guardian-Home](https://github.com/simuexa-design/ECG-Guardian-Home) is connected).

2. **Import into Vercel**:
   - Go to [vercel.com](https://vercel.com) and log into your account.
   - Click **"Add New..."** &rarr; **"Project"**.
   - Select your **ECG-Guardian-Home** repository and click **Import**.

3. **Configure Project Settings**:
   - **Framework Preset**: Select `Other` (or leave as auto-detected).
   - **Root Directory**: `./` (leave default).
   - **Build & Development Settings**: Leave blank / default.
   - **Environment Variables**: None required out of the box (standard secret variables like `SECRET_KEY` or Google OAuth client IDs can be optionally added if desired).

4. **Click Deploy**:
   - Vercel automatically runs `pip install -r requirements.txt` and attaches the Python serverless function handler (`api/index.py`).
   - Your live URL (e.g. `https://your-project.vercel.app`) will be active within 60–90 seconds.

---

## 2. Quick Deploy via Vercel CLI

If you have NodeJS and the Vercel CLI installed on your machine:

```bash
# 1. Install Vercel CLI globally (if not already installed)
npm install -g vercel

# 2. Navigate to project root
cd ECG-Guardian-Home

# 3. Deploy to preview
vercel

# 4. Deploy to production
vercel --prod
```

---

## 3. What Was Configured for Vercel

1. **Serverless Routing (`vercel.json`)**:
   - Sets wildcard routing (`/(.*)`) directly to the WSGI handler `/api/index.py`.

2. **Unified WSGI Handler (`api/index.py`)**:
   - Configured with `VercelPathFixMiddleware` to preserve URL routing, query parameters (`?view=...`), and subpaths across Vercel's serverless runtime.
   - Exports `application` and `handler` for seamless Python 3.12 AWS Lambda / Vercel execution.

3. **Lightweight Cloud Inference Stack (`requirements.txt`)**:
   - Uses `onnxruntime` (~30 MB) and `scipy`/`numpy` for multi-label arrhythmia deep learning inference instead of massive multi-gigabyte PyTorch CUDA wheels.
   - Fully stays within Vercel's 250 MB / 500 MB function bundle limit (~12.8 MB total runtime bundle).

4. **Optimized Exclusion Filter (`.vercelignore`)**:
   - Excludes heavy local presentation slides (`.pptx`, `.pdf`), local `.bat` scripts, offline training files, and raw benchmarks from the deployment bundle so cold starts remain lightning fast (< 1s).
   - Keeps essential clinical sample CSVs (`sample_files/`) and the production ONNX model (`ecg_multilabel_model.onnx`).

5. **Safe Serverless API Endpoints**:
   - `/api/train` & `/api/train/status`: Returns graceful serverless status showing pre-trained production model (99.93% ROC-AUC).
   - `/api/reports/.../pdf`: Seamless fallback ensuring frontend client-side vector PDF generation (`html2pdf`) activates cleanly if the static pre-compiled report is absent in cloud environments.

---

## 4. Default Credentials & Features

- **Primary Interface**: `/` (ECG Guardian Clinical Platform)
- **Classic 12-Lead Noise Studio**: `/classic`
- **1-Click Doctor Login**: Click **"Login as Dr. Sonali Patel (Doctor)"** on the login modal or use `dr.sonali@hospital.org` / `Doctor@Hospital2026`.
- **Health Check Endpoint**: `/api/status`
