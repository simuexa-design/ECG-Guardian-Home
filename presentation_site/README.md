# ECG Guardian · Clinical Telemetry & AI Presentation Deck

An interactive, responsive 16:9 presentation deck and publication-grade PDF export engineered for the **ECG Guardian Clinical Platform**.

Lead Investigator & Author: **Dr. Sonali Patel, MD (Cardiology)**  
Institution: Apex Heart & Vascular Hospital, Jaipur  

---

## 🖥️ Interactive Presentation Site Features

- **Widescreen 16:9 Layout:** Auto-scales to any desktop, iPad, or conference display.
- **Dark Obsidian Clinical Aesthetic:** Authentic medical styling with animated ECG CRT waveform monitor sweep and live 72 BPM pulse indicator.
- **11 Curated Clinical Slides:**
  1. Title & Executive Briefing
  2. Clinical Motivation & Multi-Arrhythmia Challenge
  3. End-to-End System Architecture (5-Stage Pipeline)
  4. Platform Landing & Real-Time Phosphor CRT Telemetry
  5. Security, HIPAA Compliance & Role-Based Access Control
  6. Cardiologist Command Dashboard & Live Department Telemetry
  7. Multi-Modal ECG Ingestion & Noise Studio
  8. Multi-Label SE-ResNet-1D Deep Learning & 1D Grad-CAM Saliency
  9. Clinical Decision Support (CDS) & Differential Diagnosis Engine
  10. Formal Hospital Diagnostic Report & Dr. Sonali Patel Sign-Off
  11. Patient Telehealth Portal & Quantitative Validation Metrics
- **Keyboard Shortcuts:**
  - `→` / `Space` / `PageDown`: Next Slide
  - `←` / `Backspace` / `PageUp`: Previous Slide
  - `Tab` / `O`: Slide Grid Overview & Quick Jump
  - `F`: Toggle Fullscreen Presentation Mode
  - `P`: Browser Print / PDF Export Dialog
  - `Home` / `End`: First / Last Slide
- **Touch / Swipe Gestures:** Native touch support on tablets and mobile screens.

---

## 📄 Formats & Deliverables Included

1. **Web Presentation:** `index.html` (pure HTML5, CSS3, ES6 vanilla JS - zero external build dependencies).
2. **Interactive Web PDF:** `ECG_Guardian_Presentation_Deck.pdf` (11-page pixel-perfect 16:9 PDF rendered with Playwright & Microsoft Edge).
3. **PowerPoint Presentation:** `ECG_Guardian_Presentation_Deck.pptx` (Microsoft PowerPoint deck with 16:9 widescreen layout and high-res screenshots).
4. **ReportLab Slide Deck:** `ECG_Guardian_Clinical_Presentation.pdf` (Vector report generated via Python ReportLab).

---

## 🚀 How to Run & Regenerate Deliverables

### 1. View Presentation Locally
Double-click `index.html` or serve via Python:
```bash
python -m http.server 8000 --directory presentation_site
# Open http://localhost:8000/
```

### 2. Export HTML Presentation to PDF (via Playwright)
```bash
python presentation_site/generate_pdf.py
```

### 3. Regenerate PowerPoint (.pptx) Deck
```bash
python build_ppt_presentation.py
```
