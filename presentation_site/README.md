# ECG Guardian · Interactive Web Presentation

An interactive, animated 16:9 presentation website built to showcase the **ECG Guardian** platform, matching its deep clinical green, warm parchment, and phosphor CRT telemetry aesthetic.

Lead Researcher & Cardiologist: **Dr. Sonali Patel**

---

## Tech Stack & Features

- **Three.js (WebGL):** Real-time 3D cardiac dipole P-QRS-T particle ribbon and ambient telemetry particles that react across slide transitions.
- **Anime.js:** Choreographed slide reveals, SVG heartbeat trace line-drawing, and smooth metric roll-up animations.
- **Live Platform Screenshots:** Fresh 1440×900 captures taken directly from the running web application using Playwright (`capture_screenshots.py`).
- **Interactive Controls:**
  - Mouse wheel / trackpad scrolling across slides
  - `→` / `Space` / `PageDown`: Next slide
  - `←` / `Backspace` / `PageUp`: Previous slide
  - `Tab` / `G` / `O`: Slide Overview Grid
  - `F`: Fullscreen mode
  - `P`: Browser Print / Save as PDF
  - Click any screenshot to open full-size Lightbox zoom

---

## Local Scripts

### 1. Capture Fresh Screenshots from Running App (`http://127.0.0.1:5000`)
```bash
python presentation_site/capture_screenshots.py
```

### 2. Convert Web Presentation to 11-Page PDF (`ECG_Guardian_Presentation_Deck.pdf`)
```bash
python presentation_site/generate_pdf.py
```
