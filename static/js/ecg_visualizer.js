/**
 * Medical ECG & 1D Grad-CAM Canvas Visualizer
 * Renders calibrated clinical ECG grids, 1D waveforms, Pan-Tompkins R-peaks,
 * and continuous Explainable AI Grad-CAM heatmaps with interactive cursor scrubbing.
 */

class ECGVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.rawSignal = [];
    this.cleanSignal = [];
    this.heatmaps = {};
    this.activeClass = null;
    this.rPeaks = [];
    this.fs = 250.0; // 250 samples per second

    this.showRaw = false;
    this.showHeatmap = true;
    this.showSaliency = false;
    this.showRPeaks = true;
    this.heatmapOpacity = 0.65;
    this.saliency = [];

    // Viewport window
    this.viewStart = 0; // in samples
    this.viewEnd = 2500;
    this.zoomLevel = 1.0;

    // Hover state
    this.hoverIdx = -1;
    this.hudElement = document.getElementById('cursorHud');

    this._initCanvasSize();
    this._attachEvents();
  }

  _initCanvasSize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.render();
  }

  _attachEvents() {
    window.addEventListener('resize', () => this._initCanvasSize());

    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const totalViewSamples = this.viewEnd - this.viewStart;
      const sampleIdx = Math.floor(this.viewStart + (mouseX / this.width) * totalViewSamples);

      if (sampleIdx >= 0 && sampleIdx < this.cleanSignal.length) {
        this.hoverIdx = sampleIdx;
        this._updateHud(sampleIdx, mouseX, mouseY);
        this.render();
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoverIdx = -1;
      if (this.hudElement) this.hudElement.style.display = 'none';
      this.render();
    });
  }

  _updateHud(idx, x, y) {
    if (!this.hudElement) return;
    const timeSec = (idx / this.fs).toFixed(3);
    const volt = this.cleanSignal[idx] !== undefined ? this.cleanSignal[idx].toFixed(2) : '0.00';
    let camScore = '0.00';

    if (this.activeClass && this.heatmaps[this.activeClass]) {
      const arr = this.heatmaps[this.activeClass];
      if (arr[idx] !== undefined) {
        camScore = (arr[idx] * 100).toFixed(1) + '%';
      }
    }

    this.hudElement.innerHTML = `Time: <b>${timeSec}s</b> | Ampl: <b>${volt} mV</b> | Grad-CAM Salience: <b style="color:#f43f5e">${camScore}</b>`;
    this.hudElement.style.display = 'block';
  }

  setData(data) {
    this.rawSignal = data.raw_signal || [];
    this.cleanSignal = data.clean_signal || [];
    this.rPeaks = data.r_peaks || [];
    this.fs = data.fs || 250.0;
    this.viewStart = 0;
    this.viewEnd = this.cleanSignal.length > 0 ? this.cleanSignal.length : 2500;
    this.render();
  }

  setHeatmaps(heatmaps, primaryClass) {
    this.heatmaps = heatmaps || {};
    this.activeClass = primaryClass;
    this.render();
  }

  setActiveExplainClass(classId) {
    this.activeClass = classId;
    this.render();
  }

  setHeatmapOpacity(opacity) {
    this.heatmapOpacity = parseFloat(opacity);
    this.render();
  }

  toggleRaw(show) {
    this.showRaw = show;
    this.render();
  }

  toggleHeatmap(show) {
    this.showHeatmap = show;
    this.render();
  }

  setSaliency(saliencyArr) {
    this.saliency = saliencyArr || [];
    this.render();
  }

  toggleSaliency(show) {
    this.showSaliency = show;
    this.render();
  }

  toggleRPeaks(show) {
    this.showRPeaks = show;
    this.render();
  }

  zoom(factor) {
    const currentLen = this.viewEnd - this.viewStart;
    const center = (this.viewStart + this.viewEnd) / 2;
    const newLen = Math.max(250, Math.min(this.cleanSignal.length, currentLen * factor));
    this.viewStart = Math.max(0, Math.floor(center - newLen / 2));
    this.viewEnd = Math.min(this.cleanSignal.length, Math.floor(center + newLen / 2));
    this.render();
  }

  resetZoom() {
    this.viewStart = 0;
    this.viewEnd = this.cleanSignal.length > 0 ? this.cleanSignal.length : 2500;
    this.render();
  }

  // Get color for Grad-CAM attribution score (0.0 to 1.0)
  _getHeatColor(val, alpha = 1.0) {
    // 0.0 -> deep blue, 0.35 -> cyan, 0.7 -> yellow/amber, 1.0 -> vivid red
    const v = Math.max(0, Math.min(1, val));
    let r, g, b;
    if (v < 0.33) {
      const t = v / 0.33;
      r = Math.floor(30 + t * (6 - 30));
      g = Math.floor(41 + t * (182 - 41));
      b = Math.floor(59 + t * (212 - 59));
    } else if (v < 0.66) {
      const t = (v - 0.33) / 0.33;
      r = Math.floor(6 + t * (234 - 6));
      g = Math.floor(182 + t * (179 - 182));
      b = Math.floor(212 + t * (8 - 212));
    } else {
      const t = (v - 0.66) / 0.34;
      r = Math.floor(234 + t * (239 - 234));
      g = Math.floor(179 - t * (179 - 68));
      b = Math.floor(8 + t * (68 - 8));
    }
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  render() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // 1. Clear background
    ctx.fillStyle = '#090e17';
    ctx.fillRect(0, 0, w, h);

    // 2. Draw Clinical ECG Paper Grid (25 mm/s, 10 mm/mV)
    this._drawEcgGrid(ctx, w, h);

    const sig = this.cleanSignal;
    if (!sig || sig.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Select a benchmark case or upload an ECG recording to visualize signals', w / 2, h / 2);
      return;
    }

    const startIdx = Math.max(0, this.viewStart);
    const endIdx = Math.min(sig.length - 1, this.viewEnd);
    const totalViewSamples = endIdx - startIdx;
    if (totalViewSamples <= 0) return;

    // Amplitude scaling: map voltage to canvas height
    // Find min and max within view to scale comfortably
    let minV = -2.5;
    let maxV = 2.5;
    for (let i = startIdx; i <= endIdx; i++) {
      if (sig[i] < minV) minV = sig[i] - 0.5;
      if (sig[i] > maxV) maxV = sig[i] + 0.5;
    }
    const rangeV = maxV - minV || 1;
    const centerY = h / 2;

    const getX = (idx) => ((idx - startIdx) / totalViewSamples) * w;
    const getY = (val) => h - ((val - minV) / rangeV) * (h * 0.8) - (h * 0.1);

    // 3. Draw Grad-CAM Heatmap Ribbon under waveform
    const camArr = (this.activeClass && this.heatmaps[this.activeClass]) ? this.heatmaps[this.activeClass] : null;
    if (this.showHeatmap && camArr && camArr.length > 0) {
      // Draw vertical gradient bands along the timeline
      const step = Math.max(1, Math.floor(totalViewSamples / (w * 0.75)));
      for (let i = startIdx; i < endIdx; i += step) {
        const x1 = getX(i);
        const x2 = getX(Math.min(endIdx, i + step));
        const val = camArr[i] || 0;

        if (val > 0.05) {
          const color = this._getHeatColor(val, val * this.heatmapOpacity * 0.45);
          ctx.fillStyle = color;
          ctx.fillRect(x1, 0, x2 - x1 + 1, h);
        }
      }
    }

    // 4. Draw Raw Signal if enabled (grey/muted)
    if (this.showRaw && this.rawSignal && this.rawSignal.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.lineWidth = 1.2;
      for (let i = startIdx; i <= endIdx; i++) {
        const x = getX(i);
        const y = getY(this.rawSignal[i]);
        if (i === startIdx) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 5. Draw Filtered ECG Waveform with Grad-CAM Dynamic Stroke Color
    ctx.lineWidth = 2.2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let i = startIdx; i < endIdx; i++) {
      const x1 = getX(i);
      const y1 = getY(sig[i]);
      const x2 = getX(i + 1);
      const y2 = getY(sig[i + 1]);

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      if (this.showHeatmap && camArr && camArr[i] !== undefined) {
        const camVal = camArr[i];
        // Glow stronger with high attribution
        ctx.strokeStyle = this._getHeatColor(camVal, 0.95);
      } else {
        ctx.strokeStyle = '#06b6d4'; // Standard clinical cyan
      }
      ctx.stroke();
    }

    // 6. Draw Pan-Tompkins R-Peaks Markers
    if (this.showRPeaks && this.rPeaks && this.rPeaks.length > 0) {
      for (const p of this.rPeaks) {
        if (p >= startIdx && p <= endIdx) {
          const px = getX(p);
          const py = getY(sig[p]);

          // Diamond R-peak marker
          ctx.fillStyle = '#ef4444';
          ctx.beginPath();
          ctx.moveTo(px, py - 12);
          ctx.lineTo(px + 6, py - 6);
          ctx.lineTo(px, py);
          ctx.lineTo(px - 6, py - 6);
          ctx.closePath();
          ctx.fill();

          // Subtle label
          ctx.fillStyle = '#fca5a5';
          ctx.font = '9px monospace';
          ctx.textAlign = 'center';
          ctx.fillText('R', px, py - 15);
        }
      }
    }

    // 6.5 Draw Integrated Gradients / Saliency Lane if enabled
    if (this.showSaliency && this.saliency && this.saliency.length > 0) {
      const salH = 32;
      const salBaseY = h - 18;

      ctx.fillStyle = 'rgba(217, 70, 239, 0.06)';
      ctx.fillRect(0, salBaseY - salH, w, salH);

      ctx.strokeStyle = 'rgba(217, 70, 239, 0.25)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(0, salBaseY);
      ctx.lineTo(w, salBaseY);
      ctx.stroke();

      ctx.fillStyle = '#d946ef';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Integrated Gradients (Sample Saliency)', 10, salBaseY - salH + 11);

      ctx.beginPath();
      ctx.moveTo(getX(startIdx), salBaseY);
      for (let i = startIdx; i <= endIdx; i++) {
        const val = this.saliency[i] || 0;
        const sx = getX(i);
        const sy = salBaseY - val * (salH - 5);
        ctx.lineTo(sx, sy);
      }
      ctx.lineTo(getX(endIdx), salBaseY);
      ctx.closePath();
      ctx.fillStyle = 'rgba(217, 70, 239, 0.25)';
      ctx.fill();

      ctx.beginPath();
      for (let i = startIdx; i <= endIdx; i++) {
        const val = this.saliency[i] || 0;
        const sx = getX(i);
        const sy = salBaseY - val * (salH - 5);
        if (i === startIdx) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.strokeStyle = '#d946ef';
      ctx.lineWidth = 1.3;
      ctx.stroke();
    }

    // 7. Hover Cursor Line
    if (this.hoverIdx >= startIdx && this.hoverIdx <= endIdx) {
      const hx = getX(this.hoverIdx);
      const hy = getY(sig[this.hoverIdx]);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(hx, 0);
      ctx.lineTo(hx, h);
      ctx.stroke();
      ctx.setLineDash([]);

      // Point circle
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(hx, hy, 4.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  _drawEcgGrid(ctx, w, h) {
    // Minor Grid (1mm equivalent ~ 12px)
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.08)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (let x = 0; x < w; x += 15) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += 15) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    // Major Grid (5mm equivalent ~ 75px)
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.22)';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    for (let x = 0; x < w; x += 75) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += 75) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
  }

  renderSpectrum(spectrumData) {
    const canvas = document.getElementById('spectrumCanvas');
    if (!canvas || !spectrumData) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#070c18';
    ctx.fillRect(0, 0, w, h);

    const freqs = spectrumData.frequencies || [];
    const mags = spectrumData.magnitudes || [];
    if (freqs.length === 0 || mags.length === 0) return;

    const maxFreq = 50.0;
    const maxMag = Math.max(...mags, 1.0);

    // Grid lines & labels
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let f = 10; f <= 50; f += 10) {
      const x = (f / maxFreq) * (w - 60) + 40;
      ctx.beginPath();
      ctx.moveTo(x, 10);
      ctx.lineTo(x, h - 25);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${f} Hz`, x, h - 10);
    }

    // Passband zone (0.5 to 45 Hz)
    const pbStart = (0.5 / maxFreq) * (w - 60) + 40;
    const pbEnd = (45.0 / maxFreq) * (w - 60) + 40;
    ctx.fillStyle = 'rgba(6, 182, 212, 0.06)';
    ctx.fillRect(pbStart, 10, pbEnd - pbStart, h - 35);

    // Spectrum Curve
    const getX = (f) => (f / maxFreq) * (w - 60) + 40;
    const getY = (m) => (h - 25) - (m / maxMag) * (h - 45);

    ctx.beginPath();
    ctx.moveTo(getX(freqs[0]), h - 25);
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] <= maxFreq) {
        ctx.lineTo(getX(freqs[i]), getY(mags[i]));
      }
    }
    ctx.lineTo(getX(Math.min(maxFreq, freqs[freqs.length - 1])), h - 25);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(0, 10, 0, h - 25);
    gradient.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
    gradient.addColorStop(1, 'rgba(56, 189, 248, 0.02)');
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line stroke
    ctx.beginPath();
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] <= maxFreq) {
        const x = getX(freqs[i]);
        const y = getY(mags[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }
}

window.ECGVisualizer = ECGVisualizer;

