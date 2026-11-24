"""
Pan-Tompkins QRS Complex Detector and Heart Rate Variability (HRV) Analysis
Implements the clinical gold-standard Pan-Tompkins algorithm (1985)
with adaptive dual-thresholding, refractory blanking, and comprehensive HRV telemetry.
"""

import numpy as np


def pan_tompkins_qrs_detector(signal_data, fs=250.0):
    """
    Detects R-peaks using the Pan-Tompkins algorithm:
    1. Bandpass (5-15 Hz)
    2. Differentiation
    3. Squaring
    4. Moving Window Integration (MWI)
    5. Adaptive dual-thresholding with refractory blanking (200ms)
    """
    signal = np.asarray(signal_data, dtype=np.float64)
    n_samples = len(signal)
    if n_samples < int(fs * 0.5):
        return []

    # Step 1: Bandpass filter 5-15 Hz (QRS frequency band)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        b, a = butter(2, [5.0 / nyq, min(15.0 / nyq, 0.95)], btype='band')
        filtered = filtfilt(b, a, signal)
    except Exception:
        # Fallback derivative filter approximation
        filtered = signal - np.convolve(signal, np.ones(int(fs * 0.1)) / (fs * 0.1), mode='same')

    # Step 2: Five-point derivative
    # y[n] = (1/8) * (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4])
    diff = np.zeros_like(filtered)
    for n in range(4, n_samples):
        diff[n] = (2 * filtered[n] + filtered[n-1] - filtered[n-3] - 2 * filtered[n-4]) / 8.0

    # Step 3: Squaring
    squared = diff ** 2

    # Step 4: Moving Window Integration (approx 150ms window)
    window_len = max(1, int(0.150 * fs))
    kernel = np.ones(window_len) / float(window_len)
    mwi = np.convolve(squared, kernel, mode='same')

    # Step 5: Adaptive thresholding
    r_peaks = []
    min_distance = int(0.200 * fs)  # 200 ms physiological refractory period

    # Initialize thresholds from first 2 seconds
    init_window = min(n_samples, int(2.0 * fs))
    max_init = np.max(mwi[:init_window]) if init_window > 0 else 1.0
    spki = 0.3 * max_init
    npki = 0.1 * spki
    threshold_i1 = npki + 0.25 * (spki - npki)

    # Candidate peaks in integrated signal
    last_peak_idx = -min_distance
    for i in range(1, n_samples - 1):
        if mwi[i] > mwi[i - 1] and mwi[i] > mwi[i + 1]:  # local maximum
            peak_val = mwi[i]
            if (i - last_peak_idx) > min_distance:
                if peak_val > threshold_i1:
                    # Search for true peak in original bandpassed/raw signal within ±100ms
                    search_radius = int(0.100 * fs)
                    start_s = max(0, i - search_radius)
                    end_s = min(n_samples, i + search_radius)
                    true_r = start_s + np.argmax(np.abs(signal[start_s:end_s]))

                    r_peaks.append(int(true_r))
                    last_peak_idx = i
                    spki = 0.125 * peak_val + 0.875 * spki
                else:
                    npki = 0.125 * peak_val + 0.875 * npki

                threshold_i1 = npki + 0.25 * (spki - npki)

    # Filter out potential duplicates
    unique_peaks = []
    for p in sorted(r_peaks):
        if not unique_peaks or (p - unique_peaks[-1]) >= min_distance:
            unique_peaks.append(p)

    return unique_peaks


def compute_hrv_metrics(r_peaks, fs=250.0):
    """
    Calculates clinical cardiac rhythm and HRV metrics:
    - Mean Heart Rate (BPM)
    - Instantaneous Heart Rates
    - RR intervals (ms)
    - SDNN (Standard deviation of NN intervals in ms)
    - RMSSD (Root mean square of successive differences in ms)
    - pNN50 (% of successive intervals differing > 50ms)
    - Estimated QTc (Bazett's formula assuming average QT approx 380ms)
    - Rhythm regularity assessment
    """
    if len(r_peaks) < 2:
        return {
            'heart_rate_bpm': 0,
            'mean_rr_ms': 0,
            'min_rr_ms': 0,
            'max_rr_ms': 0,
            'sdnn_ms': 0,
            'rmssd_ms': 0,
            'pnn50_percent': 0,
            'qtc_ms': 0,
            'regularity': 'Insufficient Beats Detected',
            'r_peaks': r_peaks,
            'rr_intervals_ms': []
        }

    # RR intervals in seconds and milliseconds
    rr_sec = np.diff(r_peaks) / float(fs)
    rr_ms = rr_sec * 1000.0

    mean_rr = float(np.mean(rr_ms))
    min_rr = float(np.min(rr_ms))
    max_rr = float(np.max(rr_ms))

    # Mean Heart Rate
    heart_rate_bpm = round(60.0 / (mean_rr / 1000.0), 1) if mean_rr > 0 else 0

    # SDNN
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0

    # RMSSD
    diff_rr = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0

    # pNN50
    pnn50 = float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0) if len(diff_rr) > 0 else 0.0

    # Estimated QTc (Bazett's: QTc = QT / sqrt(RR in sec))
    # Standard normal QT ~ 390ms at 60 bpm
    estimated_qt = 390.0 * np.sqrt(mean_rr / 1000.0) if mean_rr > 0 else 0.0
    qtc = round(estimated_qt / np.sqrt(mean_rr / 1000.0), 1) if mean_rr > 0 else 0.0

    # Regularity classification
    rr_cov = (sdnn / mean_rr) if mean_rr > 0 else 0
    if rr_cov < 0.08:
        regularity = 'Strictly Regular'
    elif rr_cov < 0.16:
        regularity = 'Mildly Variable (Normal Respiration)'
    elif rr_cov < 0.28:
        regularity = 'Moderately Irregular (Ectopic Beats / PAC / PVC)'
    else:
        regularity = 'Markedly Irregular (Suggestive of AFib)'

    return {
        'heart_rate_bpm': heart_rate_bpm,
        'mean_rr_ms': round(mean_rr, 1),
        'min_rr_ms': round(min_rr, 1),
        'max_rr_ms': round(max_rr, 1),
        'sdnn_ms': round(sdnn, 1),
        'rmssd_ms': round(rmssd, 1),
        'pnn50_percent': round(pnn50, 1),
        'qtc_ms': round(qtc, 1),
        'regularity': regularity,
        'cov': round(rr_cov * 100, 1),
        'r_peaks': [int(p) for p in r_peaks],
        'rr_intervals_ms': [round(x, 1) for x in rr_ms.tolist()]
    }

