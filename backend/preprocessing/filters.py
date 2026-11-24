"""
Advanced ECG Signal Filtering and Preprocessing Pipeline
Includes Butterworth Bandpass, Notch Filtering, Baseline Wander Removal,
and Z-score normalization for clinical electrocardiogram signals.
"""

import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def butter_bandpass(lowcut=0.5, highcut=45.0, fs=250.0, order=3):
    """Generates Butterworth bandpass filter coefficients."""
    if not HAS_SCIPY:
        return None, None
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal_data, lowcut=0.5, highcut=45.0, fs=250.0, order=3):
    """
    Applies zero-phase Butterworth bandpass filter to preserve wave morphology.
    Removes low-frequency baseline drift (< 0.5 Hz) and high-frequency EMG noise (> 45 Hz).
    """
    signal_arr = np.asarray(signal_data, dtype=np.float64)
    if not HAS_SCIPY:
        # Simple moving average baseline subtraction + smoothing fallback
        baseline = np.convolve(signal_arr, np.ones(int(fs)) / fs, mode='same')
        detrended = signal_arr - baseline
        smoothed = np.convolve(detrended, np.ones(3) / 3.0, mode='same')
        return smoothed

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt provides forward-backward zero-phase filtering (no phase distortion)
    filtered = filtfilt(b, a, signal_arr)
    return filtered


def apply_notch_filter(signal_data, notch_freq=50.0, quality_factor=30.0, fs=250.0):
    """
    Removes 50 Hz or 60 Hz AC powerline interference using an IIR notch filter.
    """
    signal_arr = np.asarray(signal_data, dtype=np.float64)
    if not HAS_SCIPY:
        return signal_arr

    nyq = 0.5 * fs
    if notch_freq >= nyq:
        return signal_arr

    b, a = iirnotch(notch_freq, quality_factor, fs)
    filtered = filtfilt(b, a, signal_arr)
    return filtered


def remove_baseline_wander(signal_data, fs=250.0):
    """
    Removes respiratory baseline wander using double median filtering
    (200ms window for P-QRS-T complexes, followed by 600ms window for ST-T).
    """
    signal_arr = np.asarray(signal_data, dtype=np.float64)
    win1 = int(0.2 * fs)
    if win1 % 2 == 0:
        win1 += 1
    win2 = int(0.6 * fs)
    if win2 % 2 == 0:
        win2 += 1

    try:
        from scipy.ndimage import median_filter
        baseline1 = median_filter(signal_arr, size=win1, mode='nearest')
        baseline2 = median_filter(baseline1, size=win2, mode='nearest')
        return signal_arr - baseline2
    except Exception:
        # Fallback to linear detrend
        x = np.arange(len(signal_arr))
        p = np.polyfit(x, signal_arr, deg=2)
        trend = np.polyval(p, x)
        return signal_arr - trend


def normalize_signal(signal_data, method='zscore'):
    """
    Standardizes ECG amplitude.
    'zscore': (x - mean) / std (robust to gain variations across ECG machines)
    'minmax': scales to [-1, 1] range
    """
    signal_arr = np.asarray(signal_data, dtype=np.float64)
    if method == 'zscore':
        std = np.std(signal_arr)
        if std < 1e-8:
            std = 1.0
        return (signal_arr - np.mean(signal_arr)) / std
    elif method == 'minmax':
        min_v = np.min(signal_arr)
        max_v = np.max(signal_arr)
        if max_v - min_v < 1e-8:
            return signal_arr - min_v
        return 2.0 * (signal_arr - min_v) / (max_v - min_v) - 1.0
    return signal_arr


def full_ecg_preprocessing(raw_signal, fs=250.0, powerline_hz=50.0):
    """
    Executes the standard multi-stage medical ECG preprocessing pipeline:
    1. Baseline wander removal
    2. Powerline notch filtering (50 Hz / 60 Hz)
    3. Butterworth bandpass (0.5 - 45 Hz)
    4. Lead normalization
    """
    raw_arr = np.asarray(raw_signal, dtype=np.float64)
    # Stage 1: Baseline removal
    stage1 = remove_baseline_wander(raw_arr, fs=fs)
    # Stage 2: Powerline notch
    stage2 = apply_notch_filter(stage1, notch_freq=powerline_hz, fs=fs)
    # Stage 3: Bandpass
    stage3 = apply_bandpass_filter(stage2, lowcut=0.5, highcut=45.0, fs=fs, order=3)
    # Stage 4: Normalization
    clean_signal = normalize_signal(stage3, method='zscore')

    return {
        'raw': raw_arr.tolist(),
        'clean': clean_signal.tolist(),
        'fs': float(fs),
        'duration_sec': round(len(raw_arr) / fs, 2),
        'num_samples': len(raw_arr)
    }

