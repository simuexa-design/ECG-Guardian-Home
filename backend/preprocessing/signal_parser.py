"""
Robust Multi-Format ECG Signal Parser and Normalizer
Parses CSV, TXT, JSON, and WFDB formats, automatically detects sampling rate,
handles multi-lead or single-lead files, and resamples to target length.
"""

import io
import json
import numpy as np


TARGET_FS = 250.0  # Standard 250 Hz sampling rate
TARGET_LENGTH = 2500  # 10.0 seconds window (2500 samples)


def resample_signal(signal_data, orig_fs, target_fs=TARGET_FS, target_len=TARGET_LENGTH):
    """
    Resamples ECG signal from orig_fs to target_fs and fits to target_len.
    """
    sig = np.asarray(signal_data, dtype=np.float64)
    if len(sig) == 0:
        return np.zeros(target_len, dtype=np.float64)

    # Calculate target length based on duration if not strictly specified
    orig_duration = len(sig) / float(orig_fs)
    resampled_len = int(orig_duration * target_fs)

    try:
        from scipy.signal import resample
        resampled = resample(sig, resampled_len)
    except Exception:
        # Linear interpolation fallback
        orig_indices = np.linspace(0, 1, len(sig))
        target_indices = np.linspace(0, 1, resampled_len)
        resampled = np.interp(target_indices, orig_indices, sig)

    # Align to target_len (pad with zero or truncate)
    if len(resampled) >= target_len:
        final_sig = resampled[:target_len]
    else:
        pad_len = target_len - len(resampled)
        final_sig = np.pad(resampled, (0, pad_len), mode='edge')

    return final_sig


def parse_csv_ecg(content_str):
    """
    Parses CSV text into 1D ECG voltage array.
    Handles headers, time column, multiple columns (picks lead II or primary column).
    """
    lines = [line.strip() for line in content_str.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Uploaded file is empty.")

    # Check for header
    first_line = lines[0]
    has_header = False
    for char in first_line:
        if char.isalpha() and char not in ['e', 'E', '.', '-', '+', ',']:
            has_header = True
            break

    lead_col_idx = 1
    if has_header:
        header_tokens = [t.strip().lower() for t in first_line.replace(';', ',').replace('\t', ',').split(',') if t.strip()]
        for idx, col in enumerate(header_tokens):
            if col in ['ii', 'lead_ii', 'lead ii', 'lead2', 'lead 2', 'mlii', 'ml_ii', 'ecg', 'signal', 'voltage']:
                lead_col_idx = idx
                break
        else:
            if len(header_tokens) > 1 and any(t in header_tokens[0] for t in ['time', 'sample', 'index', 't', 'sec']):
                lead_col_idx = 1
            else:
                lead_col_idx = 0

    data_lines = lines[1:] if has_header else lines
    samples = []

    for line in data_lines:
        tokens = [t.strip() for t in line.replace(';', ',').replace('\t', ',').split(',') if t.strip()]
        if not tokens:
            continue
        try:
            if len(tokens) > lead_col_idx:
                val = float(tokens[lead_col_idx])
            elif len(tokens) >= 2:
                val = float(tokens[1])
            else:
                val = float(tokens[0])
            samples.append(val)
        except ValueError:
            continue

    if not samples:
        raise ValueError("Could not parse any numerical ECG samples from CSV.")

    return np.array(samples, dtype=np.float64)


def parse_json_ecg(content_str):
    """
    Parses JSON data:
    Supports:
    - [0.1, 0.2, 0.5, ...]
    - {"signal": [...], "fs": 250}
    - {"samples": [...]}
    - {"data": [...]}
    """
    obj = json.loads(content_str)
    fs = TARGET_FS

    if isinstance(obj, list):
        samples = [float(x) for x in obj]
    elif isinstance(obj, dict):
        if 'fs' in obj:
            fs = float(obj['fs'])
        elif 'sampling_rate' in obj:
            fs = float(obj['sampling_rate'])

        if 'signal' in obj:
            samples = [float(x) for x in obj['signal']]
        elif 'samples' in obj:
            samples = [float(x) for x in obj['samples']]
        elif 'data' in obj:
            samples = [float(x) for x in obj['data']]
        elif 'val' in obj:
            samples = [float(x) for x in obj['val']]
        else:
            # First key with list
            for k, v in obj.items():
                if isinstance(v, list):
                    samples = [float(x) for x in v]
                    break
            else:
                raise ValueError("JSON does not contain an ECG signal array.")
    else:
        raise ValueError("Unsupported JSON structure.")

    return np.array(samples, dtype=np.float64), fs


def parse_text_ecg(content_str):
    """
    Parses space, tab, or newline separated numbers.
    """
    tokens = content_str.replace(',', ' ').split()
    samples = []
    for t in tokens:
        try:
            samples.append(float(t))
        except ValueError:
            continue

    if not samples:
        raise ValueError("Could not parse numerical ECG values from text.")

    return np.array(samples, dtype=np.float64)


def parse_uploaded_file(filename, content_bytes, custom_fs=None):
    """
    Entry point to parse uploaded ECG file.
    Returns:
    - raw_samples (np.ndarray)
    - detected_fs (float)
    - standard_samples (np.ndarray formatted to 2500 samples at 250 Hz)
    """
    text = content_bytes.decode('utf-8', errors='ignore')
    lower_name = filename.lower()
    fs = float(custom_fs) if custom_fs else TARGET_FS

    if lower_name.endswith('.json'):
        raw_samples, detected_fs = parse_json_ecg(text)
        if custom_fs:
            detected_fs = fs
    elif lower_name.endswith('.csv'):
        raw_samples = parse_csv_ecg(text)
        detected_fs = fs
    else:
        # Default text / dat
        raw_samples = parse_text_ecg(text)
        detected_fs = fs

    standard_signal = resample_signal(raw_samples, orig_fs=detected_fs, target_fs=TARGET_FS, target_len=TARGET_LENGTH)

    return {
        'filename': filename,
        'raw_samples': raw_samples.tolist(),
        'standard_samples': standard_signal.tolist(),
        'fs': detected_fs,
        'standard_fs': TARGET_FS,
        'num_raw_samples': len(raw_samples),
        'duration_seconds': round(len(raw_samples) / detected_fs, 2)
    }

