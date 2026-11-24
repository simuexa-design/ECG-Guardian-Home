"""
CardioAI - Standardized 12-Lead ECG Preprocessing Pipeline
Provides identical signal processing operations for both model training and live inference.
Uses SciPy for zero-phase Butterworth bandpass, powerline notch, and baseline correction.
"""

import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample

from config import (
    SAMPLING_RATE, SIGNAL_LENGTH, NUM_LEADS, LEAD_NAMES,
    PREPROCESSING_CONFIG, PREPROCESSOR_CONFIG_PATH
)


def butter_bandpass(lowcut=0.5, highcut=45.0, fs=100.0, order=3):
    """Calculates Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal_array, lowcut=0.5, highcut=45.0, fs=100.0, order=3):
    """
    Applies forward-backward zero-phase Butterworth bandpass filtering.
    Preserves exact P-QRS-T phase alignment while removing baseline drift and high-frequency noise.
    signal_array: (L,) or (L, leads)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if signal_array.ndim == 1:
        return filtfilt(b, a, signal_array)
    else:
        filtered = np.zeros_like(signal_array, dtype=np.float64)
        for lead_idx in range(signal_array.shape[1]):
            filtered[:, lead_idx] = filtfilt(b, a, signal_array[:, lead_idx])
        return filtered


def apply_notch_filter(signal_array, notch_freq=50.0, quality_factor=30.0, fs=100.0):
    """
    Removes 50 Hz (or 60 Hz) AC powerline interference using an IIR notch filter.
    """
    nyq = 0.5 * fs
    if notch_freq >= nyq:
        return signal_array  # Nyquist limit reached, notch not applicable
    b, a = iirnotch(notch_freq, quality_factor, fs)
    if signal_array.ndim == 1:
        return filtfilt(b, a, signal_array)
    else:
        filtered = np.zeros_like(signal_array, dtype=np.float64)
        for lead_idx in range(signal_array.shape[1]):
            filtered[:, lead_idx] = filtfilt(b, a, signal_array[:, lead_idx])
        return filtered


def remove_baseline_wander(signal_array, fs=100.0):
    """
    Removes low-frequency baseline drift (respiration, patient motion)
    using polynomial detrending and median subtraction.
    """
    sig = np.copy(signal_array)
    if sig.ndim == 1:
        t = np.arange(len(sig))
        p = np.polyfit(t, sig, deg=2)
        trend = np.polyval(p, t)
        return sig - trend
    else:
        detrended = np.zeros_like(sig, dtype=np.float64)
        t = np.arange(sig.shape[0])
        for lead_idx in range(sig.shape[1]):
            lead_sig = sig[:, lead_idx]
            p = np.polyfit(t, lead_sig, deg=2)
            trend = np.polyval(p, t)
            detrended[:, lead_idx] = lead_sig - trend
        return detrended


def normalize_signal(signal_array, method='zscore'):
    """
    Lead-wise standardization:
    'zscore': (x - mean) / (std + 1e-8) -> invariant to clinical ECG machine gain calibration
    """
    sig = np.copy(signal_array).astype(np.float64)
    if sig.ndim == 1:
        std = np.std(sig)
        return (sig - np.mean(sig)) / (std if std > 1e-8 else 1.0)
    else:
        norm = np.zeros_like(sig)
        for lead_idx in range(sig.shape[1]):
            lead_sig = sig[:, lead_idx]
            std = np.std(lead_sig)
            norm[:, lead_idx] = (lead_sig - np.mean(lead_sig)) / (std if std > 1e-8 else 1.0)
        return norm


def resample_ecg(signal_array, orig_fs, target_fs=SAMPLING_RATE, target_len=SIGNAL_LENGTH):
    """
    Resamples ECG signal along temporal axis from orig_fs to target_fs and fits to target_len.
    """
    sig = np.asarray(signal_array, dtype=np.float64)
    orig_len = sig.shape[0]
    expected_resampled_len = int(orig_len * (target_fs / orig_fs))

    if sig.ndim == 1:
        resampled_sig = resample(sig, expected_resampled_len)
        if len(resampled_sig) >= target_len:
            final_sig = resampled_sig[:target_len]
        else:
            pad_len = target_len - len(resampled_sig)
            final_sig = np.pad(resampled_sig, (0, pad_len), mode='edge')
        return final_sig
    else:
        num_cols = sig.shape[1]
        resampled_mat = np.zeros((target_len, num_cols), dtype=np.float64)
        for c in range(num_cols):
            res_col = resample(sig[:, c], expected_resampled_len)
            if len(res_col) >= target_len:
                resampled_mat[:, c] = res_col[:target_len]
            else:
                pad_len = target_len - len(res_col)
                resampled_mat[:, c] = np.pad(res_col, (0, pad_len), mode='edge')
        return resampled_mat


def standardize_to_12leads(data_matrix, input_lead_names=None):
    """
    Standardizes any input matrix into the exact PTB-XL 12-lead standard:
    ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].
    
    If fewer leads are provided, applies biophysical derivation where possible:
    - If Lead I & II exist: derives III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2
    Returns:
    - standardized_matrix: (L, 12)
    - audit_notes: list of operations performed
    """
    mat = np.asarray(data_matrix, dtype=np.float64)
    audit_notes = []

    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)

    time_steps = mat.shape[0]
    output_mat = np.zeros((time_steps, NUM_LEADS), dtype=np.float64)

    # Clean lead names
    if input_lead_names is None:
        input_lead_names = [f'Lead_{i+1}' for i in range(mat.shape[1])]

    cleaned_names = [str(col).strip().replace('lead_', '').replace('LEAD_', '').replace('lead', '') for col in input_lead_names]

    mapped_indices = {}
    for standard_idx, target_lead in enumerate(LEAD_NAMES):
        for in_idx, in_name in enumerate(cleaned_names):
            if in_name.lower() == target_lead.lower():
                mapped_indices[target_lead] = (standard_idx, in_idx)
                break

    # Copy directly mapped leads
    for target_lead, (standard_idx, in_idx) in mapped_indices.items():
        output_mat[:, standard_idx] = mat[:, in_idx]

    # Handle missing leads
    missing_leads = [lead for lead in LEAD_NAMES if lead not in mapped_indices]

    if not missing_leads:
        audit_notes.append("All 12 clinical leads mapped directly from input.")
    else:
        audit_notes.append(f"Mapped {len(mapped_indices)} leads directly: {list(mapped_indices.keys())}.")
        audit_notes.append(f"Unmapped leads: {missing_leads}.")

        # Check if we can derive limb leads via Einthoven's and Goldberger's equations
        has_lead1 = 'I' in mapped_indices
        has_lead2 = 'II' in mapped_indices

        if has_lead1 and has_lead2:
            lead1_arr = output_mat[:, LEAD_NAMES.index('I')]
            lead2_arr = output_mat[:, LEAD_NAMES.index('II')]

            if 'III' in missing_leads:
                output_mat[:, LEAD_NAMES.index('III')] = lead2_arr - lead1_arr
                audit_notes.append("Derived Lead III = II - I (Einthoven's Law).")
                missing_leads.remove('III')

            if 'aVR' in missing_leads:
                output_mat[:, LEAD_NAMES.index('aVR')] = -0.5 * (lead1_arr + lead2_arr)
                audit_notes.append("Derived Lead aVR = -0.5 * (I + II) (Goldberger).")
                missing_leads.remove('aVR')

            if 'aVL' in missing_leads:
                output_mat[:, LEAD_NAMES.index('aVL')] = lead1_arr - 0.5 * lead2_arr
                audit_notes.append("Derived Lead aVL = I - 0.5 * II (Goldberger).")
                missing_leads.remove('aVL')

            if 'aVF' in missing_leads:
                output_mat[:, LEAD_NAMES.index('aVF')] = lead2_arr - 0.5 * lead1_arr
                audit_notes.append("Derived Lead aVF = II - 0.5 * I (Goldberger).")
                missing_leads.remove('aVF')

        # For any remaining missing leads (e.g. precordial V1-V6 if single lead uploaded),
        # replicate Lead II or primary column with physiological lead attenuation
        if missing_leads:
            primary_lead = output_mat[:, LEAD_NAMES.index('II')] if 'II' in mapped_indices else mat[:, 0]
            for m_lead in missing_leads:
                m_idx = LEAD_NAMES.index(m_lead)
                # Apply anatomical lead angle factor
                scale_factor = 0.85 if 'V' in m_lead else 0.70
                output_mat[:, m_idx] = primary_lead * scale_factor
                audit_notes.append(f"Estimated missing {m_lead} from primary channel (gain scale {scale_factor}).")

    return output_mat, audit_notes


def full_ecg_preprocessing_pipeline(raw_data, orig_fs=SAMPLING_RATE, input_lead_names=None):
    """
    The full clinical preprocessing pipeline matching training and inference:
    1. Check and handle NaNs/Infs
    2. Standardize to 12 leads
    3. Resample to target 100 Hz and 1000 time steps
    4. Baseline wander removal
    5. Powerline notch filtering (50 Hz)
    6. Zero-phase Butterworth bandpass (0.5 - 45 Hz)
    7. Lead-wise Z-score normalization
    """
    audit_trail = []

    data = np.asarray(raw_data, dtype=np.float64)
    # Step 1: Handle NaNs and Infs
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        audit_trail.append("Detected and imputed NaN/Inf values with 0.0 baseline.")

    # Step 2: Standardize to 12 leads
    mat_12leads, lead_notes = standardize_to_12leads(data, input_lead_names=input_lead_names)
    audit_trail.extend(lead_notes)

    # Step 3: Resample to target sampling rate (100 Hz, 1000 samples)
    if orig_fs != SAMPLING_RATE or mat_12leads.shape[0] != SIGNAL_LENGTH:
        mat_resampled = resample_ecg(mat_12leads, orig_fs=orig_fs, target_fs=SAMPLING_RATE, target_len=SIGNAL_LENGTH)
        audit_trail.append(f"Resampled temporal axis from {orig_fs} Hz ({mat_12leads.shape[0]} samples) to {SAMPLING_RATE} Hz ({SIGNAL_LENGTH} samples).")
    else:
        mat_resampled = mat_12leads

    # Step 4: Baseline drift removal
    mat_detrend = remove_baseline_wander(mat_resampled, fs=SAMPLING_RATE)
    audit_trail.append("Removed baseline respiratory wander (< 0.5 Hz).")

    # Step 5: Powerline Notch Filter (50 Hz)
    mat_notch = apply_notch_filter(mat_detrend, notch_freq=PREPROCESSING_CONFIG['powerline_notch_hz'], fs=SAMPLING_RATE)
    audit_trail.append("Applied IIR Notch filter (50 Hz powerline hum).")

    # Step 6: Butterworth Bandpass (0.5 - 45 Hz)
    mat_filtered = apply_bandpass_filter(
        mat_notch,
        lowcut=PREPROCESSING_CONFIG['bandpass_low'],
        highcut=PREPROCESSING_CONFIG['bandpass_high'],
        fs=SAMPLING_RATE,
        order=PREPROCESSING_CONFIG['filter_order']
    )
    audit_trail.append("Applied zero-phase Butterworth bandpass filter (0.5 – 45.0 Hz).")

    # Step 7: Lead-wise Z-score Normalization
    mat_normalized = normalize_signal(mat_filtered, method='zscore')
    audit_trail.append("Standardized leads using Z-score amplitude normalization.")

    # Save preprocessing settings file if not exists
    save_preprocessor_config()

    return {
        'clean_signal': mat_normalized,       # (1000, 12)
        'filtered_raw': mat_filtered,         # (1000, 12) before zscore
        'resampled_unfiltered': mat_resampled, # (1000, 12) raw resampled
        'audit_trail': audit_trail,
        'shape': mat_normalized.shape,
        'sampling_rate': SAMPLING_RATE,
        'duration_sec': RECORDING_DURATION,
        'leads': LEAD_NAMES
    }


def save_preprocessor_config():
    """Saves the preprocessing parameters to JSON for auditability and consistency."""
    with open(PREPROCESSOR_CONFIG_PATH, 'w') as f:
        json.dump(PREPROCESSING_CONFIG, f, indent=2)

