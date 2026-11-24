"""
Physiologically Accurate Clinical ECG Generator and Benchmark Catalog
Generates realistic multi-label cardiac signals using the biophysical P-QRS-T
Gaussian dipole model with autonomic respiration, heart rate variability,
pathological bundle branch blocks, ischemia ST-elevation, and ectopic beats.
"""

import math
import numpy as np


def generate_single_beat(beat_type='normal', hr=72.0, fs=250.0, st_elev=0.0):
    """
    Synthesizes a single cardiac P-QRS-T complex using Gaussian dipole components.
    beat_type: 'normal', 'pvc', 'pac', 'lbbb', 'rbbb'
    """
    rr_sec = 60.0 / hr
    beat_samples = int(rr_sec * fs)
    t = np.linspace(0, rr_sec, beat_samples, endpoint=False)
    # Center R-peak at ~40% into the RR cycle
    r_center = 0.40 * rr_sec

    voltage = np.zeros_like(t)

    if beat_type == 'normal':
        # P-wave
        p_t = r_center - 0.16
        voltage += 0.15 * np.exp(-((t - p_t) ** 2) / (2 * (0.025 ** 2)))
        # Q-wave
        q_t = r_center - 0.04
        voltage -= 0.15 * np.exp(-((t - q_t) ** 2) / (2 * (0.010 ** 2)))
        # R-wave (sharp positive deflection)
        voltage += 1.20 * np.exp(-((t - r_center) ** 2) / (2 * (0.014 ** 2)))
        # S-wave (sharp negative deflection)
        s_t = r_center + 0.04
        voltage -= 0.25 * np.exp(-((t - s_t) ** 2) / (2 * (0.012 ** 2)))
        # ST segment shift + T-wave
        st_t = r_center + 0.15
        t_wave_t = r_center + 0.26
        voltage += st_elev * np.exp(-((t - st_t) ** 2) / (2 * (0.050 ** 2)))
        voltage += (0.30 - 0.5 * st_elev) * np.exp(-((t - t_wave_t) ** 2) / (2 * (0.055 ** 2)))

    elif beat_type == 'lbbb':
        # Left Bundle Branch Block: prolonged QRS (>120ms), slurred notched R wave, absent Q
        # P-wave
        p_t = r_center - 0.18
        voltage += 0.14 * np.exp(-((t - p_t) ** 2) / (2 * (0.025 ** 2)))
        # Slurred R peak 1
        voltage += 0.95 * np.exp(-((t - (r_center - 0.025)) ** 2) / (2 * (0.035 ** 2)))
        # Slurred R peak 2 (M-shaped notch)
        voltage += 1.05 * np.exp(-((t - (r_center + 0.030)) ** 2) / (2 * (0.038 ** 2)))
        # Broad S wave
        voltage -= 0.35 * np.exp(-((t - (r_center + 0.090)) ** 2) / (2 * (0.040 ** 2)))
        # Discordant inverted T wave
        t_wave_t = r_center + 0.28
        voltage -= 0.28 * np.exp(-((t - t_wave_t) ** 2) / (2 * (0.065 ** 2)))

    elif beat_type == 'rbbb':
        # Right Bundle Branch Block: rSR' pattern (rabbit ears)
        p_t = r_center - 0.16
        voltage += 0.14 * np.exp(-((t - p_t) ** 2) / (2 * (0.025 ** 2)))
        # Initial small r wave
        voltage += 0.55 * np.exp(-((t - (r_center - 0.030)) ** 2) / (2 * (0.015 ** 2)))
        # Intermediate S deflection
        voltage -= 0.40 * np.exp(-((t - r_center) ** 2) / (2 * (0.018 ** 2)))
        # Tall second R' wave
        voltage += 1.15 * np.exp(-((t - (r_center + 0.045)) ** 2) / (2 * (0.028 ** 2)))
        # T wave
        voltage -= 0.22 * np.exp(-((t - (r_center + 0.25)) ** 2) / (2 * (0.055 ** 2)))

    elif beat_type == 'pvc':
        # Premature Ventricular Contraction: No preceding P-wave, wide bizarre QRS, large discordant T-wave
        voltage -= 0.45 * np.exp(-((t - (r_center - 0.040)) ** 2) / (2 * (0.035 ** 2)))
        voltage += 1.45 * np.exp(-((t - (r_center + 0.020)) ** 2) / (2 * (0.050 ** 2)))
        voltage -= 0.85 * np.exp(-((t - (r_center + 0.090)) ** 2) / (2 * (0.045 ** 2)))
        # Large opposite T wave
        voltage -= 0.55 * np.exp(-((t - (r_center + 0.28)) ** 2) / (2 * (0.075 ** 2)))

    elif beat_type == 'pac':
        # Premature Atrial Contraction: Abnormal early P wave, followed by normal narrow QRS
        p_t = r_center - 0.13
        voltage += 0.32 * np.exp(-((t - p_t) ** 2) / (2 * (0.022 ** 2)))
        voltage -= 0.12 * np.exp(-((t - (r_center - 0.035)) ** 2) / (2 * (0.010 ** 2)))
        voltage += 1.15 * np.exp(-((t - r_center) ** 2) / (2 * (0.014 ** 2)))
        voltage -= 0.22 * np.exp(-((t - (r_center + 0.035)) ** 2) / (2 * (0.012 ** 2)))
        voltage += 0.28 * np.exp(-((t - (r_center + 0.24)) ** 2) / (2 * (0.050 ** 2)))

    return voltage


def generate_clinical_record(labels, duration_sec=10.0, fs=250.0, seed=None):
    """
    Generates a full 10-second ECG recording matching target multi-label profile.
    labels: list of class IDs, e.g. ['LBBB', 'STACH'] or ['AFIB'] or ['NORM']
    """
    if seed is not None:
        np.random.seed(seed)

    total_samples = int(duration_sec * fs)
    ecg = np.zeros(total_samples, dtype=np.float64)

    is_afib = 'AFIB' in labels
    is_afl = 'AFL' in labels
    is_stach = 'STACH' in labels
    is_sbrad = 'SBRAD' in labels
    is_lbbb = 'LBBB' in labels
    is_rbbb = 'RBBB' in labels
    has_pvc = 'PVC' in labels
    has_pac = 'PAC' in labels
    has_mi = 'MI' in labels

    # Target heart rate
    if is_stach:
        base_hr = np.random.uniform(108.0, 130.0)
    elif is_sbrad:
        base_hr = np.random.uniform(46.0, 56.0)
    elif is_afib:
        base_hr = np.random.uniform(105.0, 135.0)  # AFib with rapid ventricular response
    else:
        base_hr = np.random.uniform(68.0, 82.0)

    # Base RR intervals
    t_pointer = 0.1  # start offset
    beat_idx = 0

    while t_pointer < duration_sec:
        # Determine this beat's RR interval
        if is_afib:
            # Irregularly irregular RR intervals (Poisson-like variation)
            rr_var = np.random.exponential(scale=0.15) - 0.05
            current_hr = np.clip(base_hr + rr_var * 40.0, 60.0, 160.0)
        else:
            # Normal physiological respiratory sinus arrhythmia
            rsa = 4.0 * math.sin(2.0 * math.pi * 0.25 * t_pointer)
            current_hr = base_hr + rsa

        # Decide beat type
        beat_type = 'normal'
        if is_lbbb:
            beat_type = 'lbbb'
        elif is_rbbb:
            beat_type = 'rbbb'

        # Interspersed ectopic beats
        if has_pvc and (beat_idx % 4 == 2 or (beat_idx == 1 and np.random.rand() > 0.4)):
            beat_type = 'pvc'
        elif has_pac and (beat_idx % 4 == 1):
            beat_type = 'pac'

        st_elev = 0.35 if has_mi else 0.0
        beat_signal = generate_single_beat(beat_type=beat_type, hr=current_hr, fs=fs, st_elev=st_elev)

        start_samp = int(t_pointer * fs)
        end_samp = start_samp + len(beat_signal)

        if end_samp <= total_samples:
            ecg[start_samp:end_samp] += beat_signal
        else:
            avail = total_samples - start_samp
            if avail > 0:
                ecg[start_samp:] += beat_signal[:avail]
            break

        # Advance pointer
        step_rr = (60.0 / current_hr)
        if beat_type == 'pvc':
            step_rr *= 1.35  # Compensatory pause after PVC
        elif beat_type == 'pac':
            step_rr *= 0.85  # Premature early step
        t_pointer += step_rr
        beat_idx += 1

    # Atrial baseline characteristics
    time_axis = np.linspace(0, duration_sec, total_samples, endpoint=False)
    if is_afib:
        # Fibrillatory waves (350-600 /min = ~6-9 Hz chaotic baseline)
        f_waves = (
            0.08 * np.sin(2 * np.pi * 6.5 * time_axis + np.random.rand()) +
            0.06 * np.sin(2 * np.pi * 8.2 * time_axis + np.random.rand()) +
            0.04 * np.random.randn(total_samples)
        )
        ecg += f_waves
    elif is_afl:
        # Classic sawtooth flutter baseline (~4.5 Hz / 270 bpm)
        flutter = 0.18 * (2.0 * (time_axis * 4.5 - np.floor(time_axis * 4.5 + 0.5)))
        ecg += flutter

    # Realistic physiological noise
    # 1. Subtle baseline respiration drift (0.2 Hz)
    resp_drift = 0.12 * np.sin(2 * np.pi * 0.22 * time_axis)
    # 2. 50 Hz powerline hum
    powerline = 0.03 * np.sin(2 * np.pi * 50.0 * time_axis)
    # 3. High-frequency EMG muscle tremor
    emg_noise = 0.02 * np.random.randn(total_samples)

    raw_ecg = ecg + resp_drift + powerline + emg_noise
    return raw_ecg


# Standard Clinical Benchmark Cases
BENCHMARK_CASES = [
    {
        'id': 'case_01_normal_sinus',
        'title': 'Normal Sinus Rhythm (NSR)',
        'patient': '42 yo Male, Routine Health Screening',
        'labels': ['NORM'],
        'description': 'Normal sinus rhythm at 72 bpm, uniform P-QRS-T complexes, PR interval 160ms, QRS 85ms.',
        'clinical_notes': 'Cardiologically benign baseline recording with physiological heart rate variability.'
    },
    {
        'id': 'case_02_atrial_fibrillation',
        'title': 'Atrial Fibrillation with RVR',
        'patient': '67 yo Female, Palpitations & Dyspnea',
        'labels': ['AFIB', 'STACH'],
        'description': 'Irregularly irregular ventricular response (HR 122 bpm) with absence of distinct P waves and fine fibrillatory baseline.',
        'clinical_notes': 'High thromboembolic stroke risk. Anticoagulation assessment and ventricular rate control required.'
    },
    {
        'id': 'case_03_lbbb_tachycardia',
        'title': 'Left Bundle Branch Block + Sinus Tachycardia',
        'patient': '58 yo Male, Heart Failure History',
        'labels': ['LBBB', 'STACH'],
        'description': 'Sinus tachycardia (112 bpm) with widened slurred M-shaped QRS complex (> 145 ms) indicative of delayed left ventricular conduction.',
        'clinical_notes': 'Indicates structural heart disease or dilated cardiomyopathy; masks ischemic ST changes.'
    },
    {
        'id': 'case_04_rbbb_pvc',
        'title': 'Right Bundle Branch Block with Frequent PVCs',
        'patient': '71 yo Male, Syncope Evaluation',
        'labels': ['RBBB', 'PVC'],
        'description': 'RBBB with classic rSR\' "rabbit ear" pattern in precordial leads accompanied by premature ventricular ectopic complexes.',
        'clinical_notes': 'Ectopic ventricular activity with conduction defect. Requires evaluation for structural right ventricular overload.'
    },
    {
        'id': 'case_05_st_elevation_mi',
        'title': 'Acute Myocardial Infarction + Bradycardia',
        'patient': '63 yo Female, Crushing Retrosternal Chest Pain',
        'labels': ['MI', 'SBRAD'],
        'description': 'Severe ST-segment elevation (> 3.5 mm) with hyperacute T waves and sinus bradycardia (52 bpm).',
        'clinical_notes': 'EMERGENCY: Transmural inferior/anterior myocardial infarction. Immediate catheterization lab activation indicated.'
    },
    {
        'id': 'case_06_afib_pvc',
        'title': 'Atrial Fibrillation with Frequent PVCs',
        'patient': '75 yo Male, Chronic AFib on Digoxin',
        'labels': ['AFIB', 'PVC'],
        'description': 'Underlying irregular atrial fibrillation complicated by multifocal premature ventricular beats with compensatory pauses.',
        'clinical_notes': 'Complex multi-arrhythmia state. Rule out digitalis toxicity or hypokalemia.'
    },
    {
        'id': 'case_07_willy_mckee_infarction',
        'title': 'Willy McKee (61yo M): Anteroseptal Infarction',
        'patient': 'Willy McKee (ID 123123), 61yo M',
        'labels': ['NORM', 'MI'],
        'description': 'Clinical 12-Lead Report: Normal sinus rhythm (80 bpm, PR 186ms, QRS 100ms, QTc 416ms) with Anteroseptal Infarction & Ischemic ST-T Changes.',
        'clinical_notes': 'Patient Report: 004 Normal sinus rhythm, 193 Anteroseptal infarction (probably old), 239 Ischemic ST-T changes in posterior leads. Deep Q in V1-V2 (66-68ms) with ST-T ischemic alterations.'
    }
]

