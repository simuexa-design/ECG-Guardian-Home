"""
CardioAI - Configuration File
Standardized parameters for PTB-XL ECG preprocessing,
Deep Learning model architecture, and Explainable AI.
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ptbxl_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_files')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ECG Signal Properties (PTB-XL Standard)
SAMPLING_RATE = 100       # Target sampling rate: 100 Hz
RECORDING_DURATION = 10   # 10.0 seconds window
SIGNAL_LENGTH = SAMPLING_RATE * RECORDING_DURATION  # 1000 time steps
NUM_LEADS = 12

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Standard 12-lead limb lead derivation relations (Einthoven / Goldberger)
# III = II - I
# aVR = -(I + II)/2
# aVL = I - II/2
# aVF = II - I/2

# Preprocessing Parameters
PREPROCESSING_CONFIG = {
    'sampling_rate': SAMPLING_RATE,
    'signal_length': SIGNAL_LENGTH,
    'num_leads': NUM_LEADS,
    'lead_names': LEAD_NAMES,
    'bandpass_low': 0.5,       # Highpass cutoff for baseline drift (Hz)
    'bandpass_high': 45.0,     # Lowpass cutoff for EMG / high frequency noise (Hz)
    'filter_order': 3,
    'powerline_notch_hz': 50.0, # 50 Hz powerline hum removal
    'quality_factor': 30.0,
    'normalization': 'zscore'   # Lead-wise standardization: (x - mean) / std
}

# PTB-XL Selected Rhythm-Related Arrhythmia Classes
# Filtered from scp_statements.csv (rhythm statements & diagnostic arrhythmias)
RHYTHM_CLASSES = [
    {
        'code': 'SR',
        'name': 'Sinus Rhythm',
        'category': 'Normal / Pacemaker',
        'description': 'Normal physiological cardiac rhythm originating from the sinoatrial node (60–100 bpm).'
    },
    {
        'code': 'AFIB',
        'name': 'Atrial Fibrillation',
        'category': 'Supraventricular Arrhythmia',
        'description': 'Rapid, chaotic atrial depolarizations with irregularly irregular ventricular intervals and absent P-waves.'
    },
    {
        'code': 'STACH',
        'name': 'Sinus Tachycardia',
        'category': 'Sinus Rhythm Variant',
        'description': 'Accelerated normal sinus rhythm with resting heart rate strictly exceeding 100 beats per minute.'
    },
    {
        'code': 'SBRAD',
        'name': 'Sinus Bradycardia',
        'category': 'Sinus Rhythm Variant',
        'description': 'Slow normal sinus rhythm with resting heart rate below 60 beats per minute.'
    },
    {
        'code': 'AFLT',
        'name': 'Atrial Flutter',
        'category': 'Supraventricular Arrhythmia',
        'description': 'Organized macro-reentrant atrial tachycardia producing characteristic continuous sawtooth flutter waves.'
    },
    {
        'code': 'SVT',
        'name': 'Supraventricular Tachycardia',
        'category': 'Supraventricular Arrhythmia',
        'description': 'Paroxysmal tachycardia originating above the ventricles, characterized by regular narrow QRS complexes.'
    },
    {
        'code': 'PVC',
        'name': 'Premature Ventricular Contraction',
        'category': 'Ventricular Ectopy',
        'description': 'Premature ectopic ventricular depolarization exhibiting wide, bizarre QRS morphology and compensatory pause.'
    },
    {
        'code': 'PAC',
        'name': 'Premature Atrial Contraction',
        'category': 'Atrial Ectopy',
        'description': 'Premature supraventricular depolarization featuring an early, morphologically distinct ectopic P-wave.'
    },
    {
        'code': 'BIGU',
        'name': 'Ventricular Bigeminy',
        'category': 'Ventricular Arrhythmia',
        'description': 'Coupled cardiac rhythm where every normal sinus beat is consistently paired with an ectopic premature ventricular beat.'
    },
    {
        'code': 'PACE',
        'name': 'Artificial Pacemaker Rhythm',
        'category': 'Paced Rhythm',
        'description': 'Rhythm driven by an artificial electronic cardiac pacemaker, evidenced by sharp pacing spikes before P or QRS complexes.'
    }
]

CLASS_CODES = [c['code'] for c in RHYTHM_CLASSES]
CLASS_NAMES = [c['name'] for c in RHYTHM_CLASSES]
NUM_CLASSES = len(RHYTHM_CLASSES)

# Deep Learning Hyperparameters (1D CNN + BiLSTM)
MODEL_CONFIG = {
    'input_shape': (SIGNAL_LENGTH, NUM_LEADS),  # (1000, 12)
    'num_classes': NUM_CLASSES,
    'cnn_filter_1': 64,
    'cnn_kernel_1': 7,
    'cnn_filter_2': 128,
    'cnn_kernel_2': 5,
    'bilstm_units': 64,
    'dense_units': 64,
    'dropout_rate': 0.35,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 15,
    'early_stopping_patience': 4,
    'loss': 'binary_crossentropy',
    'output_activation': 'sigmoid'
}

# Model Checkpoint Paths
MODEL_KERAS_PATH = os.path.join(MODELS_DIR, 'cardioai_cnn_bilstm.keras')
MODEL_METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')
PREPROCESSOR_CONFIG_PATH = os.path.join(MODELS_DIR, 'preprocessor_config.json')

