"""
CardioAI - Deep Learning Model Architecture
Hybrid 1D CNN + Bidirectional LSTM for Multi-Label 12-Lead ECG Arrhythmia Classification.

Layer-by-Layer Rationale:
1. Input Layer (1000, 12):
   - Ingests 10-second standardized 12-lead ECG recordings at 100 Hz.
2. Conv1D Layer 1 (64 filters, kernel 7):
   - Extracts localized morphological features (QRS sharpness, P-wave deflections, T-wave repolarization).
3. Batch Normalization & ReLU:
   - Stabilizes gradient flow, accelerates convergence, prevents internal covariate shift.
4. MaxPooling1D (pool_size=2):
   - Downsamples temporal length by 50% while preserving translation invariance.
5. Conv1D Layer 2 (128 filters, kernel 5):
   - Captures higher-level morphological interactions across leads (e.g., ST-elevation patterns, notched R-peaks).
6. Batch Normalization & ReLU:
   - Further non-linear transformation and regularization.
7. MaxPooling1D (pool_size=2):
   - Further compresses temporal representation to allow temporal recurrence to focus on beat-to-beat cadence.
8. Bidirectional LSTM (64 units = 128 forward+backward):
   - Learns contextual rhythm dependencies and temporal transitions in both forward and backward time directions.
   - Crucial for detecting irregular beat sequences (AFib RR variability, premature beats with compensatory pauses).
9. Dropout (rate=0.35):
   - Prevents co-adaptation of neurons and mitigates overfitting.
10. Dense Layer (64 units, ReLU):
    - Synthesizes temporal sequence representations into dense feature embeddings.
11. Output Layer (NUM_CLASSES, Sigmoid):
    - Independent binary classification neurons. Sigmoid activation ensures multi-label support
      without mutual exclusivity (unlike Softmax, which is invalid for multi-label).
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import MODEL_CONFIG, NUM_CLASSES, SIGNAL_LENGTH, NUM_LEADS


def build_cnn_bilstm_model(input_shape=(SIGNAL_LENGTH, NUM_LEADS), num_classes=NUM_CLASSES):
    """
    Constructs the Keras Hybrid 1D CNN + BiLSTM model.
    Returns:
        compiled keras.Model instance
    """
    inputs = keras.Input(shape=input_shape, name="ecg_12lead_input")

    # --- CNN Block 1: Low-Level Morphology Extraction ---
    x = layers.Conv1D(
        filters=MODEL_CONFIG['cnn_filter_1'],
        kernel_size=MODEL_CONFIG['cnn_kernel_1'],
        padding="same",
        kernel_initializer="he_normal",
        name="conv1d_stage1"
    )(inputs)
    x = layers.BatchNormalization(name="bn_stage1")(x)
    x = layers.Activation("relu", name="relu_stage1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_stage1")(x)

    # --- CNN Block 2: Complex Pattern Extraction (Grad-CAM Target Layer) ---
    x = layers.Conv1D(
        filters=MODEL_CONFIG['cnn_filter_2'],
        kernel_size=MODEL_CONFIG['cnn_kernel_2'],
        padding="same",
        kernel_initializer="he_normal",
        name="conv1d_stage2_target"  # Final conv layer used for 1D Grad-CAM
    )(x)
    x = layers.BatchNormalization(name="bn_stage2")(x)
    x = layers.Activation("relu", name="relu_stage2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_stage2")(x)

    # --- Temporal Learning: Bidirectional LSTM ---
    x = layers.Bidirectional(
        layers.LSTM(
            units=MODEL_CONFIG['bilstm_units'],
            return_sequences=False,
            dropout=0.15,
            recurrent_dropout=0.1,
            name="lstm_core"
        ),
        name="bidirectional_lstm"
    )(x)

    # --- Classification Head ---
    x = layers.Dropout(MODEL_CONFIG['dropout_rate'], name="dropout_classifier")(x)
    x = layers.Dense(
        MODEL_CONFIG['dense_units'],
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_embedding"
    )(x)

    # Multi-Label Sigmoid Output Layer (NO SOFTMAX)
    outputs = layers.Dense(
        num_classes,
        activation=MODEL_CONFIG['output_activation'],
        name="multilabel_sigmoid_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CardioAI_CNN_BiLSTM")

    # Compile with Adam optimizer & Binary Crossentropy
    optimizer = keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=MODEL_CONFIG['loss'],
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(curve="ROC", multi_label=True, name="roc_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model


def get_model_summary_string(model):
    """Returns model architecture summary as formatted text."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

