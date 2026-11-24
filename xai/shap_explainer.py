"""
CardioAI - SHAP & Gradient Attribution Engine for ECG Explanations
Quantifies multi-lead importance and positive/negative evidence for arrhythmia predictions.
"""

import numpy as np
import tensorflow as tf
from config import LEAD_NAMES, NUM_LEADS, SIGNAL_LENGTH


def compute_lead_attribution(model, input_ecg, class_idx):
    """
    Computes normalized gradient-based attribution across all 12 leads for target class.
    input_ecg: shape (1000, 12)
    """
    inp_tensor = tf.convert_to_tensor(np.expand_dims(input_ecg, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inp_tensor)
        preds = model(inp_tensor)
        target_score = preds[0, class_idx]

    grads = tape.gradient(target_score, inp_tensor)
    if grads is None:
        return {lead: 8.33 for lead in LEAD_NAMES}

    grad_arr = np.abs(grads[0].numpy())  # (1000, 12)
    lead_sums = np.sum(grad_arr, axis=0)  # (12,)
    total = np.sum(lead_sums)

    lead_dict = {}
    if total > 1e-8:
        for i, lead in enumerate(LEAD_NAMES):
            lead_dict[lead] = round(float((lead_sums[i] / total) * 100.0), 1)
    else:
        for lead in LEAD_NAMES:
            lead_dict[lead] = round(100.0 / NUM_LEADS, 1)

    return lead_dict


def compute_shap_temporal_attribution(model, input_ecg, class_idx, background_samples=None, n_samples=20):
    """
    Approximates temporal SHAP values using Path Integrated Gradients
    (Axiomatically sound, equivalent to Aumann-Shapley values for neural networks).
    input_ecg: (1000, 12)
    """
    inp = np.asarray(input_ecg, dtype=np.float32)
    baseline = np.zeros_like(inp) if background_samples is None else np.mean(background_samples, axis=0)

    alphas = np.linspace(0.0, 1.0, n_samples)
    diff = inp - baseline
    accum_grads = np.zeros_like(inp)

    for alpha in alphas:
        interpolated = baseline + alpha * diff
        inp_tensor = tf.convert_to_tensor(np.expand_dims(interpolated, axis=0), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inp_tensor)
            preds = model(inp_tensor)
            score = preds[0, class_idx]

        grad = tape.gradient(score, inp_tensor)
        if grad is not None:
            accum_grads += grad[0].numpy()

    # Integrated Gradients = (Input - Baseline) * Average Gradient
    ig = diff * (accum_grads / float(n_samples))  # (1000, 12)

    # Lead-averaged temporal attribution trace
    temporal_shap = np.mean(ig, axis=1)  # (1000,)

    # Positive (evidence in favor) vs Negative (evidence against)
    pos_evidence = np.maximum(temporal_shap, 0)
    neg_evidence = np.minimum(temporal_shap, 0)

    return {
        'temporal_shap': temporal_shap,
        'pos_evidence': pos_evidence,
        'neg_evidence': neg_evidence,
        'lead_integrated_grads': ig
    }

