"""
CardioAI - 1D Grad-CAM (Gradient-Weighted Class Activation Mapping) for Keras
Computes temporal and lead-wise attribution for 12-lead ECG signals
to explain multi-label arrhythmia predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d

from config import LEAD_NAMES, SIGNAL_LENGTH, NUM_LEADS


class GradCAM1D_Keras:
    """
    1D Grad-CAM Explainer for Keras Hybrid 1D CNN + BiLSTM models.
    Supports multi-label target class selection and 12-lead spatial decomposition.
    """
    def __init__(self, model, target_conv_layer_name="conv1d_stage2_target"):
        self.model = model
        self.target_conv_layer_name = target_conv_layer_name
        self.grad_model = self._build_gradient_submodel()

    def _build_gradient_submodel(self):
        """Constructs an intermediate sub-model that returns conv activations and logits."""
        conv_layer = self.model.get_layer(self.target_conv_layer_name)
        return keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output]
        )

    def explain(self, input_ecg, class_idx):
        """
        Computes 1D Grad-CAM heatmap for a specific class index.
        input_ecg: shape (1, 1000, 12) or (1000, 12)
        class_idx: int (0 to NUM_CLASSES - 1)

        Returns:
            heatmap_1d: (1000,) normalized to [0, 1]
            lead_importance: dict mapping lead name to relative importance percentage
            salient_segments: list of dicts with {start_time, end_time, duration_ms, peak_score}
        """
        if input_ecg.ndim == 2:
            input_tensor = tf.convert_to_tensor(np.expand_dims(input_ecg, axis=0), dtype=tf.float32)
        else:
            input_tensor = tf.convert_to_tensor(input_ecg, dtype=tf.float32)

        # Gradient tape tracking
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = self.grad_model(input_tensor)
            # Target class logit / probability
            target_class_score = predictions[0, class_idx]

        # 1. Gradients with respect to convolutional feature maps
        conv_grads = tape.gradient(target_class_score, conv_outputs)  # shape (1, L_conv, filters)

        # 2. Input-level gradients for 12-lead attribution
        input_grads = tape.gradient(target_class_score, input_tensor)  # shape (1, 1000, 12)
        del tape

        if conv_grads is None:
            return np.zeros(SIGNAL_LENGTH), {lead: 8.33 for lead in LEAD_NAMES}, []

        # 3. Global Temporal Average Pooling of Gradients
        # alpha_k = (1 / L') * sum_t (dScore / dAct_k)
        weights = tf.reduce_mean(conv_grads, axis=1)  # shape (1, filters)

        # 4. Weighted Linear Combination of Forward Activations
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)  # shape (1, L_conv)

        # 5. Apply ReLU: only positive influences on this diagnosis
        cam = tf.maximum(cam, 0)
        cam_arr = cam[0].numpy()

        # 6. Upsample to original signal length (1000 samples)
        orig_len = SIGNAL_LENGTH
        conv_len = len(cam_arr)
        if conv_len != orig_len:
            x_old = np.linspace(0, 1, conv_len)
            x_new = np.linspace(0, 1, orig_len)
            f_interp = interp1d(x_old, cam_arr, kind='linear', fill_value="extrapolate")
            upsampled_cam = f_interp(x_new)
        else:
            upsampled_cam = cam_arr

        # 7. Normalize to [0.0, 1.0]
        max_val = np.max(upsampled_cam)
        min_val = np.min(upsampled_cam)
        if max_val - min_val > 1e-8:
            norm_cam = (upsampled_cam - min_val) / (max_val - min_val)
        else:
            norm_cam = np.zeros_like(upsampled_cam)

        # 8. Compute Lead Importance Scores
        lead_scores = {}
        if input_grads is not None:
            raw_lead_grads = np.sum(np.abs(input_grads[0].numpy()), axis=0)  # (12,)
            total_lead_grad = np.sum(raw_lead_grads)
            if total_lead_grad > 1e-8:
                pcts = (raw_lead_grads / total_lead_grad) * 100.0
            else:
                pcts = np.ones(NUM_LEADS) * (100.0 / NUM_LEADS)
            for i, lead in enumerate(LEAD_NAMES):
                lead_scores[lead] = round(float(pcts[i]), 1)
        else:
            for lead in LEAD_NAMES:
                lead_scores[lead] = round(100.0 / NUM_LEADS, 1)

        # 9. Extract Salient Time Intervals (regions exceeding 50% activation)
        salient_segments = self._extract_salient_intervals(norm_cam, fs=100.0)

        return norm_cam, lead_scores, salient_segments

    def _extract_salient_intervals(self, cam_array, threshold=0.45, min_gap=15, fs=100.0):
        """Identifies continuous temporal segments triggering high attribution."""
        high_idx = np.where(cam_array >= threshold)[0]
        if len(high_idx) == 0:
            return []

        intervals = []
        start = high_idx[0]
        prev = high_idx[0]

        for idx in high_idx[1:]:
            if idx - prev > min_gap:
                max_score = float(np.max(cam_array[start:prev + 1]))
                intervals.append({
                    'start_sample': int(start),
                    'end_sample': int(prev),
                    'start_time_sec': round(float(start / fs), 2),
                    'end_time_sec': round(float(prev / fs), 2),
                    'duration_ms': round(float((prev - start) * (1000.0 / fs)), 1),
                    'peak_attribution': round(float(max_score), 3)
                })
                start = idx
            prev = idx

        # Last interval
        max_score = float(np.max(cam_array[start:prev + 1]))
        intervals.append({
            'start_sample': int(start),
            'end_sample': int(prev),
            'start_time_sec': round(float(start / fs), 2),
            'end_time_sec': round(float(prev / fs), 2),
            'duration_ms': round(float((prev - start) * (1000.0 / fs)), 1),
            'peak_attribution': round(float(max_score), 3)
        })

        intervals.sort(key=lambda x: x['peak_attribution'], reverse=True)
        return intervals[:5]

