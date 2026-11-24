"""
1D Grad-CAM (Gradient-Weighted Class Activation Mapping) for ECG Time-Series
Explains deep learning multi-label predictions by localizing the exact
temporal segments and wave complexes (P-QRS-T) that contributed to each specific arrhythmia class.
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, Exception):
    HAS_TORCH = False
    torch = None
    F = None


class GradCAM1D:
    """
    Computes 1D Gradient-weighted Class Activation Mapping for 1D CNN / ResNet.
    Allows clinical researchers to query explanations for ANY target class independently.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate(self, input_tensor, target_class_idx):
        """
        Generates 1D Grad-CAM heatmap for a given input tensor and class index.
        input_tensor: torch.Tensor of shape (1, 1, 2500)
        target_class_idx: int (0 to num_classes - 1)

        Returns:
            heatmap (np.ndarray): 1D array of length 2500, normalized to [0, 1]
            peak_regions (list): List of salient segment intervals [(start_idx, end_idx, score)]
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)

        target_score = logits[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros(input_tensor.shape[-1], dtype=np.float32), []

        # Gradients: (1, Channels, L')
        # Activations: (1, Channels, L')
        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Global average pooling of gradients along the temporal dimension L'
        # alpha_k = (1 / L') * sum_t (dScore / dAct_k,t)
        weights = torch.mean(grads, dim=2, keepdim=True)  # (1, Channels, 1)

        # Weighted linear combination of forward activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (1, 1, L')

        # Pass through ReLU (only features that have a positive influence on target score)
        cam = F.relu(cam)

        # Interpolate back to original ECG signal length (e.g. 2500 samples)
        orig_len = input_tensor.shape[-1]
        cam_upsampled = F.interpolate(cam, size=orig_len, mode='linear', align_corners=False)
        cam_array = cam_upsampled[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        max_v = np.max(cam_array)
        min_v = np.min(cam_array)
        if max_v - min_v > 1e-8:
            normalized_cam = (cam_array - min_v) / (max_v - min_v)
        else:
            normalized_cam = np.zeros_like(cam_array)

        # Identify top salient temporal regions (segments above 50th percentile of activation)
        peak_regions = self._extract_salient_regions(normalized_cam)

        return normalized_cam, peak_regions

    def _extract_salient_regions(self, cam_array, threshold=0.45, min_gap=50):
        """Extracts continuous temporal intervals with high attribution."""
        high_idx = np.where(cam_array >= threshold)[0]
        if len(high_idx) == 0:
            return []

        regions = []
        start = high_idx[0]
        prev = high_idx[0]

        for idx in high_idx[1:]:
            if idx - prev > min_gap:
                max_score = float(np.max(cam_array[start:prev + 1]))
                regions.append({
                    'start_sample': int(start),
                    'end_sample': int(prev),
                    'duration_ms': round(float((prev - start) * 4.0), 1),  # assuming 250 Hz (4ms/sample)
                    'max_attribution': round(float(max_score), 3)
                })
                start = idx
            prev = idx

        # Add last region
        max_score = float(np.max(cam_array[start:prev + 1]))
        regions.append({
            'start_sample': int(start),
            'end_sample': int(prev),
            'duration_ms': round(float((prev - start) * 4.0), 1),
            'max_attribution': round(float(max_score), 3)
        })

        # Sort by peak attribution descending
        regions.sort(key=lambda r: r['max_attribution'], reverse=True)
        return regions[:5]  # Top 5 most influential segments

    def cleanup(self):
        """Removes forward/backward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
