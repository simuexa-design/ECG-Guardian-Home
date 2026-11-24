"""
Point-by-Point Gradient Saliency and Integrated Gradients for ECG Attribution
Calculates sample-level input sensitivity to understand fine-grained morphology
such as ST elevation/depression, QRS widening, or ectopic beats.
"""

import numpy as np
import torch


def compute_saliency_map(model, input_tensor, target_class_idx):
    """
    Computes standard Gradient Saliency (first-order derivative dScore / dInput).
    Shows how infinitesimal perturbations at each millivolt sample shift the diagnosis.
    """
    model.eval()
    inp = input_tensor.clone().detach().requires_grad_(True)
    logits = model(inp)
    score = logits[0, target_class_idx]
    score.backward()

    grad = inp.grad.detach()[0, 0].cpu().numpy()
    abs_grad = np.abs(grad)

    max_v = np.max(abs_grad)
    if max_v > 1e-8:
        saliency_norm = abs_grad / max_v
    else:
        saliency_norm = np.zeros_like(abs_grad)

    return saliency_norm


def compute_smoothgrad(model, input_tensor, target_class_idx, num_samples=20, noise_level=0.15):
    """
    SmoothGrad: Averages gradient saliency maps over noisy copies of the input signal
    to eliminate gradient artifacts and highlight robust morphological features.
    """
    model.eval()
    signal_std = float(torch.std(input_tensor).item())
    sigma = noise_level * max(signal_std, 1e-4)

    accumulated_grads = np.zeros(input_tensor.shape[-1], dtype=np.float64)

    for _ in range(num_samples):
        noise = torch.randn_like(input_tensor) * sigma
        noisy_inp = (input_tensor + noise).clone().detach().requires_grad_(True)
        logits = model(noisy_inp)
        score = logits[0, target_class_idx]
        score.backward()

        grad = noisy_inp.grad.detach()[0, 0].cpu().numpy()
        accumulated_grads += (grad ** 2)

    smooth_saliency = accumulated_grads / num_samples
    max_v = np.max(smooth_saliency)
    if max_v > 1e-8:
        smooth_saliency /= max_v

    return smooth_saliency


def compute_integrated_gradients(model, input_tensor, target_class_idx, steps=25):
    """
    Integrated Gradients (Sundararajan et al., 2017):
    Path integral of gradients along a straight line from a baseline (zero signal) to input:
    IG_i = (x_i - x_base_i) * integral_0^1 (dModel(x_base + alpha*(x - x_base)) / dx_i) dAlpha
    Satisfies Completeness and Implementation Invariance axioms.
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor)
    diff = input_tensor - baseline

    alphas = torch.linspace(0.0, 1.0, steps=steps)
    accum_grads = torch.zeros_like(input_tensor)

    for alpha in alphas:
        interpolated = (baseline + alpha * diff).clone().detach().requires_grad_(True)
        logits = model(interpolated)
        score = logits[0, target_class_idx]
        score.backward()
        accum_grads += interpolated.grad.detach()

    avg_grads = accum_grads / float(steps)
    ig = (diff * avg_grads)[0, 0].cpu().numpy()

    # Absolute attribution
    abs_ig = np.abs(ig)
    max_v = np.max(abs_ig)
    if max_v > 1e-8:
        ig_norm = abs_ig / max_v
    else:
        ig_norm = np.zeros_like(abs_ig)

    return ig_norm

