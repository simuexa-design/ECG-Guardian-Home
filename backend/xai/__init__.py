# XAI package
try:
    from .gradcam1d import GradCAM1D
    from .saliency import compute_saliency_map, compute_smoothgrad, compute_integrated_gradients
except (ImportError, Exception):
    GradCAM1D = None
    compute_saliency_map = None
    compute_smoothgrad = None
    compute_integrated_gradients = None

from .interpreter import generate_clinical_interpretation

