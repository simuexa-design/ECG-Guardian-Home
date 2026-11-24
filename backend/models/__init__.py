# Models package
from .ecg_model import CLASSES, CLASS_IDS, CLASS_NAMES, NUM_CLASSES

try:
    from .ecg_model import ECGResNet1D
    from .trainer import check_model_status, load_trained_model, train_multilabel_model
except (ImportError, Exception):
    ECGResNet1D = None
    check_model_status = None
    load_trained_model = None
    train_multilabel_model = None

