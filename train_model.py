import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from models.trainer import train_multilabel_model, check_model_status

print("=== Starting Real SE-ResNet-1D Multi-Label Training ===")
metrics, history = train_multilabel_model(epochs=12, batch_size=32, lr=1e-3)
print("\n=== Training Completed ===")
print("Final Metrics:", metrics)
status = check_model_status()
print("Model Status:", status['status'])

