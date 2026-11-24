import os
import sys

# Add root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.trainer import train_cnn_bilstm_model, check_keras_model_status

print("=== CardioAI: Training Keras Hybrid 1D CNN + BiLSTM on PTB-XL Folds ===")
model, metrics = train_cnn_bilstm_model(epochs=10, batch_size=32)

print("\n=== Model Training Complete! ===")
print("Parameters:", metrics['total_parameters'])
print("Test Macro ROC-AUC:", metrics['test_metrics']['macro_roc_auc'])
print("Test Macro F1:", metrics['test_metrics']['macro_f1'])
print("Test Subset Accuracy:", metrics['test_metrics']['subset_accuracy'])
print("Hamming Loss:", metrics['test_metrics']['hamming_loss'])

status = check_keras_model_status()
print("\nModel Status:", status['status'])

