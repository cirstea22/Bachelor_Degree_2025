import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# === Paths ===
DATA_DIR  = r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA"
MODEL_DIR = r"D:\Proiect_Licenta_2025\MODEL"

# === Load preprocessed test data ===
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

# Normalize inputs
X_test = X_test / np.max(X_test)
# Add channel dimension
X_test = X_test[..., np.newaxis]
# One-hot encoding
y_test_cat = to_categorical(y_test, num_classes=4)

# === Load best model ===
best_model_path = os.path.join(MODEL_DIR, 'instrument_classifier_optimized.keras')
model = load_model(best_model_path)
print(f"Loaded model from: {best_model_path}")

# === Evaluate ===
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# === Predictions & Metrics ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test

# Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['DRUM','GUITAR','PIANO','VIOLIN']))

# === Plotting optional (requires matplotlib) ===
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    acc = accuracy
    # You can add more plots here if desired
    # e.g., ROC curves per class or a heatmap of the confusion matrix


# === Plot Confusion Matrix Heatmap ===
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['DRUM','GUITAR','PIANO','VIOLIN'],
            yticklabels=['DRUM','GUITAR','PIANO','VIOLIN'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()

# === Plot Classification Report as Heatmap ===
from sklearn.metrics import precision_recall_fscore_support

# Extract metrics
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
metrics = np.vstack([precision, recall, f1]).T

plt.figure(figsize=(6,4))
sns.heatmap(metrics, annot=True, fmt='.2f', cmap='YlGnBu',
            xticklabels=['Precision','Recall','F1-score'],
            yticklabels=['DRUM','GUITAR','PIANO','VIOLIN'])
plt.title('Classification Report')
plt.ylabel('Class')
plt.tight_layout()
plt.show()

# === Optional: ROC Curves per class ===
from sklearn.metrics import roc_curve, auc

y_test_bin = to_categorical(y_true, num_classes=4)
fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6,5))
for i, color in zip(range(4), ['blue','red','green','orange']):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f"{['DRUM','GUITAR','PIANO','VIOLIN'][i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Class')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# End of extended evaluate_model.py
