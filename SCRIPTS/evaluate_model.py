import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# 1. Încarcă datele de validare
X_validation = np.load(r"D:\Proiect_Licenta_2025\DATA\OLD_PRE_PROCESSED_DATA\X_validation.npy")
y_val_raw = np.load(r"D:\Proiect_Licenta_2025\DATA\OLD_PRE_PROCESSED_DATA\y_validation.npy")

# 2. Normalizează și adaugă canalul suplimentar (exact ca la antrenare)
X_validation = X_validation / np.max(X_validation)
X_validation = X_validation[..., np.newaxis]

# 3. Transformă etichetele în one-hot
y_validation = to_categorical(y_val_raw, num_classes=4)

# 4. Încarcă modelul antrenat
model = load_model(r"D:\Proiect_Licenta_2025\MODEL\instrument_classifier_optimized.keras")

# 5. (Opțional) Verifică acuratețea direct cu model.evaluate
loss, acc = model.evaluate(X_validation, y_validation)
print(f"Acuratețea din evaluate: {acc:.2f}")

# 6. Obține predicțiile și generează matricea de confuzie
y_pred = np.argmax(model.predict(X_validation), axis=1)
y_true = np.argmax(y_validation, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Matricea de Confuzie:")
print(cm)
