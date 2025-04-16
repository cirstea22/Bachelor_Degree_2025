import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Paths and Data Load
data_dir = r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA"
model_dir = r"D:\Proiect_Licenta_2025\MODEL"
os.makedirs(model_dir, exist_ok=True)

# Încarcă datele preprocesate
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_val   = np.load(os.path.join(data_dir, 'X_validation.npy'))
y_val   = np.load(os.path.join(data_dir, 'y_validation.npy'))

# Normalizează datele (0-1)
X_train = X_train / np.max(X_train)
X_val   = X_val   / np.max(X_val)

# Adaugă canalul grayscale
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]

# One-hot encoding pentru 4 clase
y_train = to_categorical(y_train, num_classes=4)
y_val   = to_categorical(y_val,   num_classes=4)

# Construiește modelul CNN
model = Sequential([
    # Bloc 1
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Bloc 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Bloc 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Regularizare
    Dropout(0.3),

    # Pooling global
    GlobalAveragePooling2D(),

    # Strat dens intermediar
    Dense(64, activation='relu'),
    Dropout(0.3),

    # Strat de ieșire
    Dense(4, activation='softmax')
])

# Compilează modelul
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Callback EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Antrenare
ehistory = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# Salvează modelul
model.save(os.path.join(model_dir, 'instrument_classifier_optimized.keras'))
print("Model salvat în:", os.path.join(model_dir, 'instrument_classifier_optimized.keras'))
