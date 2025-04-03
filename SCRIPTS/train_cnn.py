import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# Încarcă datele preprocesate clar
X_train = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\X_train.npy")
y_train = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\y_train.npy")
X_validation = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\X_validation.npy")
y_validation = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\y_validation.npy")

# Normalizează datele (valorile vor fi între 0 și 1)
X_train = X_train / np.max(X_train)
X_validation = X_validation / np.max(X_validation)

# Adaugă un canal suplimentar pentru CNN (deoarece imaginile sunt grayscale)
X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]

# Transformă etichetele în one-hot encoding (pentru 4 clase: de ex. Tobă, Chitară, Pian, Vioară)
y_train = to_categorical(y_train, num_classes=4)
y_validation = to_categorical(y_validation, num_classes=4)

# Construiește modelul CNN optimizat
model = Sequential([
    # Primul bloc de convoluție
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, X_train.shape[2], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Al doilea bloc de convoluție
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Al treilea bloc de convoluție
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Dropout pentru a reduce overfitting-ul
    Dropout(0.3),

    # În loc de Flatten, folosim GlobalAveragePooling pentru a reduce numărul de parametri
    GlobalAveragePooling2D(),

    # Straturi Dense suplimentare pentru a învăța caracteristici abstracte
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    # Stratul final cu softmax pentru clasificare în 4 clase
    Dense(4, activation='softmax')
])

# Folosește optimizatorul Adam cu un learning rate mic
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Afișează sumarul modelului pentru a verifica arhitectura
model.summary()

# Antrenează modelul - poți ajusta epocile, batch size-ul și adăuga callback-uri pentru early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_validation, y_validation),
    callbacks=[early_stop]
)

# Salvează modelul optimizat
model.save('MODEL/instrument_classifier_optimized.keras')