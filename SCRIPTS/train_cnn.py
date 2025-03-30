import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Încarcă datele preprocesate clar
X_train = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\X_train.npy")
y_train = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\y_train.npy")
X_validation = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\X_validation.npy")
y_validation = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\y_validation.npy")

# Ajustare shape date pentru CNN
X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]

# One-hot encoding pentru etichete
y_train = to_categorical(y_train, num_classes=4)
y_validation = to_categorical(y_validation, num_classes=4)

# Model CNN simplu (3 nivele convoluționale clar definite)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, X_train.shape[2], 1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),

    Dense(4, activation='softmax') # 4 instrumente
])

# Compilează modelul clar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Afișează sumarul modelului clar și simplu
model.summary()

# Antrenarea propriu-zisă a modelului
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_validation, y_validation)
)

# Salvarea modelului final
model.save('MODEL/instrument_classifier.keras')
