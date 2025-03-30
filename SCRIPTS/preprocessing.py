import os
import librosa
import numpy as np

# Setează clar calea către datele tale
data_path = r"D:\Proiect_Licenta_2025\DATA\TEST_DATA"  # modifică ulterior pentru VALIDATION_DATA și TEST_DATA

# Numele instrumentelor
instruments = ['DRUM', 'GUITAR', 'PIANO', 'VIOLIN']

X, y = [], []

# Parcurge fiecare instrument și fișier audio
for idx, instrument in enumerate(instruments):
    print(f"Procesez instrumentul: {instrument}")
    instrument_path = os.path.join(data_path, instrument)

    for file in os.listdir(instrument_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(instrument_path, file)

            # Încarcă fișierul audio la 22050 Hz
            audio, sr = librosa.load(file_path, sr=22050)

            # Setează lungimea dorită (exact 7 secunde)
            target_length = 7 * sr  # 7 secunde

            # Padding (pentru audio scurt) sau decupare (pentru audio lung)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            # Generare spectrogramă Mel și conversie la dB
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Adaugă în listele finale
            X.append(spectrogram_db)
            y.append(idx)

# Conversia listelor în numpy arrays
X = np.array(X)
y = np.array(y)

# Alegere automată a numelui fișierului în funcție de setul procesat
if "TRAIN" in data_path:
    np.save('X_train.npy', X)
    np.save('y_train.npy', y)
elif "VALIDATION" in data_path:
    np.save('X_validation.npy', X)
    np.save('y_validation.npy', y)
elif "TEST" in data_path:
    np.save('X_test.npy', X)
    np.save('y_test.npy', y)

print(f"Finalizat! Salvate {len(X)} sample-uri din {data_path}.")
