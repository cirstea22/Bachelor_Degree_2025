import os
import librosa
import numpy as np

# Setează calea către date (schimbă între TRAIN_DATA, VALIDATION_DATA și TEST_DATA după caz)
data_path = r"D:\Proiect_Licenta_2025\DATA\TEST_DATA"  # exemplu pentru datele de antrenare

# Lista de instrumente (asigură-te că folderele din DATA corespund acestor nume)
instruments = ['DRUM', 'GUITAR', 'PIANO', 'VIOLIN']

X, y = [], []
target_duration = 7          # durata dorită în secunde
sr_target = 22050            # rata de eșantionare
target_length = target_duration * sr_target  # numărul de eșantioane pentru 7 secunde

# Parcurgem fiecare instrument
for idx, instrument in enumerate(instruments):
    print(f"Procesez instrumentul: {instrument}")
    instrument_path = os.path.join(data_path, instrument)
    
    for file in os.listdir(instrument_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(instrument_path, file)
            # Încarcă fișierul audio la rata dorită
            audio, sr = librosa.load(file_path, sr=sr_target)
            
            processed_samples = []
            if "TRAIN" in data_path.upper():
                # Pentru datele de antrenare: NU irosim nicio informație
                if len(audio) < target_length:
                    # Dacă audio-ul este prea scurt: repetă conținutul pentru a obține 7 secunde
                    reps = int(np.ceil(target_length / len(audio)))
                    audio_extended = np.tile(audio, reps)
                    processed_samples.append(audio_extended[:target_length])
                else:
                    # Dacă audio-ul este mai lung, îl împărțim în segmente de 7 secunde
                    num_full_chunks = len(audio) // target_length
                    for i in range(num_full_chunks):
                        chunk = audio[i * target_length: (i + 1) * target_length]
                        processed_samples.append(chunk)
                    # Pentru porțiunea rămasă, repetăm conținutul pentru a umple complet cele 7 secunde
                    remainder = len(audio) % target_length
                    if remainder > 0:
                        remainder_audio = audio[-remainder:]
                        reps = int(np.ceil(target_length / len(remainder_audio)))
                        padded = np.tile(remainder_audio, reps)[:target_length]
                        processed_samples.append(padded)
            else:
                # Pentru datele de validare și test: preprocesăm "normal"
                if len(audio) < target_length:
                    # Dacă este prea scurt, zero-pad-uiește până la target_length
                    pad_width = target_length - len(audio)
                    processed_samples.append(np.pad(audio, (0, pad_width), mode='constant'))
                else:
                    # Dacă e prea lung, taie la target_length
                    processed_samples.append(audio[:target_length])
            
            # Pentru fiecare segment obținut
            for sample in processed_samples:
                # Generare spectrogramă Mel și conversie la dB
                spectrogram = librosa.feature.melspectrogram(y=sample, sr=sr_target, n_mels=128)
                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                X.append(spectrogram_db)
                y.append(idx)

# Convertim listele în numpy arrays
X = np.array(X)
y = np.array(y)

# Salvare automată în funcție de tipul setului de date
if "TRAIN" in data_path.upper():
    np.save('X_train.npy', X)
    np.save('y_train.npy', y)
elif "VALIDATION" in data_path.upper():
    np.save('X_validation.npy', X)
    np.save('y_validation.npy', y)
elif "TEST" in data_path.upper():
    np.save('X_test.npy', X)
    np.save('y_test.npy', y)

print(f"Finalizat! Salvate {len(X)} sample-uri din {data_path}.")
