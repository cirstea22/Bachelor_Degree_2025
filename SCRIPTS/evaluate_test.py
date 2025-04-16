import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model

# === Configurări de bază ===
# Calea absolută către sample-ul audio de test (modifică după necesitate)
sample_path = r"D:\Proiect_Licenta_2025\piano.wav"  # actualizează calea

# Rata de eșantionare și durata target (7 secunde)
sr_target = 22050
target_duration = 7  # secunde
target_length = target_duration * sr_target  # numărul de eșantioane pentru 7 secunde

# Calea către modelul salvat (actualizează calea dacă e necesar)
model_path = r"D:\Proiect_Licenta_2025\MODEL\instrument_classifier_optimized_V3.keras"

# Maparea indicilor de clasă la numele instrumentelor (asigură-te că ordinea corespunde exact cu cea folosită la antrenare)
instrumente = {0: "DRUM", 1: "GUITAR", 2: "PIANO", 3: "VIOLIN"}

# === Funcția de preprocesare audio ===
def preprocess_audio(path, sr_target, target_length):
    # Încarcă fișierul audio la rata dorită
    audio, sr = librosa.load(path, sr=sr_target)
    print(f"Lungimea audio inițială: {len(audio)} eșantioane")
    
    # Completează (zero-pad) dacă sample-ul este prea scurt sau taie dacă este prea lung
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    else:
        audio = audio[:target_length]
    print(f"Lungimea audio după preprocesare: {len(audio)} eșantioane")
    
    # Generează spectrograma Mel (folosind 128 de mel-bands)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr_target, n_mels=128)
    # Convertește spectrograma la decibeli
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalizare locală pentru a avea valori între 0 și 1 (a se face astfel dacă la antrenare s-a făcut similar)
    spectrogram_db = spectrogram_db / np.max(spectrogram_db)
    
    return spectrogram_db

# === Încarcă modelul (o singură dată) ===
model = load_model(model_path)
print("Modelul a fost încărcat!")

# === Funcția de testare a unui singur sample ===
def test_single_sample(path):
    # Preprocesează sample-ul audio
    spectrogram_db = preprocess_audio(path, sr_target, target_length)
    print("Forma spectrogramei generate:", spectrogram_db.shape)  # Ex: (128, N)
    
    # Pregătește datele pentru model: adaugă dimensiunea batch și cea de canal
    input_data = spectrogram_db[np.newaxis, ..., np.newaxis]
    print("Forma input_data pentru model:", input_data.shape)  # Trebuie să fie (1, 128, N, 1)
    
    # Efectuează predicția
    predictions = model.predict(input_data)
    print("Vectorul de probabilități:", predictions)
    
    # Se ia indexul celui mai mare element
    pred_class = np.argmax(predictions, axis=1)[0]
    predicted_instrument = instrumente[pred_class]
    return spectrogram_db, predicted_instrument, predictions

# === Execută testul pentru sample-ul specificat ===
if os.path.exists(sample_path):
    spect_db, predicted_instrument, pred_vector = test_single_sample(sample_path)
    print("Instrumentul prezis:", predicted_instrument)
    
    # Afișează spectrograma pentru sample-ul testat
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect_db, sr=sr_target, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogramă Mel (Predicție: {predicted_instrument})")
    plt.tight_layout()
    plt.show()
else:
    print(f"Fișierul nu a fost găsit: {sample_path}")
