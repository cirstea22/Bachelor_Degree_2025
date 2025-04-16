import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model

# === Configurări de bază ===
# Actualizează calea către sample-ul audio de testare
sample_path = r"D:\Proiect_Licenta_2025\blabla.wav"

# Rata de eșantionare și durata target (7 secunde)
sr_target = 22050
target_duration = 7  # secunde
target_length = target_duration * sr_target  # numărul total de eșantioane pentru 7 secunde

# Calea către modelul salvat (actualizează calea dacă e necesar)
model_path = r"D:\Proiect_Licenta_2025\MODEL\instrument_classifier_optimized.keras"

# Maparea indicilor de clasă la numele instrumentelor (asigură-te că ordinea corespunde)
instrumente = {0: "DRUM", 1: "GUITAR", 2: "PIANO", 3: "VIOLIN"}

# === Funcția de preprocesare audio ===
def preprocess_audio(path, sr_target, target_length):
    # Încarcă audio la rata dorită
    audio, sr = librosa.load(path, sr=sr_target)
    
    # Debug: afișează lungimea audio-ului inițial
    print(f"Lungimea audio inițială: {len(audio)} eșantioane")
    
    # Dacă sample-ul este prea scurt, repetă conținutul pentru a-l umple la target_length.
    if len(audio) < target_length:
        reps = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, reps)[:target_length]
    else:
        # Dacă sample-ul este prea lung, taie-l la target_length.
        audio = audio[:target_length]
    
    # Verifică forma audio după preprocesare
    print(f"Lungimea audio după preprocesare: {len(audio)} eșantioane")
    
    # Generează spectrograma Mel și convertește-o la decibeli
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr_target, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Încarcă modelul o singură dată (în afara funcției)
model = load_model(model_path)
print("Modelul a fost încărcat!")

# === Funcția de testare și predicție pentru un singur sample ===
def test_single_sample(path):
    # Preprocesează sample-ul audio
    spectrogram_db = preprocess_audio(path, sr_target, target_length)
    
    # Verifică forma spectrogramei: ar trebui să fie (n_mels, N)
    print("Forma spectrogramei:", spectrogram_db.shape)
    
    # Pregătește inputul pentru model: adaugă dimensiunea batch și cea de canal (1)
    input_data = spectrogram_db[np.newaxis, ..., np.newaxis]
    print("Forma input_data pentru model:", input_data.shape)
    
    # Efectuează predicția
    predictions = model.predict(input_data)
    print("Vectorul de probabilități:", predictions)
    
    pred_class = np.argmax(predictions, axis=1)[0]
    predicted_instrument = instrumente[pred_class]
    
    return spectrogram_db, predicted_instrument

# === Executarea testului pentru un singur sample ===
if os.path.exists(sample_path):
    spect_db, predicted_instrument = test_single_sample(sample_path)
    print("Instrumentul prezis:", predicted_instrument)
    
    # Afișează spectrograma
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect_db, sr=sr_target, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogramă Mel (Predicție: {predicted_instrument})")
    plt.tight_layout()
    plt.show()
else:
    print(f"Fișierul nu a fost găsit: {sample_path}")
