import argparse, os
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ordinea claselor așa cum ai antrenat
INSTRUMENTS = ['DRUM', 'GUITAR', 'PIANO', 'VIOLIN']

# parametrii tăi
SR = 22050
DURATION = 7
TARGET_LEN = SR * DURATION
MODEL_PATH = r"D:\Proiect_Licenta_2025\MODEL\instrument_classifier_optimized.keras"

# Maximul global pe care l-ai calculat la pasul anterior:
GLOBAL_MAX = 3.8146973e-06  # <— înlocuiește cu valoarea ta exactă

def preprocess(path):
    y, sr = librosa.load(path, sr=SR)
    # pad / truncate
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), 'constant')
    else:
        y = y[:TARGET_LEN]
    # mel-spectrogramă + dB
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    db = librosa.power_to_db(m, ref=np.max)
    # **aceeași** normalizare ca la antrenare
    db = db / GLOBAL_MAX
    return db

def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', help='cale către unul sau mai multe WAV-uri')
    p.add_argument('--model', default=MODEL_PATH, help='cale model .keras')
    args = p.parse_args()

    model = load_model(args.model)
    print(f"\nModel încărcat: {args.model}\n")

    for wav in args.files:
        if not os.path.isfile(wav):
            print(f"Fișier inexistent: {wav}\n")
            continue

        spec = preprocess(wav)
        inp = spec[np.newaxis, ..., np.newaxis]  # (1,128, time,1)

        probs = model.predict(inp)[0]
        idx = int(np.argmax(probs))
        label = INSTRUMENTS[idx]

        print(f"{os.path.basename(wav)} → Predicție: {label}")
        print("  Vector probabilități:")
        for i, inst in enumerate(INSTRUMENTS):
            print(f"    {inst:6s}: {probs[i]:.4f}")
        print()

        # afișăm și spectrograma
        plt.figure(figsize=(8,3))
        librosa.display.specshow(spec * GLOBAL_MAX, sr=SR, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{os.path.basename(wav)} → {label}")
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    main()
