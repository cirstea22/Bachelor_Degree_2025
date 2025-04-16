import os
import librosa
import numpy as np

def preprocess_dataset(data_path, output_dir, mode):
    sr_target       = 22050
    target_duration = 7
    target_length   = sr_target * target_duration
    min_duration    = 3
    min_length      = sr_target * min_duration

    instruments = ['DRUM', 'GUITAR', 'PIANO', 'VIOLIN']
    X, y = [], []

    for idx, instr in enumerate(instruments):
        instr_dir = os.path.join(data_path, instr)
        if not os.path.isdir(instr_dir): continue
        print(f"\n>>> Procesăm clasa {instr} (index {idx})")

        for fname in os.listdir(instr_dir):
            if not fname.lower().endswith('.wav'): continue
            path = os.path.join(instr_dir, fname)
            audio, _ = librosa.load(path, sr=sr_target)
            length = len(audio)
            # Ignoră fișierele foarte scurte
            if length < min_length:
                continue

            segments = []
            if mode == 'train':
                # 3–7s → repetă până la 7s
                if length < target_length:
                    reps = int(np.ceil(target_length / length))
                    seg = np.tile(audio, reps)[:target_length]
                    segments.append(seg)
                else:
                    # segmente complete de 7s
                    n_full = length // target_length
                    for i in range(n_full):
                        segments.append(audio[i*target_length:(i+1)*target_length])
                    # restul
                    rem = length % target_length
                    if rem >= min_length:
                        tail = audio[-rem:]
                        reps = int(np.ceil(target_length / rem))
                        seg = np.tile(tail, reps)[:target_length]
                        segments.append(seg)
            else:
                # validation/test: pad sau slice
                if length < target_length:
                    seg = np.pad(audio, (0, target_length-length), mode='constant')
                else:
                    seg = audio[:target_length]
                segments.append(seg)

            # generează spectrograma Mel → dB și adaugă la X, y
            for seg in segments:
                melspec = librosa.feature.melspectrogram(
                    y=seg, sr=sr_target, n_mels=128
                )
                melspec_db = librosa.power_to_db(melspec, ref=np.max)
                X.append(melspec_db)
                y.append(idx)

    X = np.array(X)
    y = np.array(y)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'X_{mode}.npy'), X)
    np.save(os.path.join(output_dir, f'y_{mode}.npy'), y)
    print(f"\n→ {mode.upper()} gata: {X.shape[0]} sample‑uri salvate în {output_dir}")

if __name__ == "__main__":
    BASE       = r"D:\Proiect_Licenta_2025\DATA"
    OUT        = r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA"
    # Preprocesează TRAIN, VALIDATION și TEST
    preprocess_dataset(os.path.join(BASE, 'TRAIN_DATA'),      OUT, 'train')
    preprocess_dataset(os.path.join(BASE, 'VALIDATION_DATA'), OUT, 'validation')
    preprocess_dataset(os.path.join(BASE, 'TEST_DATA'),       OUT, 'test')
