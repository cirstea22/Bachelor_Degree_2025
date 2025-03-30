import librosa
import librosa.display
import matplotlib.pyplot as plt

# Încarcă un fișier audio pentru verificare (de exemplu, unul din TRAIN_DATA)
file_path = r"D:\Proiect_Licenta_2025\DATA\TRAIN_DATA\DRUM\685466__digitalunderglow__drums_115_breaky_full.wav" 
audio, sr = librosa.load(file_path, sr=22050)

# Extragere MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Afișare grafică
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC (13 Coeficienți)')
plt.tight_layout()
plt.show()
