import numpy as np

# Încarcă spectrogramele de antrenare
X_train = np.load(r"D:\Proiect_Licenta_2025\DATA\PRE_PROCESSED_DATA\X_train.npy")
# Află maximul lor
global_max = np.max(X_train)
print(global_max)