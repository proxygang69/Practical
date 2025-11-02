# Assignment 4: ECG Anomaly Detection using Autoencoders

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =====================
# 1. Load ECG Dataset
# =====================
url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
data = pd.read_csv(url, header=None)

# Last column = labels (0 = anomaly, 1 = normal)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# =====================
# 2. Preprocessing
# =====================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train only on "normal" samples (label = 1)
X_normal = X_scaled[y == 1]

# Split into train-test sets
x_train, x_test = train_test_split(X_normal, test_size=0.2, random_state=42)

# Also keep anomalies for evaluation
x_test_all = X_scaled
y_test_all = y

# =====================
# 3. Build Autoencoder
# =====================
input_dim = X.shape[1]  # number of features

input_layer = Input(shape=(input_dim,))
# Encoder
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="relu")(encoded)
# Decoder
decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# =====================
# 4. Train Autoencoder
# =====================
history = autoencoder.fit(x_train, x_train,
                          epochs=20,
                          batch_size=512,
                          validation_data=(x_test, x_test),
                          verbose=1)

# =====================
# 5. Plot Training Loss
# =====================
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Autoencoder Training Loss")
plt.show()

# =====================
# 6. Reconstruction Error
# =====================
reconstructions = autoencoder.predict(x_test_all)
mse = np.mean(np.power(x_test_all - reconstructions, 2), axis=1)

# Set threshold = mean + std of training reconstruction error
threshold = np.mean(mse) + np.std(mse)
print("Reconstruction error threshold:", threshold)

# =====================
# 7. Anomaly Detection
# =====================
y_pred = [0 if e > threshold else 1 for e in mse]

from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_all, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_all, y_pred))