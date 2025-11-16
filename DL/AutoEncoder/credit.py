import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]
X_normal = X[y == 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normal)
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]      # 30 features
encoding_dim = 14                 # latent dimension

input_layer = Input(shape=(input_dim,))
encoder = Dense(20, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder)

decoder = Dense(20, activation='relu')(encoder)
decoder = Dense(input_dim, activation='linear')(decoder)

# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mse']
)
history = autoencoder.fit(
    X_train, X_train,
    epochs=25,
    batch_size=64,
    validation_data=(X_val, X_val),
    shuffle=True
)

plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Reconstruction Loss")
plt.legend()
plt.show()

X_all_scaled = scaler.transform(X)
X_pred = autoencoder.predict(X_all_scaled)

mse = np.mean(np.power(X_all_scaled - X_pred, 2), axis=1)
df["Reconstruction_error"] = mse

# threshold (you can adjust)
threshold = df[df["Class"] == 0]["Reconstruction_error"].quantile(0.95)

df["Predicted"] = (df["Reconstruction_error"] > threshold).astype(int)

accuracy = np.mean(df["Predicted"] == df["Class"])
print(f"✅ Overall Detection Accuracy: {accuracy*100:.2f}%")