import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Read training data
train_data = pd.read_csv("train_data.csv")
X_train = train_data.iloc[:, :-1].values.astype('float32')
y_train = train_data['label'].values

# Read test data
test_data = pd.read_csv("test_data.csv")
X_test = test_data.iloc[:, :-1].values.astype('float32')
y_test = test_data['label'].values

# Normalize pixel values to [0, 1]
X_train /= 255.0
X_test /= 255.0

# One-hot encode labels

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- Build the model ---
model = Sequential([
    Flatten(input_shape=(3072,)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Train the model ---
H = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=11,
    batch_size=128,
    verbose=1
)

# --- Evaluate the model ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- Plot training history ---
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, 11), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 11), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, 11), H.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, 11), H.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
