import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# =====================
# 1. Load Dataset (MNIST)
# =====================
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (0-1 scale)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =====================
# 2. Define Model Architecture
# =====================
model = Sequential([
    Input(shape=(28, 28)),          # Use Input layer to remove warnings
    Flatten(),                       # Flatten image
    Dense(128, activation='sigmoid'),# Hidden layer
    Dense(64, activation='sigmoid'), # Hidden layer
    Dense(10, activation='softmax')  # Output layer (10 classes)
])

# =====================
# 3. Compile Model (SGD optimizer)
# =====================
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =====================
# 4. Train Model
# =====================
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    batch_size=32)

# =====================
# 5. Evaluate Model
# =====================
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# =====================
# 6. Plot Loss & Accuracy
# =====================
plt.figure(figsize=(12,5))

# Training vs Validation Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")

# Training vs Validation Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")

plt.show()
