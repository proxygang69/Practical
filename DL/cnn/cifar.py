import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt






# Load datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Separate features and labels
X_train = train_data.iloc[:, :-1].values.astype('float32')
y_train = train_data['label'].values

X_test = test_data.iloc[:, :-1].values.astype('float32')
y_test = test_data['label'].values

# Reshape to (32, 32, 3)
X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

# Normalize pixel values to [0, 1]
X_train /= 255.0
X_test /= 255.0

# One-hot encode labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# =====================
# 2. Define CNN Model
# =====================
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='Conv_1'),
    MaxPooling2D((2, 2), name='Pool_1'),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', name='Conv_2'),
    MaxPooling2D((2, 2), name='Pool_2'),

    # Classification Head
    Flatten(name='Flatten_Layer'),
    Dense(100, activation='relu', name='Dense_1'),
    Dense(10, activation='softmax', name='Output_Layer')
])

# =====================
# 3. Compile Model
# =====================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# 4. Train Model
# =====================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=128,
    verbose=1
)

# =====================
# 5. Evaluate Model
# =====================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# =====================
# 6. Plot Training History (ggplot style)
# =====================
plt.style.use("ggplot")
plt.figure(figsize=(10, 5))

plt.plot(np.arange(0, 10), history.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 10), history.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, 10), history.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, 10), history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
