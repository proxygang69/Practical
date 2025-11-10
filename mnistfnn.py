import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Input, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

test = pd.read_csv("mnist_test.csv")
train = pd.read_csv("mnist_train.csv")

x_test = test.iloc[:,1:].values
y_test = test.iloc[:,0].values

x_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values

x_test = x_test.astype("float32") / 225
x_train = x_train.astype("float32") / 225
x_train = x_train.reshape(-1,28,28)
x_test = x_test.reshape(-1,28,28)

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(128,activation="sigmoid"),
    Dense(64,activation="sigmoid"),
    Dense(10,activation="softmax")
])

model.compile(optimizer=SGD(learning_rate=0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(x_train,y_train,
                    validation_data=(x_test,y_test),
                    epochs = 11,
                    batch_size = 32)
loss, acc = model.evaluate(x_test,y_test,verbose=0)
print(f"Accuracy is {acc*100:.2f}%")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label = 'Value Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Value Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs validation Accuracy')

plt.show()