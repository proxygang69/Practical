import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=0.2,     
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    "Images/",              
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val = datagen.flow_from_directory(
    "Images/",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train.num_classes
print("Number of classes:", num_classes)

base_model = VGG16(include_top=False,weights=None,input_shape=(224,224,3))
base_model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False
    
    # Add custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------------------------
# d. Train Classifier Layers First
# -----------------------------------------------
model.compile(
    optimizer=Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    train,
    validation_data=val,
    epochs=10,
    verbose=1
)
# -----------------------------------------------
# e. Fine-Tune Upper Layers
# -----------------------------------------------
# Unfreeze last few convolutional layers
for layer in base_model.layers[-8:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),   # smaller LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history_fine = model.fit(
    train,
    validation_data=val,
    epochs=10,
    verbose=1
)
model.save("final_transfer_learning_model.h5")
# -----------------------------------------------
# Evaluate Final Model
# -----------------------------------------------
loss, acc = model.evaluate(val)
print(f"\n Final Validation Accuracy: {acc*100:.2f}%")
