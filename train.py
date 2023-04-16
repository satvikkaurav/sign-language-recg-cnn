from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

from tensorflow import keras
from tensorflow.keras import layers

# Importing the Keras libraries and packages


import os
sz = 310
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Step 1 - Building the CNN
# Initializing the CNN
classifier = keras.Sequential()

# First convolution layer and pooling
classifier.add(layers.Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation="relu"))
classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(layers.Convolution2D(32, (3, 3), activation="relu"))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(layers.Flatten())

# Adding a fully connected layer
classifier.add(layers.Dense(units=128, activation="relu"))
classifier.add(layers.Dropout(0.40))
classifier.add(layers.Dense(units=96, activation="relu"))
classifier.add(layers.Dropout(0.40))
classifier.add(layers.Dense(units=64, activation="relu"))
classifier.add(layers.Dense(units=26, activation="softmax"))  # softmax for more than 2


# Compiling the CNN
classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)  # categorical_crossentropy for more than 2

# Step 2 - Preparing the train/test data and training the model
classifier.summary()
# Code copied from - https://keras.io/preprocessing/image/
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# training_set = train_datagen.flow_from_directory(
#     "data2/train",
#     target_size=(sz, sz),
#     batch_size=10,
#     color_mode="grayscale",
#     class_mode="categorical",
# )

# test_set = test_datagen.flow_from_directory(
#     "data2/test",
#     target_size=(sz, sz),
#     batch_size=10,
#     color_mode="grayscale",
#     class_mode="categorical",
# )
# classifier.fit(
#     training_set,
#     steps_per_epoch=12841,  # No of images in training set
#     epochs=3,
#     validation_data=test_set,
#     validation_steps=4268,
# )  # No of images in test set

#
from tensorflow.keras.preprocessing import image_dataset_from_directory
train_ds = image_dataset_from_directory(
    directory='data2/train/',
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(310, 310))
validation_ds = image_dataset_from_directory(
    directory='data2/test/',
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(310, 310))

classifier.fit(train_ds,epochs=15,validation_data=validation_ds)

# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print("Model Saved")
classifier.save_weights("model-bw.h5")
print("Weights saved")