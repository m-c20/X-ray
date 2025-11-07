
# at this point we load images from directories
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#paths

train_dir = "/home/mert/X-ray-ai/dataset/train"
val_dir = "/home/mert/X-ray-ai/dataset/val"

# image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# create data generators
train_datagen = ImageDataGenerator(
        rescale = 1./255,       # normalize pixel values
        rotation_range = 12,    # random rotations
        zoom_range= 0.1,        # random zoom
        horizontal_flip = True  # random flip        
)
# 
val_datagen = ImageDataGenerator(rescale=1./255) # validation : no augmentation

#create iterators
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (IMG_HEIGHT,IMG_WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary'
        )


val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary' 
        )


# show class indicies
print("Class mapping", train_generator.class_indices)

# preview first batch of images
images, labels = next(train_generator)
print("batch image shape",images.shape)
print("batch labels", labels)
