import numpy as np
import sys
sys.path.append("/home/neel/Desktop/Models_swaroop/github/MesoNet/")


from classifiers import *
from pipeline import *
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def make_predicton(image_path):
    classifier = Meso4()
    
    # Most important to be changed according to the space we are working
    classifier.load("../github/MesoNet/weights/Meso4_DF.h5")

    # but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

    # 3 - Predict

    # Load the image
    image = tf.io.read_file(image_path)
    # Decode the image
    image = tf.image.decode_image(image, channels=3)
    # Resize the image to the desired dimensions
    image = tf.image.resize(image, (256, 256))
    # Normalize the pixel values to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Expand the dimensions to create a batch of size 1
    image = tf.expand_dims(image, axis=0).numpy()

    return  classifier.predict(image)


def list_images_in_directory(
    directory_path, image_extensions=[".jpg", ".jpeg", ".png", ".gif"]
):
    image_files = [
        f
        for f in os.listdir(directory_path)
        if f.lower().endswith(tuple(image_extensions))
    ]
    return image_files

def load_and_predict(directory_path, image_file):
    return make_predicton((directory_path + "/" + image_file))

if __name__ == "__main__":
    # Most important to be changed according to the space we are working
    directory_path = "test_images/df"

    image_files = list_images_in_directory(directory_path)

    if image_files:
        print("Image files in the directory:")
        for image_file in image_files:
            print("Predicted: ", make_predicton((directory_path + "/" + image_file)))
    else:
        print("No image files found in the directory.")