import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_images(image_dir, target_size=(32, 32)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = load_img(os.path.join(image_dir, filename), target_size=target_size, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

def get_data():
    image_path = 'data/sample images'
   
    images=load_images(image_path)
    return images