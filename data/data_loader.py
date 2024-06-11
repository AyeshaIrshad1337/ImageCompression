import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
def load_images(path, target_size):
    images=[]
    ext=['jpg','jpeg','png']
    for filename in os.listdir(path):
        if filename.endwswith(tuple(ext)):
            img=load_img(os.path.join(path,filename),target_size=target_size,color='grayscale')
            img=img_to_array(img)/255.0
            images.append(img)
    return np.array(images)
def get_data():
    image_path = 'data/sample images'
    target_size = (28, 28)
    images=load_images(image_path,target_size)
    images=images.reshape((images.shape[0],-1))
    return images