import numpy as np
from data.data_loader import get_data
from models.sparse_autoencoder import build_model
from utils.visualize import visualize_reconstruction
images=get_data()
input_dim=images.shape[1]
hidden_dim=64

model=build_model(input_dim,hidden_dim)
model.fit(images,images, epochs=500, batch_size=256,shuffle=True, validation_split=0.2)
model.save('models/sparse_autoencoder.h5')

reconstructed_images=model.predict(images)
visualize_reconstruction(images,reconstructed_images)

