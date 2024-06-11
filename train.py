import numpy as np
from data.data_loader import get_data
from models.sparse_autoencoder import build_model
from utils.visualize import visualize_reconstruction
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
images = get_data()

if len(images) == 0:
    raise ValueError("No images found in the specified directory. Please check the directory path and image files.")

# Ensure images are reshaped correctly
images = images.reshape(-1, 32, 32, 1)  # Reshape images to 4D tensor
input_shape = images.shape[1:]

# Build and train the model
optimizer_choice = 'adamw'  # Change this to 'adamw', 'rmsprop', or 'nadam'
model = build_model(input_shape, optimizer=optimizer_choice)

# Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(images, images, epochs=300, batch_size=128, shuffle=True, validation_split=0.1, callbacks=[early_stopping, reduce_lr])

# Get the reconstructed images
reconstructed = model.predict(images)

# Visualize the results
visualize_reconstruction(images, reconstructed)