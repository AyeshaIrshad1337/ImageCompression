# SparseAutoencoderImageCompression

This project demonstrates image compression and reconstruction using a sparse autoencoder built with TensorFlow.

## File Structure

- `data/`: Contains data loading scripts and sample images.
- `models/`: Defines the sparse autoencoder model.
- `utils/`: Includes utility scripts for visualizing the results.
- `main.py`: Main script to load data, train the model, and visualize results.
- `requirements.txt`: Lists the required Python packages.

## How to Run

1. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

2. Place your sample images in the `data/sample_images/` directory.

3. Run the main script to train the model and visualize results:
    ```
    python train.py
    ```

## Description

The project trains a sparse autoencoder to compress and reconstruct images. The autoencoder consists of an encoder that compresses the image and a decoder that reconstructs it. The sparsity constraint encourages the model to use fewer neurons, effectively learning a more compact representation of the input data.
