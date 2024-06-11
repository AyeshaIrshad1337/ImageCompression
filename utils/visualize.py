import matplotlib.pyplot as plt
import numpy as np

def visualize_reconstruction(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(32, 32), cmap='gray')
        plt.title("Original")
        plt.axis("off")
        
        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(32, 32), cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()
