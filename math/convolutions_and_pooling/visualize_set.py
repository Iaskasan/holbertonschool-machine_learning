import numpy as np
from matplotlib import pyplot as plt

DATASET_PATH = 'animals_1.npz'
SAMPLE_SIZE = 10

def visualize_set(verbose='True'):
    """Visualizes a sample of images with their labels."""
    data = np.load(DATASET_PATH)
    images = data['data']
    labels = data['labels']
    sample_size = SAMPLE_SIZE
    if verbose:
        print(f'Total images: {images.shape[0]}')
        print(f'Image dimensions: {images.shape[1:]}')
        print(f"Images shown: {sample_size}")
        print(f"DATASET used: {DATASET_PATH}")
    plt.figure(figsize=(15, 5))
    for i in range(sample_size):
        plt.subplot(2, sample_size // 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_set()