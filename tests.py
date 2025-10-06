import matplotlib.pyplot as plt
import numpy as np

images = np.random.rand(5, 28, 28)  # 5 grayscale images

# Example: put 5 images side by side into one big array
big_image = np.hstack(images)

plt.imshow(big_image, cmap="gray")
plt.axis("off")
plt.show()

