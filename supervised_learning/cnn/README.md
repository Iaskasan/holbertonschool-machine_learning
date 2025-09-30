# 🧠 Convolutional Neural Networks (CNNs)

This document explains the core building blocks of CNNs, how forward and backward propagation work for convolutional and pooling layers, and how to implement CNNs using TensorFlow and Keras.

---

## 📌 What is a Convolutional Layer?

A **convolutional layer** is the heart of a CNN.  
It applies small filters (kernels) that slide across the input image (or feature maps) to detect local patterns like edges, textures, or shapes.

- **Input:** a tensor `(batch_size, height, width, channels)`
- **Parameters:** filters `W` of shape `(filter_height, filter_width, input_channels, output_channels)` + biases
- **Output:** a new tensor `(batch_size, new_height, new_width, output_channels)`

👉 Each filter learns to detect one kind of feature.

---

## 📌 What is a Pooling Layer?

A **pooling layer** reduces the spatial size of feature maps, keeping the most important information while lowering computation.

- **Max pooling:** takes the maximum value in each window → keeps strongest signal.
- **Average pooling:** takes the average → smooths the signal.

Example with a `2x2` max pooling kernel:

```
1 3 → 3
2 4 → 4
```

---


## 📌 Forward Propagation

### 🔹 Convolutional Layer (forward pass)

For each sliding window `(h, w)` and each filter `k`:

\[
Z_{(h, w, k)} = \sum (A_{\text{slice}} \cdot W_k) + b_k
\]

- `A_slice` = local patch of the input  
- `W_k` = filter weights  
- `b_k` = bias term  

Then apply an activation (e.g. ReLU, sigmoid).

---

### 🔹 Pooling Layer (forward pass)

For each window `(h, w)` in the input:

- **Max pooling:**
  \[
  A_{(h, w, c)} = \max(A_{\text{slice}})
  \]

- **Average pooling:**
  \[
  A_{(h, w, c)} = \text{mean}(A_{\text{slice}})
  \]

---

## 📌 Back Propagation

### 🔹 Convolutional Layer (backward pass)

Given `dZ` (gradient wrt pre-activation output):

- Gradient wrt input:
  \[
  dA_{\text{prev}} += W \cdot dZ
  \]

- Gradient wrt filters:
  \[
  dW += A_{\text{slice}} \cdot dZ
  \]

- Gradient wrt bias:
  \[
  db = \sum dZ
  \]

---

### 🔹 Pooling Layer (backward pass)

- **Max pooling:** pass the gradient back only to the max element in each window.  
- **Average pooling:** distribute the gradient evenly across all elements in the window.

---

## 📌 Building a CNN with TensorFlow & Keras

Here’s a minimal CNN for image classification (e.g. MNIST digits):

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train on data (X_train, y_train)
# model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
```
---
### 📝 Explanation

- **Conv2D layers**: Apply convolution filters that automatically learn useful features such as edges, textures, or shapes.  
- **MaxPooling2D layers**: Reduce the spatial dimensions (height and width), keeping only the most important information while lowering computation.  
- **Flatten layer**: Converts the 2D feature maps into a 1D vector so it can be fed into fully connected (Dense) layers.  
- **Dense layers**: Standard neural network layers that combine features and perform classification.  
- **Softmax layer**: Produces a probability distribution across the target classes.

---

## ✅ Summary

- **Convolutional layer**: extracts local features from the input (edges, shapes, patterns).  
- **Pooling layer**: reduces the feature map size and preserves important information.  
- **Forward propagation**: computes the output of convolution and pooling operations given the input.  
- **Backward propagation**: computes the gradients of loss with respect to filters, biases, and inputs, enabling learning through gradient descent.  
- **Building CNNs in TensorFlow/Keras**: straightforward using layers like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`.  

👉 CNNs are the backbone of modern computer vision tasks such as **image classification, object detection, and segmentation** 🚀.
 