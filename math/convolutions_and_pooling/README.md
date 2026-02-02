# 📘 Convolutions & Pooling – Quick Guide

This document explains the fundamentals of **convolutional operations** in image processing and deep learning.

---

## 🔹 What is a convolution?
A **convolution** is an operation where a small grid of numbers (called a **kernel** or **filter**) slides over an image, performing element-wise multiplication and summation.  

- Input: image (grid of pixel values).  
- Kernel: small matrix (e.g. 3×3).  
- Output: **feature map** that highlights specific patterns (edges, textures, etc.).  

👉 Convolutions allow neural networks to detect **local features** in an image.

---

## 🔹 What is max pooling? average pooling?
**Pooling** reduces the size of feature maps while keeping important information.  

- **Max pooling**: takes the **largest value** in each window.  
  Example (2×2 window):
```python
[1, 3]
[2, 4] → max = 4
```

- **Average pooling**: takes the **average value** in each window.  
```python
[1, 3]
[2, 4] → avg = (1+2+3+4)/4 = 2.5
```


👉 Pooling helps reduce computation and makes the model more robust to small shifts in the image.

---

## 🔹 What is a kernel/filter?
A **kernel** (or filter) is a small matrix of weights used in convolution.  
- Size: typically 3×3, 5×5, etc.  
- Function: detects features such as edges, corners, or textures.  
- In CNNs, kernels are **learned** automatically during training.

---

## 🔹 What is padding?
**Padding** means adding extra pixels (usually zeros) around the image borders before applying convolution.  

Purpose:
- Ensures kernels can cover edge pixels.  
- Controls the output size.

---

## 🔹 What is “same” padding? “valid” padding?
- **“Same” padding**: add just enough padding so the output has the **same spatial dimensions** (height, width) as the input.  
- **“Valid” padding**: no padding; kernel only slides where it fully fits inside the image → output is **smaller**.

---

## 🔹 What is a stride?
The **stride** is how far the kernel moves at each step.  
- Stride = 1 → move kernel 1 pixel at a time (maximum overlap).  
- Stride = 2 → move kernel 2 pixels at a time (downsamples the output).  

Bigger stride → smaller output.

---

## 🔹 What are channels?
**Channels** represent different components of the image:  
- Grayscale image → 1 channel.  
- RGB image → 3 channels (Red, Green, Blue).  
- CNNs can learn multiple **feature channels** (e.g. 32, 64, 128 filters).

---

## 🔹 How to perform a convolution over an image
1. Place the kernel at the top-left corner of the image.  
2. Multiply the kernel values by the overlapping image pixels.  
3. Sum the results → this is **one pixel** in the output feature map.  
4. Slide the kernel across the image (according to stride).  
5. Repeat until the whole image is processed.

👉 Formula for output size with padding `(ph, pw)` and stride `(sh, sw)`:
```python
output_height = (h + 2ph - kh) // sh + 1
output_width = (w + 2pw - kw) // sw + 1
```


---

## 🔹 How to perform max/average pooling over an image
1. Choose a **pooling window** (e.g. 2×2) and stride.  
2. Slide the window over the feature map.  
3. For each region:
   - **Max pooling**: take the largest value.  
   - **Average pooling**: take the mean value.  
4. Store the result in the output map.  

👉 Example (2×2 max pooling on a 4×4 feature map):
```python
[1, 3, 2, 0] [6, 5]
[4, 6, 5, 1] → [8, 9]
[7, 8, 9, 2]
[0, 1, 3, 4]
```


---

## ✅ Summary
- **Convolution**: extracts features using kernels.  
- **Pooling**: reduces feature map size (max = strongest, average = smoothing).  
- **Padding**: controls edges and output size.  
- **Stride**: controls how much you move the kernel.  
- **Channels**: color or learned feature depth.  

Together, these are the building blocks of **Convolutional Neural Networks (CNNs)**.  

---
