## Object Detection Concepts & YOLO Overview
### What is OpenCV and how do you use it?

OpenCV (Open Source Computer Vision Library) is a powerful open-source library for computer vision, image processing, and machine learning.
It provides tools to:

- Read, display, and manipulate images (cv2.imread, cv2.imshow, etc.)

- Detect and track objects (faces, features, motion)

- Apply transformations, filters, and edge detection

#### Example:
```python
import cv2
image = cv2.imread('dog.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
```
---
#### What is object detection?

Object detection is the process of identifying and locating objects within an image or video.
Unlike image classification, which assigns a single label to the entire image, object detection outputs bounding boxes that indicate the position of each detected object.

Common applications:

- Self-driving cars (detecting pedestrians, vehicles, traffic signs)

- Security systems (face or motion detection)

- Retail analytics (counting products or people)
---
### What is the Sliding Windows algorithm?
The Sliding Window algorithm is a traditional approach to object detection.
It involves moving a small rectangular window across the image at different scales and positions, classifying each region individually.

#### Steps:
1. Define a fixed-size window
2. Slide it across the image (left to right, top to bottom)
3. Extract features from each window
4. Classify whether it contains an object
---
### What is a single-shot detector?
A Single-Shot Detector (SSD) is a type of deep learning object detector that predicts bounding boxes and class labels in a single forward pass through the network.
#### Key ideas:
- No sliding windows or region proposals
- Fast inference, suitable for real-time applications
- Examples: YOLO, SSD, RetinaNet
---
### What is the YOLO algorithm?
YOLO (You Only Look Once) is a real-time object detection algorithm that formulates detection as a single regression problem.

It divides the input image into an S × S grid, and each grid cell predicts:
- Bounding boxes (x, y, w, h)
- Confidence score (objectness)
- Class probabilities

YOLO characteristics:

- Extremely fast (single pass through the network)
- Multi-scale detection for small, medium, and large objects
- Uses anchor boxes to predict multiple object shapes per grid cell
---
### What is IOU and how do you calculate it?
Intersection over Union (IOU) measures how much two bounding boxes overlap.
It is used to evaluate object detector accuracy and to remove duplicates during Non-Max Suppression.

Formula:
<p align="center">
  <br>
  <em>IOU = Area of Overlap / Area of Union</em>
</p>

Interpretation:
- ```IOU = 0``` → No overlap
- ```IOU = 1``` → Perfect overlap
- Typically, a detection is considered correct if ```IOU ≥ 0.5```
---
### What is non-max suppression (NMS)?
Non-Max Suppression (NMS) is used to remove redundant overlapping boxes that refer to the same object.

Steps:
1. Sort all predicted boxes by confidence score
2. Select the box with the highest score
3. Remove any boxes with IOU greater than a threshold (```nms_t```)
4. Repeat until no boxes remain

Purpose:
Keep only the most confident box per detected object.
---
### What are anchor boxes?
Anchor boxes are predefined bounding box templates that represent common object shapes (e.g., tall, wide, square).
Each grid cell in YOLO predicts offsets relative to these anchors.

Benefits:
- Allows detection of multiple objects per cell
- Handles varying aspect ratios efficiently

Example:
- One anchor box for people (tall and thin)
- Another for cars (wide and short)

---
### What is mAP and how do you calculate it?
mAP (mean Average Precision) is the most common metric used to evaluate object detection models.

Steps to compute mAP:

1. For each class:

    - Plot the Precision–Recall curve using detections and ground truth
    - Compute the Average Precision (AP) (area under that curve)
2. Average across all classes:

```mAP = (1 / N) * Σᵢ₌₁ⁿ (APᵢ)```


Interpretation:

- Precision → proportion of correct detections
- Recall → proportion of true objects found
- mAP → overall balance between precision and recall (higher = better)