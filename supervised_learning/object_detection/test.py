import cv2
import numpy as np

img = cv2.imread('test_files/yolo_images/dog.jpg', 0)
cv2.namedWindow('dog')
cv2.imshow('dog', img)
cv2.waitKey(0)
cv2.destroyAllWindows()