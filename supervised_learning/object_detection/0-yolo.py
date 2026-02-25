#!/usr/bin/env python3
"""Object detection with YOLOv3."""
import tensorflow as tf
import numpy as np
import cv2


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm to perform object detection:
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo that initializes the Yolo v3 model
        with the following:
            model_path: the path to where a Darknet Keras model is stored
            classes_path: the path to where the list of class names used for
                the Darknet model, listed in order of index, can be found
            class_t: a float representing the box score threshold for the
                initial filtering step
            nms_t: a float representing the IOU threshold
            for non-maximum suppression
            anchors: a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing the anchor boxes:
                outputs is the number of outputs (predictions)
            made by the Darknet model
                anchor_boxes is the number of anchor boxes used
            for each prediction
                2 => [anchor_box_width, anchor_box_height]
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
