#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3
algorithm to perform object detection"""
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore


class Yolo:
    """Class yolo"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Parameters
        ----------
        model_path : str
            Path to where a Darknet Keras model is stored.

        classes_path : str
            Path to where the list of class names used for the Darknet model,
            listed in order of index, can be found.

        class_t : float
            Box score threshold for the initial filtering step.
            Boxes with a confidence score lower than this value
            will be ignored.

        nms_t : float
            IOU (Intersection Over Union) threshold for non-max suppression.
            Used to suppress overlapping bounding boxes for the same object.

        anchors : numpy.ndarray of shape (outputs, anchor_boxes, 2)
            Array containing all of the anchor boxes used by the model.
            - outputs: number of outputs (prediction scales) made by the
            Darknet model
            - anchor_boxes: number of anchor boxes used for each prediction
            - 2: [anchor_box_width, anchor_box_height]
        """
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
