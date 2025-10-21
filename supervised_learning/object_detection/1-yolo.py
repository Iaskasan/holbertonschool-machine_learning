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
        self.model = load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the Darknet model for a single image.

        Parameters
        ----------
        outputs : list of numpy.ndarrays
            Predictions from the Darknet model for one image.
            Each array has shape
            (grid_h, grid_w, anchor_boxes,4 + 1 + classes).
        image_size : numpy.ndarray
            Original image size [image_height, image_width].

        Returns
        -------
        boxes : list of numpy.ndarrays
            Shape (grid_h, grid_w, anchor_boxes, 4),
            each containing the processed
            boundary boxes (x1, y1, x2, y2) relative to the original image.
        box_confidences : list of numpy.ndarrays
            Shape (grid_h, grid_w, anchor_boxes, 1),
            ach containing the confidence
            that the box contains an object.
        box_class_probs : list of numpy.ndarrays
            Shape (grid_h, grid_w, anchor_boxes, classes), each containing
            the class probabilities for each predicted box.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]

            # --- Split the components ---
            t_xy = output[..., :2]      # (t_x, t_y)
            t_wh = output[..., 2:4]     # (t_w, t_h)
            objectness = output[..., 4:5]  # confidence for object presence
            class_probs = output[..., 5:]  # class probabilities

            # --- Create grid for each cell ---
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            grid = np.stack((cx_grid, cy_grid), axis=-1)
            grid = grid[..., np.newaxis, :]  # shape (grid_h, grid_w, 1, 2)

            # --- Decode boxes ---
            bx_by = (1 / (1 + np.exp(-t_xy)) + grid)  # sigmoid(t_xy)
            bw_bh = (self.anchors[i] * np.exp(t_wh))  # anchor * exp(t_wh)

            # Normalize to original image size
            bx_by /= [grid_w, grid_h]
            bw_bh /= [self.model.input.shape[1], self.model.input.shape[2]]

            # Convert to corners (x1, y1, x2, y2)
            x1y1 = bx_by - (bw_bh / 2)
            x2y2 = bx_by + (bw_bh / 2)
            box = np.concatenate((x1y1, x2y2), axis=-1)

            # Scale to image pixels
            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)

            # --- Apply sigmoid to confidence and class probs ---
            box_confidence = 1 / (1 + np.exp(-objectness))
            box_class_prob = 1 / (1 + np.exp(-class_probs))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
