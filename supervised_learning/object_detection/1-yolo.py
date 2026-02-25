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

    def process_outputs(self, outputs, image_size):
        """
        Method process_outputs that processes the outputs:
            outputs: a list of numpy.ndarrays containing the predictions from
                the Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width => the height and width of the
                    grid used for the output
                    anchor_boxes => the number of anchor boxes used
                    for each prediction
                    4 => (t_x, t_y, t_w, t_h)
                    1 => box_confidence
                    classes => class probabilities for all classes
            image_size: a numpy.ndarray containing the original size of the
                image [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
                anchor_boxes, 4) containing the processed boundary boxes for
                each output, respectively:
                4 => (x1, y1, x2, y2)
                    (x1, y1, x2, y2) should represent the boundary box relative
                    to original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
                grid_width, anchor_boxes, 1) containing the box confidences for
                each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
                grid_width, anchor_boxes, classes) containing the box class
                probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            grid_h, grid_w, anchor_boxes, _ = output.shape
            box = np.zeros(output[..., :4].shape)
            for i in range(grid_h):
                for j in range(grid_w):
                    for k in range(anchor_boxes):
                        t_x, t_y, t_w, t_h = output[i, j, k, :4]
                        c_x = j
                        c_y = i
                        p_w = self.anchors[k][0]
                        p_h = self.anchors[k][1]
                        s_w = image_size[1]
                        s_h = image_size[0]
                        b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
                        b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h
                        b_w = (p_w * np.exp(t_w)) / s_w
                        b_h = (p_h * np.exp(t_h)) / s_h
                        x1 = int((b_x - (b_w / 2)) * s_w)
                        y1 = int((b_y - (b_h / 2)) * s_h)
                        x2 = int((b_x + (b_w / 2)) * s_w)
                        y2 = int((b_y + (b_h / 2)) * s_h)
                        box[i, j, k] = [x1, y1, x2, y2]
            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))
        return (boxes, box_confidences, box_class_probs)
