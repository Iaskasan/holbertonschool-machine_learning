#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3
algorithm to perform object detection"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import os
import cv2


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
            (grid_h, grid_w, anchor_boxes, 4 + 1 + classes).
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
            each containing the confidence
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the boxes based on object and class confidence thresholds.

        Parameters
        ----------
        boxes : list of numpy.ndarrays
            List of arrays of shape (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output.
        box_confidences : list of numpy.ndarrays
            List of arrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output.
        box_class_probs : list of numpy.ndarrays
            List of arrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the processed box class probabilities for each output.

        Returns
        -------
        filtered_boxes : numpy.ndarray of shape (?, 4)
            All filtered bounding boxes.
        box_classes : numpy.ndarray of shape (?,)
            Class index predicted for each box.
        box_scores : numpy.ndarray of shape (?)
            Confidence score for each box.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, bc, bcp in zip(boxes, box_confidences, box_class_probs):
            # Compute box scores per class: (grid_h, grid_w, anchors, classes)
            scores = bc * bcp

            # Get the class with the highest probability for each box
            classes = np.argmax(scores, axis=-1)    # (grid_h, grid_w, anchors)
            class_scores = np.max(scores, axis=-1)  # (grid_h, grid_w, anchors)

            # Apply the threshold to keep only high-confidence boxes
            mask = class_scores >= self.class_t         # boolean mask

            # Extract the filtered boxes and corresponding data
            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        # Concatenate results from all outputs into single arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to remove
        redundant overlapping boxes.

        Parameters
        ----------
        filtered_boxes : numpy.ndarray of shape (N, 4)
            All filtered bounding boxes from filter_boxes
            in the format (x1, y1, x2, y2).
        box_classes : numpy.ndarray of shape (N,)
            Class index predicted for each box.
        box_scores : numpy.ndarray of shape (N,)
            Confidence score for each box.

        Returns
        -------
        box_predictions : numpy.ndarray of shape (?, 4)
            The final predicted bounding boxes after NMS,
            ordered by (class, score).
        predicted_box_classes : numpy.ndarray of shape (?,)
            The class index for each predicted box,
            ordered the same way.
        predicted_box_scores : numpy.ndarray of shape (?)
            The score for each predicted box,
            ordered the same way.
        """
        # We'll collect final kept boxes for all classes here:
        final_boxes = []
        final_classes = []
        final_scores = []

        unique_classes = np.unique(box_classes)

        # Process boxes class by class
        for cls in unique_classes:
            # 1. Get all boxes of this class
            cls_mask = (box_classes == cls)

            cls_boxes = filtered_boxes[cls_mask]   # shape (K, 4)
            cls_scores = box_scores[cls_mask]      # shape (K,)

            # 2. Sort them by score (highest first)
            order = np.argsort(-cls_scores)  # descending
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep_indices = []

            # 3. Perform NMS loop
            while len(cls_boxes) > 0:
                # Take the best box (first one, highest score)
                best_box = cls_boxes[0]
                best_score = cls_scores[0]

                keep_indices.append((best_box, cls, best_score))

                if len(cls_boxes) == 1:
                    break

                # Compare this best box to the rest
                rest_boxes = cls_boxes[1:]

                # Compute IoU between best_box and all rest_boxes
                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                # Intersection area
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                # Areas of boxes
                best_area = (
                    (best_box[2] - best_box[0]) *
                    (best_box[3] - best_box[1])
                )
                rest_areas = (
                    (rest_boxes[:, 2] - rest_boxes[:, 0]) *
                    (rest_boxes[:, 3] - rest_boxes[:, 1])
                )

                # Union = A + B - intersection
                union_area = best_area + rest_areas - inter_area

                # IoU with each remaining box
                iou = inter_area / union_area

                # 4. Keep only boxes whose IoU is below the NMS threshold
                # (i.e. boxes that are not "too overlapping")
                low_overlap_mask = iou < self.nms_t

                # Update the candidate list of boxes/scores
                cls_boxes = rest_boxes[low_overlap_mask]
                cls_scores = cls_scores[1:][low_overlap_mask]

            # Save the kept results for this class
            for b, c, s in keep_indices:
                final_boxes.append(b)
                final_classes.append(c)
                final_scores.append(s)

        # Convert lists to np arrays
        final_boxes = np.array(final_boxes)
        final_classes = np.array(final_classes)
        final_scores = np.array(final_scores)

        # 5. Order final results by (class, score desc)
        sort_key = np.lexsort((
            -final_scores,      # secondary: score descending
            final_classes       # primary: class ascending
        ))
        final_boxes = final_boxes[sort_key]
        final_classes = final_classes[sort_key]
        final_scores = final_scores[sort_key]

        return final_boxes, final_classes, final_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specified folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the images.

        Returns
        -------
        images : list of numpy.ndarray
            List containing all the loaded images as NumPy arrays (BGR format).
        image_paths : list of str
            List of the corresponding image file paths.
        """
        images = []
        image_paths = []

        # Get all file names in the folder
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        all_files = os.listdir(folder_path)

        # Filter only image files
        for file_name in all_files:
            if file_name.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, file_name)
                image = cv2.imread(img_path)

                # Check if the image was successfully loaded
                if image is not None:
                    images.append(image)
                    image_paths.append(img_path)
                else:
                    print(f"⚠️ Warning: Failed to load {img_path}")

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses the images for input into the Darknet model.

        Parameters
        ----------
        images : list of numpy.ndarray
            List of images to preprocess.

        Returns
        -------
        pimages : numpy.ndarray
            Array of preprocessed images ready for model input.
        image_shapes : numpy.ndarray
            Array containing the original shapes of the images.
        """
        pimages = []
        image_shapes = []

        for img in images:
            original_shape = img.shape[:2]  # (height, width)
            image_shapes.append(original_shape)

            # Resize to model input size (e.g., 416x416)
            resized_img = cv2.resize(img, (self.model.input.shape[1],
                                           self.model.input.shape[2]),
                                     interpolation=cv2.INTER_CUBIC)

            # Normalize pixel values to [0, 1]
            normalized_img = resized_img / 255.0

            pimages.append(normalized_img)

        return np.array(pimages), np.array(image_shapes)
