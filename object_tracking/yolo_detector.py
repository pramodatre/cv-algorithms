#
# Created on Sat Mar 12 2022
#
# Copyright (c) 2022 Pramod Anantharam <pramod.atre@gmail.com>
#
import os
import cv2
import pickle
import pandas as pd
import numpy as np


class YOLOdetector:
    def __init__(self, image_dir, save_file_name="yolo_detections.pickle"):
        self.image_dir = image_dir
        self.cfg = "./data/yolov3.cfg"
        self.weights = "./data/yolov3.weights"
        self.save_file_name = save_file_name
        self.image_size = 416

    def run(self):
        detections = {}
        LABEL_COLORS = {
            "car": [255, 0, 0],
            "pedestrian": [0, 255, 0],
            "truck": [0, 0, 255],
            "trafficLight": [255, 255, 0],
            "biker": [255, 0, 255],
        }
        cap = cv2.VideoCapture(os.path.join(self.image_dir, "frame_%04d.jpg"))
        # frame index so that we can retrieve GT bounding boxes
        frame_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            boxes = self.yolo_v3(frame, 0.5, 0.5)
            # save detections to a pickle file
            detections[frame_count] = boxes
            with open(self.save_file_name, "wb") as handle:
                pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(boxes)
            for row in boxes.itertuples():
                # print(row.xmin, row.ymin, row.xmax, row.ymax)
                label = row.labels
                (startX, startY, endX, endY) = row.xmin, row.ymin, row.xmax, row.ymax
                # display the prediction
                # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                # print("[INFO] {}".format(label))
                cv2.rectangle(
                    frame, (startX, startY), (endX, endY), LABEL_COLORS[label], 2
                )
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    frame,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    LABEL_COLORS[label],
                    2,
                )

            frame_count += 1
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == 27:
                exit(0)

    def yolo_v3(self, image, confidence_threshold, overlap_threshold):
        # Load the network. Because this is cached it will only happen once.
        def load_network(config_path, weights_path):
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            output_layer_names = net.getLayerNames()
            output_layer_names = [
                output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()
            ]
            return net, output_layer_names

        net, output_layer_names = load_network(self.cfg, self.weights)

        # Run the YOLO neural net.
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (self.image_size, self.image_size),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        layer_outputs = net.forward(output_layer_names)

        # Supress detections in case of too low confidence or too much overlap.
        boxes, confidences, class_IDs = [], [], []
        H, W = image.shape[:2]
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, overlap_threshold
        )

        # Map from YOLO labels to Udacity labels.
        UDACITY_LABELS = {
            0: "pedestrian",
            1: "biker",
            2: "car",
            3: "biker",
            5: "truck",
            7: "truck",
            9: "trafficLight",
        }
        xmin, xmax, ymin, ymax, labels = [], [], [], [], []
        if len(indices) > 0:
            # loop over the indexes we are keeping
            for i in indices.flatten():
                label = UDACITY_LABELS.get(class_IDs[i], None)
                if label is None:
                    continue

                # extract the bounding box coordinates
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

                xmin.append(x)
                ymin.append(y)
                xmax.append(x + w)
                ymax.append(y + h)
                labels.append(label)

        boxes = pd.DataFrame(
            {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels}
        )
        return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]
