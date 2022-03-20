#
# Created on Sat Mar 12 2022
#
# Copyright (c) 2022 Pramod Anantharam <pramod.atre@gmail.com>
#
import os
import pickle
import argparse
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from yolo_detector import YOLOdetector
from abc import ABC, abstractclassmethod

# GT XML file: http://www.milanton.de/data/ (http://www.milanton.de/files/gt/PETS2009/PETS2009-S2L1.xml)
# Image sequence: http://cs.binghamton.edu/~mrldata/pets2009 (http://cs.binghamton.edu/~mrldata/public/PETS2009/S2_L1.tar.bz2)

# Sample invocation
# python trackers.py --image_dir './data/Crowd_PETS09/S2/L1/Time_12-34/View_001' --gt './data/PETS2009-S2L1.xml'


class Frame:
    def __init__(self) -> None:
        self.image = None
        self.detections = []

    def get_detections(self):
        return self.detections

    def get_image(self):
        return self.image


class Detection:
    def __init__(self, xmin, ymin, xmax, ymax) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def get_xmin_ymin_xmax_ymax(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class FrameDataReader:
    def __init__(
        self, image_dir, saved_detections_file="yolo_detections.pickle"
    ) -> None:
        self.frames = []
        if not os.path.exists(saved_detections_file):
            print(
                f"Could not find saved detections file: {saved_detections_file}. Will have to run YOLO on your machine which may be slow the first time. Results will be cached for future runs."
            )
            yolo = YOLOdetector(image_dir)
            yolo.run()
        else:
            with open(saved_detections_file, "rb") as handle:
                self.detections = pickle.load(handle)
            frame_count = 1
            cap = cv2.VideoCapture(os.path.join(image_dir, "frame_%04d.jpg"))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                f = Frame()
                f.image = frame
                boxes = self.detections[frame_count]
                dets = []
                for row in boxes.itertuples():
                    d = Detection(row.xmin, row.ymin, row.xmax, row.ymax)
                    dets.append(d)
                f.detections = dets
                self.frames.append(f)
                frame_count += 1

    def get_all_frames(self):
        return self.frames


class TrackingStrategy(ABC):
    @abstractclassmethod
    def track_objects(self):
        pass


class Tracker:
    def __init__(self, strategy: TrackingStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> TrackingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: TrackingStrategy):
        self._strategy = strategy

    def start_tracking(self) -> None:
        self._strategy.track_objects()


class BaselineTracker(TrackingStrategy):
    def __init__(self, image_dir) -> None:
        super().__init__()
        self.frames = FrameDataReader(image_dir).get_all_frames()
        print(self.frames)

    def predict_object_continuation(self, box_t, prev_frame_objects):
        # Select last position for each object and find overlap to box_t
        # print(prev_frame_objects)
        max_i = 0
        overlap_area = 0
        best_id = -1
        for o_id in prev_frame_objects:
            b = prev_frame_objects[o_id]
            xmin, ymin, xmax, ymax = box_t
            xmin2, ymin2, xmax2, ymax2 = b
            x_overlap = min(xmax, xmax2) - max(xmin, xmin2)
            y_overlap = min(ymax, ymax2) - max(ymin, ymin2)
            # Must check if there is a overlap in x and y direction
            # before computing overlap area. Otherwise, we may end
            # up with +ve area of overlap with both x and y direction
            # overlap is -ve
            if x_overlap > 0 and y_overlap > 0:
                overlap_area = x_overlap * y_overlap
                if overlap_area > max_i:
                    max_i = overlap_area
                    best_id = o_id
            else:
                continue

            # print(iou, overlap_area, union_area)

        if max_i > 0:
            return best_id

        return -1

    def track_objects(self):
        print("Starting the BaselineTracker ...")
        prev_frame_objects = {}
        frame_count = 1
        o_id = 1
        tracked_detections = []
        for frame in self.frames:
            cur_frame_objects = {}
            print(prev_frame_objects)
            cur_obj_id = o_id
            dets = frame.get_detections()
            image = frame.get_image()
            for det in dets:
                (startX, startY, endX, endY) = det.get_xmin_ymin_xmax_ymax()
                bb_left = startX
                bb_top = startY
                w = endX - startX
                h = endY - startY
                if frame_count == 1:
                    cur_frame_objects[cur_obj_id] = (startX, startY, endX, endY)
                    o_id += 1
                    cur_obj_id = o_id
                    tracked_detections.append(
                        {
                            "frame": frame_count,
                            "id": cur_obj_id,
                            "bb_left": bb_left,
                            "bb_top": bb_top,
                            "bb_width": w,
                            "bb_height": h,
                            "conf": 1,
                            "x": -1,
                            "y": -1,
                            "z": -1,
                        }
                    )
                else:
                    box_t = (startX, startY, endX, endY)
                    # find best matching object id from previous frame
                    pred_id = self.predict_object_continuation(
                        box_t, prev_frame_objects
                    )
                    # if a valid id is found, use the current detection
                    # as a continuation of the previous detection
                    if pred_id > 0:
                        cur_obj_id = pred_id
                        # Sometimes a single detection may map to the multiple
                        # object in the previous frame if they are too close
                        # as we are not using bipartite matching.
                        if pred_id not in cur_frame_objects:
                            tracked_detections.append(
                                {
                                    "frame": frame_count,
                                    "id": cur_obj_id,
                                    "bb_left": bb_left,
                                    "bb_top": bb_top,
                                    "bb_width": w,
                                    "bb_height": h,
                                    "conf": 1,
                                    "x": -1,
                                    "y": -1,
                                    "z": -1,
                                }
                            )
                        cur_frame_objects[cur_obj_id] = (startX, startY, endX, endY)
                    else:
                        o_id += 1
                        cur_obj_id = o_id
                        cur_frame_objects[cur_obj_id] = (startX, startY, endX, endY)
                        tracked_detections.append(
                            {
                                "frame": frame_count,
                                "id": cur_obj_id,
                                "bb_left": bb_left,
                                "bb_top": bb_top,
                                "bb_width": w,
                                "bb_height": h,
                                "conf": 1,
                                "x": -1,
                                "y": -1,
                                "z": -1,
                            }
                        )
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(
                    image,
                    str(cur_obj_id),
                    (endX, startY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    thickness=2,
                )
            cv2.imshow("image", image)
            prev_frame_objects = cur_frame_objects
            frame_count += 1

            c = cv2.waitKey(33)
            if c == 32:  # space bar pressed
                if not pause:
                    pause = True
                else:
                    pause = False
            if c & 0xFF == ord("q"):
                break

        det_df = pd.DataFrame.from_dict(tracked_detections)
        det_df.to_csv("tracker_baseline.txt", index=False)


class DetectionBasedTracker(TrackingStrategy):
    def __init__(self, image_dir) -> None:
        super().__init__()
        self.frames = FrameDataReader(image_dir).get_all_frames()
        print(self.frames)
        self.o_id_count = 1
        self.o_ids_without_updates_counts = defaultdict(lambda: 0)
        self.STALE_DET_THRESHOLD_FRAMES = 10

    def compute_iou_score(self, bbox1, bbox2):
        """Returns Intersection over Union (IoU) score for two
        bounding boxes represented by (xmin, ymin, xmax, ymax)

        (xmin1, ymin1)
            .

                . (xmin2, ymin2)

                    . (xmax1, ymax1)

                            . (xmax2, ymax2)
        Args:
            bbox1 (tuple): (xmin, ymin, xmax, ymax) of box1
            bbox2 (tuple): (xmin, ymin, xmax, ymax) of box2
        """
        box1A = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        box2A = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        x_overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
        y_overlap = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
        intersectionA = max(0, x_overlap) * max(0, y_overlap)
        iou = intersectionA / float(box1A + box2A - intersectionA)
        return iou

    def predict_object_continuation_using_bipartite_matching(self, cur_det, obj_map):
        """Connect objects in previous frame (obj_map) to objects in
        the current frame (cur_det) using an optimization technique.

        Args:
            cur_det (list): Containing Detection objects; one
                detection object per detection
            obj_map (dict): Containing object_id as key and
                corresponding bounding box as value

        Returns:
            dict: Updated obj_map
        """
        if not obj_map:
            # First frame
            for det in cur_det:
                obj_map[self.o_id_count] = det.get_xmin_ymin_xmax_ymax()
                self.o_id_count += 1
        else:
            # Rest of the frames
            cur_det_dict = {}
            rows, cols = len(list(obj_map.keys())), len(cur_det)
            cost_matrix = np.zeros((rows, cols))
            cost_martix_df = pd.DataFrame(
                data=cost_matrix, index=list(obj_map.keys()), columns=list(range(cols))
            )
            # read all detections to a dictionary
            det_count = 0
            for c_det in cur_det:
                det_bbox = c_det.get_xmin_ymin_xmax_ymax()
                print(det_bbox)
                cur_det_dict[det_count] = det_bbox
                det_count += 1

            for o_id in obj_map:
                for det_id in cur_det_dict:
                    det_bbox = cur_det_dict[det_id]
                    iou_score = self.compute_iou_score(obj_map[o_id], det_bbox)
                    if iou_score > 0:
                        cost_martix_df.loc[o_id, det_id] = iou_score
            cost_martix_array = cost_martix_df.values * -1
            row_ind, col_ind = linear_sum_assignment(cost_martix_array)
            # Update object map with best assignments
            for i, j in zip(row_ind, col_ind):
                obj_id = list(cost_martix_df.index)[i]
                det_id = cost_martix_df.columns[j]
                if cost_martix_array[i, j] == 0:
                    obj_map[self.o_id_count] = cur_det_dict[det_id]
                    self.o_id_count += 1
                else:
                    obj_map[obj_id] = cur_det_dict[det_id]

        return obj_map

    def prune_tracks(self, cur_obj_map, prev_obj_map):
        """Remove objects whose positions are not updated for certain frames.

        Args:
            cur_obj_map (dict): Object id and corresponding
                        detection bounding box
            prev_obj_map (dict): Previous frame object id and
                         corresponding bounding box

        Returns:
            dict: Pruned cur_obj_map
        """
        print(f"comparing {cur_obj_map} and {prev_obj_map}")
        for cur_id in cur_obj_map:
            if cur_id in prev_obj_map:
                if cur_obj_map[cur_id] == prev_obj_map[cur_id]:
                    self.o_ids_without_updates_counts[cur_id] += 1

        keys_to_remove = []
        for cur_id in self.o_ids_without_updates_counts:
            if (
                self.o_ids_without_updates_counts[cur_id]
                > self.STALE_DET_THRESHOLD_FRAMES
            ):
                keys_to_remove.append(cur_id)

        for cur_id in keys_to_remove:
            del cur_obj_map[cur_id]
            del self.o_ids_without_updates_counts[cur_id]

        return cur_obj_map

    def track_objects(self):
        print("Starting the DetectionBasedTracker ...")
        cur_obj_map = {}
        prev_obj_map = None
        frame_count = 1
        tracked_detections = []
        for frame in self.frames:
            cur_frame_detections = frame.get_detections()
            image = frame.get_image()
            cur_obj_map = self.predict_object_continuation_using_bipartite_matching(
                cur_frame_detections, cur_obj_map
            )
            for obj_id in cur_obj_map:
                (startX, startY, endX, endY) = cur_obj_map[obj_id]
                bb_left = startX
                bb_top = startY
                w = endX - startX
                h = endY - startY
                tracked_detections.append(
                    {
                        "frame": frame_count,
                        "id": obj_id,
                        "bb_left": bb_left,
                        "bb_top": bb_top,
                        "bb_width": w,
                        "bb_height": h,
                        "conf": 1,
                        "x": -1,
                        "y": -1,
                        "z": -1,
                    }
                )
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(
                    image,
                    str(obj_id),
                    (endX, startY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    thickness=2,
                )
            cv2.imshow("frame", image)
            c = cv2.waitKey(33)
            if c == 32:  # space bar pressed
                if not pause:
                    pause = True
                else:
                    pause = False
            if c & 0xFF == ord("q"):
                break

            if prev_obj_map is not None:
                cur_obj_map = self.prune_tracks(cur_obj_map, prev_obj_map)

            prev_obj_map = cur_obj_map.copy()
            frame_count += 1

        det_df = pd.DataFrame.from_dict(tracked_detections)
        det_df.to_csv("tracker_detection_based.txt", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Specify the directory containing image sequence",
    )
    args = parser.parse_args()
    tracker = Tracker(BaselineTracker(args.image_dir))
    tracker.start_tracking()
    tracker.strategy = DetectionBasedTracker(args.image_dir)
    tracker.start_tracking()
