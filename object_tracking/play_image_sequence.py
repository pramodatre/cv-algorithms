import os
import cv2
import argparse
import xml.etree.ElementTree as ET

# GT XML file: http://www.milanton.de/data/ (http://www.milanton.de/files/gt/PETS2009/PETS2009-S2L1.xml)
# Image sequence: http://cs.binghamton.edu/~mrldata/pets2009 (http://cs.binghamton.edu/~mrldata/public/PETS2009/S2_L1.tar.bz2)
# Example command:
# python play_image_sequence.py --image_dir '/Users/pramodanantharam/dev/data/pets2009/Crowd_PETS09/S2/L1/Time_12-34/View_001' --gt '/Users/pramodanantharam/dev/data/pets2009/PETS2009-S2L1.xml'

# Annotations GT from: http://www.milanton.de/data/
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_dir", required=True, help="Specify the directory containing image sequence"
)
parser.add_argument(
    "--gt", required=False, help="Specify the GT file if any for the image sequence"
)

args = parser.parse_args()
# read GT file if specified
gt_dict = {}
if args.gt is not None:
    tree = ET.parse(args.gt)
    root = tree.getroot()
    # print(root)
    for child in root:
        # print(child.tag, child.attrib)
        object_list = child[0]
        object_dict = {}
        for obj in object_list:
            # print(obj.tag, obj.attrib)
            # print(obj[0].attrib)
            object_dict[obj.attrib["id"]] = obj[0].attrib
        gt_dict[int(child.attrib["number"])] = object_dict

# print(gt_dict)

cap = cv2.VideoCapture(os.path.join(args.image_dir, "frame_%04d.jpg"))
# frame index so that we can retrieve GT bounding boxes
frame_count = 1
pause = False

while cap.isOpened():
    if pause:
        c = cv2.waitKey(33)
        if c == 32:  # space bar pressed
            if not pause:
                pause = True
            else:
                pause = False
        continue
    ret, frame = cap.read()
    # plot person bounding boxes
    # print(gt_dict[frame_count])
    print(frame.shape)
    frame_gt = gt_dict[frame_count]
    for k in frame_gt:
        object_id = k
        print(frame_gt[k])
        x, y, w, h = (
            frame_gt[k]["xc"],
            frame_gt[k]["yc"],
            frame_gt[k]["w"],
            frame_gt[k]["h"],
        )
        xc = int(float(x))
        yc = int(float(y))
        w = int(float(w))
        h = int(float(h))
        # print(x,y,w,h)
        # xc and yc are center of the bounding box
        # we need top left cooridnate for plotting using OpenCV
        x = int(xc - w / 2.0)
        y = int(yc - h / 2.0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            object_id,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (51, 255, 255),
            thickness=2,
        )

    cv2.imshow("frame", frame)
    frame_count += 1

    c = cv2.waitKey(33)
    if c == 32:  # space bar pressed
        if not pause:
            pause = True
        else:
            pause = False
    if c & 0xFF == ord("q"):
        break
