Welcome to the object tracking repo! Here, you will find instructions to prepare dataset and evaluate various tracking algorithms. You can refer to my blog post on object tracking for more details. Here, you will find practical instructions on getting the code running on your machine.

# Datasets
All required datasets will be downloaded and placed in appropriate directories by running the following script. This script will download to `./data`:
* PETS2009 image sequence S2L1
* PETS2009 S2L1 ground-truth annotations XML 
* YOLOv3 cfg file
* YOLOv3 weights file
```
python prepare_data.py
```

# Ground truth
You can visualize ground-truth video along with the bounding boxes using:
```
python play_image_sequence.py --image_dir './data/Crowd_PETS09/S2/L1/Time_12-34/View_001' --gt './data/PETS2009-S2L1.xml'
```

# Trackers
You can run the following command to run all trackers on the PETS2009 image sequence S2L1.
```
python trackers.py --image_dir './data/Crowd_PETS09/S2/L1/Time_12-34/View_001'
```

# Calculating HOTA metrics
## Perfect tracker
```
python scripts/run_mot_challenge.py --BENCHMARK OBJTR22 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL PerfectTracker --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1 
```
## Baseline tracker
```
python scripts/run_mot_challenge.py --BENCHMARK OBJTR22 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL OBJTR22 --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1 
```
## Detection based tracker
```
python scripts/run_mot_challenge.py --BENCHMARK OBJTR22 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL TrackByDetection --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1 
```
