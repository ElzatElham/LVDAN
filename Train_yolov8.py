from ultralytics import YOLO

from pathlib import Path
import numpy as np





## Train the model
detection_model_path = '/Data/YLS/2nd_paper_model/yolov8/runs/detect/Revolution_best.pt'
detection_model = YOLO(detection_model_path)  # build a new model from YAML
results      = detection_model.train(
    data     = '/home/Jet/ultralytics-main/2nd_paper/data_yaml/Revolution2iCT.yaml',
    datat    = '/home/Jet/ultralytics-main/2nd_paper/data_yaml/Unlabeled_iCT.yaml',
    da       = 0.00006,
    iou      = 0.5,
    conf     = 0.1, 
    epochs   = 200, 
    imgsz    = 512, 
    patience = 50, 
    batch    = 16, 
    fliplr   = 0,
    flipud   = 0,
    name     = '04.25.1.Revolution2iCT_0.00006', 
    device   = '0,1,2,3')

