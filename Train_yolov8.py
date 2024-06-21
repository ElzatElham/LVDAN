from ultralytics import YOLO

from pathlib import Path
import numpy as np





## Train the model
detection_model_path = '/Volumes/ZelinDisk/big/YJT_DA/model/Revolution_best.pt'
detection_model = YOLO(detection_model_path)  # build a new model from YAML
results      = detection_model.train(
    data     = './data/iCT2SOMATOM.yaml',
    datat    = './data/Unlabeled_SOMATOM.yaml',
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
    device   = 'cpu')

