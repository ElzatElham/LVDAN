from ultralytics import YOLO

from pathlib import Path
import numpy as np





from ultralytics import YOLO

from pathlib import Path
import numpy as np





## Train the model
detection_model_path = '/home/Jet/ultralytics-main/2nd_paper/model/iCT_best.pt'  ## pre-trained model
detection_model = YOLO(detection_model_path)  # build a new model from YAML




results      = detection_model.train(
    data     = '/home/Jet/ultralytics-main/2nd_paper/data_yaml/iCT2SOMATOM.yaml',           # source domain data
    datat    = '/home/Jet/ultralytics-main/2nd_paper/data_yaml/Unlabeled_SOMATOM.yaml',     # target domain data
    LLDA     = 0.001,                                                                      # parameter for LDA loss
    LGDA     = 2,                                                                      # parameter for GDA loss
    LTSD     = 0.001,                                                                      # parameter for TSD loss
    iou      = 0.5,
    conf     = 0.1, 
    epochs   = 200, 
    imgsz    = 512, 
    patience = 50, 
    batch    = 16, 
    fliplr   = 0,
    flipud   = 0,
    name     = 'new_model.08.13.iCT2SOMATOM.0.0001',                                       
    device   = '0,1,2,3')


