# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:07:48 2024

@author: 13621
"""
from ultralytics import YOLO

from pathlib import Path
import numpy as np



model_dir = r"/Data/PHD/test/06.11.1/runs/detect/06.11.1.Det.test.SOM/weights/best.pt"     # LVDAN model

model = YOLO(model_dir)
datadir = '/Data/PHD/test/06.11.1/test_OD/yaml/06.11.1.Det.test.iCT2SOM.yaml'    # target domain data





print('----------------------------------------------iCT--SOMATOM-----------------------------------------------------------')


metrics = model.val(data=datadir, device   = '0,1,2,3',conf= 0.1,iou = 0.5,batch = 16)

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
Map50 = float(f"{metrics.box.map50*100:.1f}")
Map75 = float(f"{metrics.box.map75*100:.1f}")
Map   = float(f"{metrics.box.map*100:.1f}")























