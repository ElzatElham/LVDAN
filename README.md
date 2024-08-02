#  LVDAN
This is the official PyTorch implementation of our paper: 

**Unsupervised Domain Adaptation of Object Detection in Axial CT Images of Lumbar Vertebrae**


# Installation

Clone repo and install [requirements.txt](https://github.com/ElzatElham/DA-YOLOV8/blob/main/requirements.txt) in a Python>=3.7 environment, including PyTorch>=1.13.1.


```
# git clone
git clone https://github.com/ElzatElham/DA-YOLOV8.git  # clone

# Create a virtual environment
conda create -n da-yolo python=3.7

conda activate da-yolo

# Enter the DA-YOLO directory
cd LVDAN-main

# install requirement
pip install -r requirements.txt

# Install ultralytics
python setup.py install

```

# t-SNE Visualization
The visualization code is located at: DA-YOLOV8/t-sne/plt_t-sne.py 

```
python DA-YOLOV8/t-sne/plt_t-sne.py --Input_path image_dir --size 2 512
```

--Input_path: All images are directly placed in this directory. Previous results used all images from those 4 datasets. 

--size: The size to resize the images. Since all images are square and have a size of 512, if size=512, it means visualizing the original images. If size=2, a large amount of data has been resized, but the results are better. If you want to plot multiple images with different resizes simultaneously, fill in multiple target size numbers separated by spaces.



