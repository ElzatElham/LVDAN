# DA-YOLOV8
This is the official PyTorch implementation of our paper: 

**DA-YOLOv8**


# Installation



Pip install the [YOLOv8](https://github.com/ultralytics/ultralytics) package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.
```
# Create a virtual environment
conda create -n da-yolov8 python=3.8
# Install YOLOv8
pip install ultralytics
# Replace the content in /root/anaconda3/envs/da-yolov8/lib/python3.8/site-packages/ultralytics with the code from master.
```

The main modification is the loss function, located in Urtlalitic/Yolo/Uttils


# t-SNE Visualization
The visualization code is located at: DA-YOLOV8/t-sne/plt_t-sne.py 

```
python DA-YOLOV8/t-sne/plt_t-sne.py --Input_path image_dir --size 2 256
```

--Input_path: All images are directly placed in this directory. Previous results used all images from those 4 datasets. 

--size: The size to resize the images. Since all images are square and have a size of 512, if size=512, it means visualizing the original images. If size=2, a large amount of data has been resized, but the results are better. If you want to plot multiple images with different resizes simultaneously, fill in multiple target size numbers separated by spaces.

