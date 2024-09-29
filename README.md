# LVDAN: Lumbar Vertebrae Detection and Adaptive Network

Welcome to the official PyTorch implementation of our groundbreaking research paper: 

**Unsupervised Domain Adaptation of Object Detection in Axial CT Images of Lumbar Vertebrae**

## Overview

LVDAN leverages advanced unsupervised domain adaptation techniques to enhance object detection performance in axial CT images of lumbar vertebrae. This model is designed to improve accuracy and robustness in medical imaging applications, addressing the challenges posed by domain shifts in data.

### Key Advantages:
- **High Accuracy**: Achieve superior detection rates in challenging medical imaging scenarios.
- **Robustness**: Effectively adapts to variations in image quality and acquisition conditions.
- **User-Friendly**: Simplified installation and training processes for seamless integration into your workflow.

## Pre-training Installation

To get started, clone the repository and install the required dependencies in a Python environment (version >= 3.7) with PyTorch (version >= 1.13.1).

### Steps:

1. Create and activate a virtual environment:
   ```bash
   conda create -n yolo python=3.7
   conda activate yolo
   ```

3. Install the required packages:
   ```bash
   pip install ultralytics
   ```

4. Download the pretrained YOLOv8x model:
   - [YOLOv8x Pretrained Model](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)

## DA-training Installation

Follow these steps to set up the DA-training environment:

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/ElzatElham/LVDAN.git
   ```

2. Create and activate a new virtual environment:
   ```bash
   conda create -n LVDAN python=3.7
   conda activate LVDAN
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install LVDAN:
   ```bash
   python setup.py install
   ```

## Training and Testing

To train and test the model, use the following commands:

### Training:
```bash
python Training.py
```

### Testing:
```bash
python Testing.py
```

All training and testing parameters can be found within `Training.py` and `Testing.py`. 

Additionally, the dataset used in the paper is open-sourced and can be accessed here: [CTLV-DAOD](https://github.com/ElzatElham/CTLV-DAOD).

---

By following these instructions, you can effectively utilize the LVDAN model for enhanced object detection in axial CT images, paving the way for improved diagnostic capabilities in medical imaging.
