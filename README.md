# SOEP-DETR

SOEP-DETR is a frequency-spatial enhanced RT-DETR framework for real-time small object detection in UAV and surveillance imagery.

The code is based on the Ultralytics RT-DETR framework and introduces the proposed Spatial-Omni-Enhanced Perception (SOEP) module into RT-DETR-R18 to improve small object detection under low-resolution, low-contrast, and cluttered-background conditions.

## 1. Project Description

Small objects in UAV and surveillance images usually occupy only a few pixels and are easily affected by background clutter, motion blur, occlusion, scale variation, and low contrast. SOEP-DETR is designed to strengthen small object representation while maintaining real-time inference performance.

The main idea is to combine frequency-domain enhancement and spatial large-kernel context aggregation.

Main components:

* **FGM**: Frequency-Guided Module for enhancing weak structural and high-frequency responses.
* **OmniKernel**: Large-kernel spatial aggregation module for improving contextual perception.
* **SOEP**: Spatial-Omni-Enhanced Perception module integrated into RT-DETR-R18.

---

## 2. Dataset Information

This repository supports experiments on two public small object detection datasets:

* **TinyPerson**
* **VisDrone2019-DET**

The datasets are not included in this repository. Please download them from their official sources and prepare them locally.

### 2.1 TinyPerson

TinyPerson is a tiny person detection dataset for long-distance surveillance scenes.

Official dataset source:

```text
https://github.com/ucas-vg/TinyBenchmark
```

Dataset configuration file in this repository:

```text
dataset/tinyperson.yaml
```

Class setting used in this repository:

```yaml
nc: 1
names: [person]
```

If the original annotations contain multiple person-related categories, such as `sea-person` and `earth-person`, please merge them into a single `person` category before training.

### 2.2 VisDrone2019-DET

VisDrone2019-DET is a UAV-based object detection dataset containing crowded aerial scenes, scale variation, occlusion, and complex backgrounds.

Official dataset source:

```text
https://github.com/VisDrone/VisDrone-Dataset
```

Dataset usage and license information:

```text
https://aiskyeye.com/data-protection/
```

Dataset configuration file in this repository:

```text
dataset/data.yaml
```

Class setting used in this repository:

```yaml
nc: 10
names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
```

### 2.3 Expected Dataset Format

The dataset should be organized in the Ultralytics detection format.

Expected structure:

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label file should use the normalized YOLO detection format:

```text
class_id x_center y_center width height
```

where `x_center`, `y_center`, `width`, and `height` are normalized to `[0, 1]`.

Please modify the `path` field in the corresponding YAML file according to your local dataset location.

Example for VisDrone2019-DET:

```yaml
path: /path/to/VisDrone2019-DET
train: images/train
val: images/val
test: images/test
nc: 10
names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
```

Example for TinyPerson:

```yaml
path: /path/to/TinyPerson
train: images/train
val: images/val
test: images/test
nc: 1
names: [person]
```

---

## 3. Code Information

Main files and folders:

```text
SOEP/
├── dataset/
│   ├── data.yaml
│   └── tinyperson.yaml
├── ultralytics/
│   ├── cfg/models/rt-detr/
│   │   └── rtdetr-r18-soep-p3p4.yaml
│   └── nn/extra_modules/
│       └── soep.py
├── train.py
├── val.py
├── ERF.py
├── frequency.py
├── requirements.txt
├── setup.py
└── README.md
```

File descriptions:

* `ultralytics/nn/extra_modules/soep.py`: implementation of SOEP-related modules.
* `ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml`: SOEP-DETR model configuration.
* `dataset/data.yaml`: VisDrone2019-DET dataset configuration.
* `dataset/tinyperson.yaml`: TinyPerson dataset configuration.
* `train.py`: training script.
* `val.py`: validation and metric reporting script.
* `ERF.py`: effective receptive field visualization script.
* `frequency.py`: frequency-domain visualization script.
* `requirements.txt`: Python package requirements.
* `setup.py`: package setup file inherited from the Ultralytics codebase.

---

## 4. Requirements

Recommended environment:

```text
Operating system: Ubuntu 20.04 / Windows 10 or later
Python: 3.8 or later
CUDA: 11.3 or later
GPU: NVIDIA GPU is recommended for training and inference
```

Tested dependency versions:

```text
torch == 1.13.1
torchvision == 0.14.1
numpy == 1.23.5
opencv-python == 4.7.0.72
matplotlib == 3.7.1
pyyaml == 6.0
tqdm == 4.65.0
pandas == 1.5.3
scipy == 1.10.1
thop == 0.1.1.post2209072238
```

Optional package:

```text
pycocotools == 2.0.6
```

Different CUDA or PyTorch versions may also work, but the versions listed above are recommended for reproducing the experiments.

---

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/guoxiuyi123/SOEP.git
cd SOEP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not installed correctly, please install the PyTorch version that matches your CUDA environment first, and then install the remaining dependencies.

---

## 6. Usage Instructions

### 6.1 Prepare the Dataset

Download the datasets from their official sources and organize them in the expected detection format.

Then edit the corresponding YAML file.

For VisDrone2019-DET:

```text
dataset/data.yaml
```

For TinyPerson:

```text
dataset/tinyperson.yaml
```

Example:

```yaml
path: /path/to/VisDrone2019-DET
train: images/train
val: images/val
test: images/test
nc: 10
names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
```

### 6.2 Train

Run:

```bash
python train.py
```

The default model configuration is:

```text
ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml
```

The default dataset configuration in `train.py` is:

```text
dataset/data.yaml
```

Training outputs will be saved in:

```text
runs/train/exp/
```

The best model is usually saved as:

```text
runs/train/exp/weights/best.pt
```

To train on TinyPerson, modify the `data` argument in `train.py`:

```python
model.train(data='dataset/tinyperson.yaml', ...)
```

To reproduce a specific experiment, please make sure that the batch size, epoch number, input size, dataset split, and random seed are consistent with the settings reported in the manuscript.

### 6.3 Validate

Run:

```bash
python val.py
```

The validation script reports detection accuracy, model parameters, GFLOPs, inference time, and FPS.

Validation results will be saved in:

```text
runs/val/exp/
```

If your checkpoint path is different, modify the checkpoint path in `val.py`.

### 6.4 Inference

Example:

```python
from ultralytics import RTDETR

model = RTDETR("runs/train/exp/weights/best.pt")
model.predict(source="path/to/images", imgsz=640, conf=0.25, save=True)
```

---

## 7. Methodology

SOEP-DETR is built on RT-DETR-R18. The proposed SOEP module is inserted into the feature extraction pipeline to enhance small object representation.

### 7.1 Frequency-Guided Module

FGM applies frequency-domain feature modulation to enhance weak structural cues such as edges, contours, and local intensity changes. This helps tiny objects remain distinguishable from cluttered backgrounds.

### 7.2 OmniKernel Module

OmniKernel uses large-kernel spatial aggregation to enlarge the receptive field and improve contextual perception. It provides surrounding context for small object localization while keeping the computational cost controlled.

### 7.3 Integration into RT-DETR

The SOEP module is integrated into RT-DETR-R18 through the following model configuration:

```text
ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml
```

The provided configuration inserts SOEP into the P3 and P4 feature stages. P3 and P4 are selected because they preserve more spatial details and are important for small object detection.

---

## 8. Outputs

Typical training outputs:

```text
runs/train/exp/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
└── results.png
```

Typical validation outputs:

```text
runs/val/exp/
```

Generated experiment outputs, model weights, and dataset files are not required to be committed to the repository.

---

## 9. Citation

If you use this code in your research, please cite:

```bibtex
@misc{guo2026soepdetr,
  title  = {SOEP-DETR: Frequency-Spatial Enhanced RT-DETR for Real-Time Small Object Detection},
  author = {Guo, Xiuyi and Liu, Hongbin and Dong, Peng and Zhou, Yitong and Li, Jilong and Wang, Baoxu},
  year   = {2026},
  note   = {Code available at: https://github.com/guoxiuyi123/SOEP}
}
```

Please also cite the original RT-DETR, TinyPerson, and VisDrone papers if their models or datasets are used.

---

## 10. License and Contribution

This repository is based on the Ultralytics codebase and follows the license terms of the original framework. The current setup file specifies the AGPL-3.0 license.

Please also follow the license and usage terms of the third-party datasets used in this study.

Contributions are welcome through issues and pull requests. Contributions may include bug fixes, documentation improvements, dataset preparation scripts, and additional experimental support.

Please do not upload original datasets, large trained weights, or generated experiment folders directly to this repository.
