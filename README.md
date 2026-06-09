# SOEP-DETR

SOEP-DETR is a frequency-spatial enhanced RT-DETR framework for real-time small object detection in UAV and surveillance imagery.

The code is based on the Ultralytics RT-DETR framework and introduces the proposed Spatial-Omni-Enhanced Perception (SOEP) module into RT-DETR-R18 to improve small object detection under low-resolution, low-contrast, and cluttered-background conditions.



## 1. Project Description

Small objects in UAV images usually occupy only a few pixels and are easily affected by background clutter, motion blur, occlusion, scale variation, and low contrast. SOEP-DETR is designed to strengthen small object representation while maintaining real-time inference performance.

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

The datasets are not included in this repository. Please download them from the official sources and prepare them locally.

### 2.1 TinyPerson

TinyPerson is a tiny person detection dataset for long-distance surveillance scenes.

Official dataset source:

```text
https://github.com/w-sugar/TinyBenchmark
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

If the original annotations contain multiple person-related categories, please merge them into the single `person` category before training.

### 2.2 VisDrone2019-DET

VisDrone2019-DET is a UAV-based object detection dataset containing crowded aerial scenes, scale variation, occlusion, and complex backgrounds.

Official dataset source:

```text
https://github.com/VisDrone/VisDrone-Dataset
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

The expected dataset structure is:

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

Please modify the `path` field in the corresponding YAML file according to your local dataset location.

Example:

```yaml
path: /path/to/VisDrone2019-DET
train: images/train
val: images/val
test: images/test
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

---

## 4. Requirements

```text
Recommended environment:
  Operating system: Ubuntu 20.04 / Windows 10 or later
  Python: 3.8 or later
  CUDA: 11.3 or later
  GPU: NVIDIA GPU is recommended for training and inference

Tested dependency versions:
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

Optional package:
  pycocotools == 2.0.6
```

Please install the required packages according to your local Python, CUDA, and PyTorch versions. Different CUDA or PyTorch versions may also work, but the versions listed above are recommended for reproducing the experiments.

---

## 5. Usage Instructions

### 5.1 Clone the Repository

```bash
git clone https://github.com/guoxiuyi123/SOEP.git
cd SOEP
```

### 5.2 Prepare the Dataset

Download the dataset from the official source and organize it in YOLO detection format.

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

### 5.3 Train

Run:

```bash
python train.py
```

The default model configuration is:

```text
ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml
```

Training results will be saved in:

```text
runs/train/exp/
```

The best model is usually saved as:

```text
runs/train/exp/weights/best.pt
```

### 5.4 Validate

Run:

```bash
python val.py
```

The validation script reports detection accuracy, model parameters, GFLOPs, inference time, and FPS.

Validation results will be saved in:

```text
runs/val/exp/
```

### 5.5 Inference

Example:

```python
from ultralytics import RTDETR

model = RTDETR("runs/train/exp/weights/best.pt")
model.predict(source="path/to/images", imgsz=640, conf=0.25, save=True)
```

---

## 6. Methodology

SOEP-DETR is built on RT-DETR-R18. The proposed SOEP module is inserted into the feature extraction pipeline to enhance small object representation.

### 6.1 Frequency-Guided Module

FGM uses frequency-domain feature modulation to enhance weak structural cues such as edges, contours, and local intensity changes. This helps tiny objects remain distinguishable from cluttered backgrounds.

### 6.2 OmniKernel Module

OmniKernel uses large-kernel spatial aggregation to enlarge the receptive field and improve contextual perception. It is designed to provide surrounding context for small object localization while keeping the computational cost controlled.

### 6.3 Integration into RT-DETR

The SOEP module is integrated into the RT-DETR-R18 model configuration. The provided YAML file inserts SOEP into the P3 and P4 feature stages:

```text
ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml
```

P3 and P4 are selected because they preserve more spatial details and are important for small object detection.

---

## 7. Citation

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

## 8. License and Contribution

This repository is released for academic research and reproducibility.

Please follow the licenses of the original third-party codebases and datasets. If an explicit license file is added later, the repository license should be updated accordingly.

Contributions are welcome. You may submit issues or pull requests for bug fixes, documentation improvements, dataset preparation scripts, or additional experimental support.
