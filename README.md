# SOEP-DETR: Frequency-Spatial Enhanced RT-DETR for Real-Time Small Object Detection

This repository provides a **minimal, training-ready** implementation of **SOEP-DETR**, a frequency-spatial enhancement framework built upon **RT-DETR** for real-time small object detection. The codebase is intentionally pruned to retain only the components required for:

- RT-DETR training/evaluation
- ResNet18 backbone baseline
- SOEP insertion at **P3/P4** (before the neck/head)
- Datasets: **VisDrone2019-DET** and **TinyPerson** (COCO format)

To facilitate reproducible research, the repository is organized as a self-contained training project rather than a general-purpose vision toolkit.

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Method Overview](#2-method-overview)
- [3. Implementation in This Repo](#3-implementation-in-this-repo)
- [4. Installation](#4-installation)
- [5. Data Preparation](#5-data-preparation)
- [6. Training](#6-training)
- [7. Evaluation](#7-evaluation)
- [8. Analysis Utilities](#8-analysis-utilities)
- [9. Reproducibility Notes](#9-reproducibility-notes)
- [10. Acknowledgements](#10-acknowledgements)
- [11. Citation](#11-citation)

## 1. Abstract

Small object detection in aerial/remote-sensing imagery is challenging due to the limited spatial support of targets and the strong interference from complex backgrounds. SOEP-DETR enhances RT-DETR by introducing a lightweight **Spatial-Omni-Enhanced Perception (SOEP)** module that couples frequency-domain and spatial-domain context modeling. SOEP integrates (i) a frequency-guided modulation mechanism to emphasize informative spectral components and (ii) a large-kernel omni-directional aggregation mechanism to improve context capture for arbitrarily oriented small objects. This repository implements SOEP-DETR with a ResNet18 backbone and provides training/evaluation scripts for VisDrone2019-DET and TinyPerson.

## 2. Method Overview

### 2.1 SOEP Module

SOEP is composed of two complementary components:

- **Frequency-Guided Module (FGM)**: applies frequency-domain modulation to suppress background noise and enhance salient object structures.
- **OmniKernel**: uses directionally decomposed large depthwise convolutions to capture multi-directional context efficiently.

### 2.2 SOEP-DETR Integration

In this repository, SOEP is inserted into the RT-DETR backbone feature hierarchy at:

- **P3** (stride 8) output features
- **P4** (stride 16) output features

This design targets small-object dominant feature scales while keeping the overall architecture lightweight.

## 3. Implementation in This Repo

This repository is intentionally minimal and **only supports training/evaluation** for the SOEP-DETR setting above. The following modules are removed on purpose to avoid irrelevant dependencies:

- SAM / FastSAM / NAS
- Tracking / export / hub integration / hyperparameter tuning
- Additional backbones and experimental modules unrelated to RT-DETR(ResNet18)+SOEP

### 3.1 Project Defaults

- Default model YAML:
  - `ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml`
- Dataset YAMLs:
  - VisDrone2019-DET: `dataset/data.yaml`
  - TinyPerson: `dataset/tinyperson.yaml`
- Default outputs:
  - Training runs: `runs/train/exp*`
  - Evaluation runs: `runs/val/exp*`

### 3.2 Key Files

- SOEP standalone module:
  - `SOEP.py`
- SOEP registered for YAML construction:
  - `ultralytics/nn/extra_modules/soep.py`
- RT-DETR ResNet18 + SOEP model definition:
  - `ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml`
- Entry scripts:
  - Training: `train.py`
  - Evaluation: `val.py`

### 3.3 SOEP Insertion in YAML (Excerpt)

The P3/P4 insertion is implemented by adding `SOEP` blocks after the corresponding backbone stages:

```yaml
- [-1, 1, Blocks, [128, BasicBlock, 2, 3, 'relu']]  # P3
- [-1, 1, SOEP, []]                                # P3 enhanced
- [-1, 1, Blocks, [256, BasicBlock, 2, 4, 'relu']]  # P4
- [-1, 1, SOEP, []]                                # P4 enhanced
```

## 4. Installation

```bash
cd /home/pc/gxy/SOEP
pip install -r requirements.txt
pip install -e .
```

## 5. Data Preparation

This repository assumes that datasets are available in **COCO format** (images + COCO JSON annotations).

### 5.1 Recommended Directory Structure

```text
datasets/
├── VisDrone2019-DET/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── TinyPerson/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
```

### 5.2 Dataset YAMLs Used by This Repo

VisDrone2019-DET:

```yaml
path: VisDrone2019-DET
train: images/train
val: images/val
test: images/test
```

TinyPerson:

```yaml
path: TinyPerson
train: images/train
val: images/val
test: images/test
```

Note: the training code resolves a relative `path:` against the global datasets directory configured by Ultralytics (typically `datasets/`). If your datasets are located elsewhere, set `path:` to an absolute path.

### 5.3 Class Names and Category Order

The dataset `names` must be consistent with the category definitions in your COCO JSON.

VisDrone2019-DET (10 categories) assumes the following order in `dataset/data.yaml`:

```yaml
names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
```

TinyPerson (single category):

```yaml
names: [person]
```

## 6. Training

The training entry script uses the SOEP-DETR model YAML by default:

```bash
python train.py
```

To train on TinyPerson, set `data=` in `train.py` to:

- `dataset/tinyperson.yaml`

### 6.1 Sanity Checks

1. Verify that the model can be constructed:

```bash
python -c "from ultralytics import RTDETR; RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml'); print('ok')"
```

2. Verify dataset YAML resolves to a valid path:

```bash
python -c \"from ultralytics.data.utils import check_det_dataset; print(check_det_dataset('dataset/data.yaml')['path'])\"
```

## 7. Evaluation

```bash
python val.py
```

The evaluation script uses the same dataset YAML mechanism as training. Ensure that the dataset split (`val`) exists and the COCO JSON is correctly formatted.

## 8. Analysis Utilities

This repository includes optional analysis scripts for academic visualization:

- Frequency analysis:
  - `frequency.py`
- Effective receptive field visualization:
  - `ERF.py`

These scripts are independent of the training pipeline and can be used for qualitative analysis and figure generation.

## 9. Reproducibility Notes

### 9.1 Dataset Path Errors

If you encounter errors such as:

```
Dataset 'dataset/data.yaml' images not found
```

resolve them by:

- placing datasets under the configured datasets root directory (commonly `datasets/`), or
- setting the dataset `path:` field to an absolute path

### 9.2 Scope of This Repository

This is a research-oriented, minimal training repository. Features unrelated to SOEP-DETR training (e.g., export/track/HUB) were removed intentionally.

## 10. Acknowledgements

This codebase is built upon open-source implementations of RT-DETR and the Ultralytics training infrastructure. We thank the authors and maintainers for making their work publicly available.

## 11. Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{guo2026soepdetr,
  title={SOEP-DETR: Frequency-Spatial Enhanced RT-DETR for Real-Time Small Object Detection},
  author={Guo, Xiuyi and Liu, Hongbin and Dong, Peng and Zhao, Yongze and Zhou, Yitong and Li, Jilong and Wang, Baoxu and Peng, Wei and Li, Chengdong},
  year={2026}
}
```
