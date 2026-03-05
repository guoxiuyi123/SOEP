# Enhancing Small Object Perception through Synergistic Frequency-Spatial Modulation (SOEP)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18872450.svg)](https://doi.org/10.5281/zenodo.18872450)

🚨 **Important Note:** This repository contains the official core implementations for the manuscript *"Enhancing Small Object Perception through Synergistic Frequency-Spatial Modulation"*, which is currently submitted to **The Visual Computer**. If you find this code or our work useful in your research, we strongly encourage and kindly request that you cite our relevant manuscript (see the Citation section below).

---

## 📖 Introduction
This repository provides the core, plug-and-play PyTorch implementations of the **Spatial-Omni-Enhanced Perception (SOEP)** module. As detailed in our manuscript, SOEP is designed to augment existing object detection frameworks (such as RT-DETR) to tackle the fundamental challenges of detecting small objects in complex scenes, mitigating feature degradation and background noise via frequency-domain modulation and large-kernel spatial aggregation.

## 📂 File Structure & Key Algorithms
The repository contains the following core files, directly corresponding to the methodologies and evaluations proposed in our paper:

- `SOEP.py`: Implements the complete **Spatial-Omni-Enhanced Perception (SOEP)** module. This file contains the core PyTorch network structures for both the **Frequency-Guided Module (FGM)** (handling FFT/IFFT spectral recalibration) and the **OmniKernel Module** (handling directionally decomposed large convolutions).
- `frequency.py`: A utility script designed for frequency domain feature comparison. It is used to analyze the spectral energy distribution, calculate the High-Frequency Energy Ratio (HFER), and generate the frequency-domain visualizations discussed in our manuscript.
- `ERF.py`: Provides the analytical script to calculate and visualize the Effective Receptive Field (ERF), corroborating the quantitative ERF_20% and ERF_50% analysis presented in our experiments.

## ⚙️ Dependencies
To use these modules, ensure your environment meets the following basic requirements:
- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision

## 📊 Datasets Preparation
If you wish to replicate our experiments, please download the **VisDrone2019-DET** and **TinyPerson** datasets from their official sources. We recommend organizing them in the standard COCO format and placing them in a `datasets/` directory at the root of your detection framework (e.g., adjacent to your RT-DETR project folder):

```text
datasets/
├── TinyPerson/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── annotations/
│       ├── instances_train.json
│       └── instances_val.json
└── VisDrone2019-DET/
    ├── images/
    │   ├── train/
    │   └── val/
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
```

## 🚀 Usage Guidelines (Integration)
Because SOEP is designed as a lightweight, plug-and-play component, you can easily integrate it into any existing CNN or Transformer-based backbone (e.g., ResNet, RT-DETR). 

Here is a minimal example of how to import and use the SOEP module in your own PyTorch project:

```python
import torch
# Import the core SOEP module from our provided script
from SOEP import SOEP 

# Example: Initialize the SOEP module 
# (Adjust the channel dimensions based on your specific backbone stage)
in_channels = 256
soep_module = SOEP(in_channels=in_channels)

# Create a dummy input tensor representing a feature map [Batch_size, Channels, Height, Width]
dummy_input = torch.randn(2, in_channels, 64, 64)

# Pass the feature map through the SOEP module for frequency-spatial enhancement
enhanced_features = soep_module(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {enhanced_features.shape}")



