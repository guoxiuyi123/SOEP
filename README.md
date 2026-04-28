# [cite_start]SOEP-DETR: Synergistic Frequency-Spatial Modulation for Small Object Detection in Remote Sensing Imagery [cite: 1]

[cite_start]Official implementation of the paper **"SOEP-DETR: Synergistic Frequency-Spatial Modulation for Small Object Detection in Remote Sensing Imagery"**[cite: 1].

## 📖 Introduction
[cite_start]Detecting small objects in high-resolution remote sensing (RS) imagery remains a fundamental challenge because these targets occupy minimal pixels and lack clear textures[cite: 1]. [cite_start]Consequently, they are easily overwhelmed by complex topographical background noise[cite: 1]. [cite_start]We propose **SOEP-DETR**, an RT-DETR-based detector built around the lightweight **Spatial-Omni-Enhanced Perception (SOEP)** module[cite: 1].

The SOEP module integrates two complementary mechanisms:
* [cite_start]**Frequency-Guided Module (FGM)**: Recalibrates spectral amplitudes to suppress terrain-induced interference and enhance salient object structures[cite: 1, 5].
* [cite_start]**OmniKernel Module**: Employs directionally decomposed large convolutions to capture multi-directional context for arbitrarily oriented aerial targets[cite: 1, 5].

## 📂 Repository Structure
* [cite_start]`SOEP.py`: Contains the core PyTorch implementations of the **SOEP**, **FGM**, and **OmniKernel** modules[cite: 5].
* [cite_start]`frequency.py`: A utility script for frequency-domain analysis, including High-Frequency Energy Ratio (HFER) calculation and spectral visualization[cite: 6].
* [cite_start]`ERF.py`: Provides tools to calculate and visualize the Effective Receptive Field (ERF) to verify the impact of large kernels[cite: 3].

## [cite_start]📊 Experimental Performance [cite: 1]
[cite_start]SOEP-DETR demonstrates superior performance on two challenging remote sensing benchmarks[cite: 1]:

### 1. TinyPerson (Long-distance Surveillance)
* [cite_start]**Significant Gains**: Achieves relative improvements of **17.86%** in AP and **21.43%** in $AP_{s}$ over the RT-DETR-R18 baseline[cite: 1].
* [cite_start]**Optimal Kernel**: Reaches peak performance using a kernel size of **K=41**[cite: 1].

### 2. VisDrone2019-DET (UAV Scenarios)
* [cite_start]**Detection Accuracy**: Attains an AP of **0.221** and an $AP_{s}$ of **0.202**[cite: 1].
* [cite_start]**Inference Speed**: Maintains a real-time speed of **113.07 FPS**, suitable for resource-constrained UAV deployment[cite: 1].

## 🚀 Usage Guidelines
[cite_start]The SOEP module is designed as a plug-and-play component for integration into existing CNN or Transformer-based backbones[cite: 4].

```python
import torch
# Import the SOEP module
from SOEP import SOEP 

# Initialize (adjust channels based on your backbone stage)
in_channels = 256
soep_module = SOEP(dim=in_channels)

# Input feature map: [Batch_size, Channels, Height, Width]
dummy_input = torch.randn(2, in_channels, 64, 64)

# Apply frequency-spatial enhancement
enhanced_features = soep_module(dummy_input)
```

## 📝 Citation
[cite_start]If you find this work useful in your research, please cite our manuscript[cite: 4]:

```bibtex
@article{guo2026soep,
  title={SOEP-DETR: Synergistic Frequency-Spatial Modulation for Small Object Detection in Remote Sensing Imagery},
  author={Guo, Xiuyi and Liu, Hongbin and Dong, Peng and Zhao, Yongze and Zhou, Yitong and Li, Jilong and Wang, Baoxu and Peng, Wei and Li, Chengdong},
  journal={Machine Vision and Applications},
  year={2026}
}
```
