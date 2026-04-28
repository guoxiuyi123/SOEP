# SOEP-DETR: Synergistic Frequency-Spatial Modulation for Small Object Detection in Remote Sensing Imagery

Official implementation of the paper **"SOEP-DETR: Synergistic Frequency-Spatial Modulation for Small Object Detection in Remote Sensing Imagery"**.

## 📖 Introduction
Detecting small objects in high-resolution remote sensing (RS) imagery remains a fundamental challenge because these targets occupy minimal pixels and lack clear textures. Consequently, they are easily overwhelmed by complex topographical background noise. We propose **SOEP-DETR**, an RT-DETR-based detector built around the lightweight **Spatial-Omni-Enhanced Perception (SOEP)** module.

The SOEP module integrates two complementary mechanisms:
* **Frequency-Guided Module (FGM)**: Recalibrates spectral amplitudes to suppress terrain-induced interference and enhance salient object structures.
* **OmniKernel Module**: Employs directionally decomposed large convolutions to capture multi-directional context for arbitrarily oriented aerial targets.

## 📂 Repository Structure
* `SOEP.py`: Contains the core PyTorch implementations of the **SOEP**, **FGM**, and **OmniKernel** modules.
* `frequency.py`: A utility script for frequency-domain analysis, including High-Frequency Energy Ratio (HFER) calculation and spectral visualization.
* `ERF.py`: Provides tools to calculate and visualize the Effective Receptive Field (ERF) to verify the impact of large kernels.

## 📊 Datasets Preparation
Download the **TinyPerson** and **VisDrone2019-DET** datasets from their official GitHub repositories:
* **[TinyPerson Dataset](https://github.com/w-sugar/TinyBenchmark)** (Official implementation for WACV 2020 "Scale Match for Tiny Person Detection")
* **[VisDrone2019-DET Dataset](https://github.com/VisDrone/VisDrone-Dataset)** (Official AISKYEYE team repository)

We recommend organizing them in the standard COCO format:
```text
datasets/
├── TinyPerson/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET/
    ├── images/
    └── annotations/
```

## 🚀 Usage Guidelines
The SOEP module is designed as a plug-and-play component for integration into existing CNN or Transformer-based backbones.

```python
import torch
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
If you find this work useful in your research, please cite our manuscript:

```bibtex
@article{guo2026soep,
  title={SOEP-DETR: Synergistic Frequency-Spatial Modulation for Small Object Detection in Remote Sensing Imagery},
  author={Guo, Xiuyi and Liu, Hongbin and Dong, Peng and Zhao, Yongze and Zhou, Yitong and Li, Jilong and Wang, Baoxu and Peng, Wei and Li, Chengdong},
  journal={Machine Vision and Applications},
  year={2026}
}
```
