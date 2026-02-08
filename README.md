# An Efficient Network for Small Object Detection via Frequency Domain Modulation and Directionally Decomposed Large Kernels

**[Preprint]** | **[Paper]** (Link will be updated upon acceptance)

本仓库为论文 **"An Efficient Network for Small Object Detection via Frequency Domain Modulation and Directionally Decomposed Large Kernels"** 的官方代码实现（部分预览）。

目前仅上传了论文核心模块（SOEP）及可视化分析工具的关键代码。**完整的训练框架、预训练权重及所有脚本将在论文正式录用（Accepted）后第一时间在此仓库开源。**

## 📂 核心代码说明 (Code Description)

本仓库当前包含以下核心文件，对应论文中提出的 **SOEP (Spatial-Omni-Enhanced Perception)** 模块及其验证实验：

### 1. `SOEP.py` - 核心模型架构
这是本论文的核心算法实现文件，包含了 **Spatial-Omni-Enhanced Perception (SOEP)** 模块的完整定义。
* **主要类 (Classes):**
    * `FGM` (Frequency-Guided Module): 频域引导模块。利用 FFT/IFFT 进行频谱重校准，并通过跨域门控机制增强高频纹理特征，抑制背景噪声。
    * `OmniKernel`: 全向大核模块。通过分解的大卷积核（$1\times K$, $K\times 1$, $K\times K$）在保持低计算成本的同时扩展有效感受野（ERF）。
    * `CSPOmniKernel`: 集成到 CSP 结构中的封装实现，方便接入 DETR 等架构。

### 2. `ERF.py` - 有效感受野可视化 (ERF Visualization)
用于生成论文中 **有效感受野 (Effective Receptive Field)** 热力图的工具脚本。
* **功能:** 计算并可视化模型在特定层对于输入图像的梯度响应分布。
* **对应论文:** 对应论文实验部分关于感受野的分析（如文中 Fig. 4），证明 SOEP 模块能有效扩展模型对小目标的关注范围。
* **使用方法:**
    ```python
    # 需要在代码中指定权重路径和目标层
    python ERF.py
    ```

### 3. `frequency.py` - 频域分析与可视化 (Frequency Domain Analysis)
用于进行频域分析的工具脚本，支持生成频谱图及计算 **HFER (High-Frequency Energy Ratio)** 指标。
* **功能:**
    * 对特征图进行 FFT 变换并可视化频谱。
    * 计算高频能量占比 (HFER)，量化评估 FGM 模块对高频细节的增强效果。
    * 生成对比图（原图 vs 特征图 vs 频谱图）。
* **对应论文:** 对应论文方法论部分关于频域重校准的验证（如文中 Fig. 3）。
* **运行示例:**
    ```bash
    python frequency.py --image path/to/img.jpg --npy_original path/to/feat_a.npy --npy_fgm path/to/feat_b.npy
    ```

---

## 📅 开源计划 (Release Plan)

我们承诺在论文被录用后公开以下内容：
- [ ] 完整的 SOEP 模型训练与推理代码 (Based on RT-DETR/YOLO)。
- [ ] 在 TinyPerson 和 VisDrone2019-DET 数据集上的复现脚本。
- [ ] 预训练模型权重 (Pre-trained Weights)。
- [ ] 完整的配置文件与环境依赖说明。

## 🔗 引用 (Citation)

如果您觉得本工作对您的研究有帮助，请在论文录用后关注引用更新。

```bibtex
@article{SOEP2026,
  title={An Efficient Network for Small Object Detection via Frequency Domain Modulation and Directionally Decomposed Large Kernels},
  author={Guo, Xiuyi and Liu, Hongbin and Dong, Peng and Zhao, Yongze and Zhou, Yitong and Li, Jilong and Wang, Baoxu and Peng, Wei and Li, Chengdong},
  journal={Neurocomputing (Under Review)},
  year={2026}
}
