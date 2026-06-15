# Local-GS: Tile-Local Coherent Acceleration for 3D Gaussian Splatting

<p align="center">
  <a href="https://arxiv.org/abs/2606.xxxxx"><img src="https://img.shields.io/badge/arXiv-2606.xxxxx-B31B1B.svg?style=flat-square" alt="Paper"></a>
  <a href="https://github.com/yourname/Local-GS"><img src="https://img.shields.io/badge/Code-Local--GS-blue.svg?style=flat-square&logo=github" alt="Code"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-green.svg?style=flat-square&logo=nvidia" alt="CUDA"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License"></a>
</p>

This is the official PyTorch/CUDA implementation of the paper **"Local-GS: Tile-Local Coherent Acceleration for 3D Gaussian Splatting"** (also designated as **TiCoGS**). 

Local-GS introduces a high-performance, plug-and-play CUDA rasterization kernel built seamlessly on top of the standard [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) infrastructure, optimizing the rendering pipeline directly at the SIMT execution boundaries.

---

## 📌 Abstract

> **Local-GS: Accelerating 3D Gaussian Splatting via Tile-Local Warp Coherence**
> *Yang Luo, Yan Gong, Jie Zhao, Xiaoying Honor Sun, Yanhe Zhu, and Yongsheng Gao*
> *State Key Laboratory of Robotics and Systems, Harbin Institute of Technology (HIT)*

3D Gaussian Splatting (3DGS) has emerged as a revolutionary paradigm for real-time novel view synthesis. However, the standard rasterization pipeline suffers from microscopic hardware inefficiencies, such as severe warp divergence caused by long-tailed ellipsoidal Gaussians and high register pressure in the inner blending loop. In this paper, we break away from traditional scene-space geometric pruning and pioneer an execution-driven micro-coherence acceleration framework called **Local-GS**.

* **Parameter Hoisting:** We extract pixel-independent Gaussian parameters to the Tile level and cache them in shared memory, compressing the inner loop into a native 6D vector dot product executed via a high-throughput native FMA chain.
* **Intra-block Warp-Level Culling:** We implement a warp-level cooperative culling strategy to eliminate inactive threads before prefetching, alongside a **Register-level Double Buffering** mechanism inside the batch-size processing loop, establishing a completely branch-free arithmetic stream.

**Key Result:** Our single-pass, plug-and-play kernel maintains exactly zero visual quality loss while introducing **zero extra DRAM memory overhead**. Local-GS achieves consistent **1.4x to 1.6x** speedups on modern NVIDIA Ada Lovelace architectures (RTX 4080/4090) and up to a **7.76x speedup** on challenging datasets like *Deep Blending*.

---

## 🔧 Setup & Installation

The hardware-level optimizations of Local-GS are self-contained and implemented entirely within the differential renderer submodule (`diff-gaussian-rasterization`).

### Prerequisites (硬件与软件要求)

Our pipeline shares the exact identical compute and compiler prerequisites as the original 3D-GS framework. Before proceeding, please ensure your system satisfies the following baseline specifications:
* **OS:** Linux (Ubuntu 20.04/22.04 fully tested) or Windows.
* **CUDA Ready GPU:** NVIDIA Compute Capability $\ge$ 7.0 (NVIDIA RTX 30-series/40-series or A100/H100 highly recommended).
* **CUDA Toolkit:** Local CUDA compiler (`nvcc`) version **11.8** or **12.1** (must match your runtime PyTorch CUDA version).
* **C++ Compiler:** `gcc` / `g++` $\ge$ 9.0 (required for hosting and linking parallel CUDA extensions).

---

### Step-by-Step Installation (逐步安装指南)

#### 1. Clone the Repository (Git 克隆主仓库与内核)
To fetch the core Python logic along with our optimized warp-coherent rasterizer submodule, clone the repository recursively:
```bash
# 递归克隆主仓库以及底层的 C++/CUDA 微分光栅化算子
git clone --recursive [https://github.com/tilaba/Local-GS.git](https://github.com/tilaba/Local-GS.git)
cd Local-GS
git submodule update --init --recursive
💡 Note: If you accidentally cloned the project without the submodules, you can retroactively initialize and sync them by running:

Bash
git submodule update --init --recursive
2. Environment Configuration (配置 Anaconda 环境)
We provide a standard distribution profile identical to graphdeco-inria/gaussian-splatting to manage dependencies automatically:

Bash

conda env create --file environment.yml
conda activate gaussian_splatting
3. Kernel Compilation & Submodule Setup (编译并构建算子)
The accelerated execution-driven operators must be compiled and bound to your target Python interpreter runtime. You can leverage our automated, one-click shell script which safely purges stale build artifacts, compiles hardware-level extensions, and registers submodules:

Bash
# 赋予执行权限并运行一键清理与重构脚本
chmod +x ./rebuild.sh
./rebuild.sh