# CoDiVSR: Rethinking What to Condition and What to Disentangle in Diffusion-based Video Super-Resolution

[![visitors](https://visitorbadge.io/api/visitors?path=yyang181%2FCoDiVSR&label=visitors&countColor=%23263759)](https://visitorbadge.io/status?path=yyang181%2FCoDiVSR)

[Paper](https://github.com/yyang181/CoDiVSR/blob/main/assets/paper.pdf)

## News
- **March 22, 2026**: We have released our code and checkpoints.

## To-Do List
- [ ] Release training code
- [x] Release testing code
- [x] Release pre-trained models
- [ ] Release demo

## Quickstart

### Environment Setup
```bash
# 1. Clone the repository and create a conda environment: 
git clone https://github.com/yyang181/CoDiVSR.git
cd CoDiVSR
conda create -n codivsr python=3.10 -y
conda activate codivsr

# 2. Install PyTorch. We recommend PyTorch 2.5.1 with CUDA 12.4:
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install other requirements:
pip install -r requirements.txt
```

### Download Checkpoints
```bash 
pip install modelscope

# Pre-trained CogVideoX1.5-5B-I2V
modelscope download ZhipuAI/CogVideoX1.5-5B-I2V --local_dir ./checkpoints/CogVideoX1.5-5B-I2V

# CoDiVSR checkpoints
modelscope download yyang181/CoDiVSR --local_dir ./checkpoints/codivsr/transformer
```

### Inference 
The inference code has been tested on:

- Ubuntu 20.04
- Python 3.10
- PyTorch 2.5.1
- 1 NVIDIA H20 GPU with CUDA 12.4 (Requires ~40GB VRAM for a video with 1080x1920x100 resolution; inference takes approximately 3 minutes and 40 seconds).
  
We use [PLLaVa-13B](https://github.com/magic-research/PLLaVA/) to extract captions.

```bash
# NTIRE 2026 SUGC-VR
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --input_dir data/input \
    --input_json data/csv/input_text.csv \
    --output_path data/output/codivsr \
    --model_path checkpoints/codivsr \
    --is_vae_st \
    --save_format yuv420p \
    --upscale 1 \
    --load_skipconv1d \
    --use_low_pass_guidance \
    --enable_midresidual 
```

## Acknowledgements

This project builds upon several excellent open-source projects:

* [CogVideo](https://github.com/THUDM/CogVideo) - A large-scale video generation model developed by Tsinghua University that provides the foundational architecture for this project.
* [DiffusionAsShader](https://github.com/IGL-HKUST/DiffusionAsShader) - Provides the foundational architecture for this project.

We thank the authors and contributors of these projects for their valuable contributions to the open-source community!