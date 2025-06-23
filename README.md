# Simple-LLaDA

A simplified implementation of **LLaDA** ([Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)) and/or **SMDM** ([Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)), since the basic model structures are similar except for their scale. This project provides a minimal, educational implementation of masked diffusion language modeling that can run on a single GPU.

## Overview

This repository implements a simplified version of LLaDA, which uses masked diffusion modeling for text generation. The implementation is based on the LLaDA and SMDM official codebases and provides:

- **Educational code**: Clean, well-commented implementation focused on clarity over optimization
- **Single GPU support**: Designed to run on consumer hardware
- **Multiple model sizes**: From 6M to 1B parameters
- **Training and inference**: Complete pipeline for training and text generation
- **WandB integration**: Training monitoring and logging

## Key Features

- **Masked Diffusion Process**: Implements the forward and reverse diffusion processes for text generation
- **Semi-Autoregressive Generation**: Supports block-wise generation with configurable block sizes
- **Classifier-Free Guidance**: Optional CFG for improved generation quality
- **Memory Efficient Training**: Gradient checkpointing and memory cleanup options

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd simple-llada
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model with different sizes:

```bash
python pretrain_mdm.py --model 85M --batch_size 16
```

### Generation

Generate text using a trained model:

```bash
python generate.py --model 85M --ckpt_path checkpoints/step_0010000.pth
```

## Model Architecture

The implementation uses a simplified Transformer encoder architecture:

- **RoPE Positional Embeddings**: Rotary positional embeddings for better sequence modeling
- **Group Query Attention**: Efficient attention mechanism with grouped queries
- **Mask Token Handling**: Special handling for the [MASK] token
- **Linear Noise Schedule**: Linear noise scheduling for the diffusion process

## Citations

This implementation is based on the following research papers and codebases:

### Papers

- **LLaDA**: "Large Language Diffusion Models" [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **SMDM**: "Scaling up Masked Diffusion Models on Text" [arXiv:2410.18514](https://arxiv.org/abs/2410.18514)

### Original Codebases

- **LLaDA**: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
- **SMDM**: [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM)

## License

This project is released under the same license as the original LLaDA implementation. Please refer to the original repositories for specific licensing information.

## Disclaimer

This is an educational implementation intended for learning and research purposes. For production use, please refer to the official LLaDA implementation by ML-GSAI.
