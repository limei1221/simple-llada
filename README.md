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
# Train a 6M parameter model
python pretrain.py --model 6M --batch_size 32 --memory_efficient

# Resume training from 19M checkpoint
python pretrain.py --model 19M --batch_size 32 --memory_efficient --resume_from checkpoints/step_0001000.pth
```

### Generation

Generate text using a trained model:

```bash
python generate.py
```

## Model Architecture

The implementation uses a simplified Transformer encoder architecture:

- **RoPE Positional Embeddings**: Rotary positional embeddings for better sequence modeling
- **Group Query Attention**: Efficient attention mechanism with grouped queries
- **Mask Token Handling**: Special handling for the [MASK] token (ID: 126336)
- **Linear Noise Schedule**: Linear noise scheduling for the diffusion process

## Configuration

Model configurations are defined in `pretrain.py`. The parameter counts (e.g., 6M) refer to the total parameters excluding embedding parameters:

- **6M**: 6 layers, 6 heads, 384 dimensions
- **19M**: 12 layers, 12 heads, 512 dimensions  
- **34M**: 16 layers, 16 heads, 640 dimensions
- **1B**: 24 layers, 24 heads, 1024 dimensions


## Citations

This implementation is based on the following research papers and codebases:

### Papers

- **LLaDA**: "Large Language Diffusion Models" by the GSAI-ML team. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **SMDM**: "Scaling up Masked Diffusion Models on Text" by the GSAI-ML team. [arXiv:2410.18514](https://arxiv.org/abs/2410.18514)

### Original Codebases

- **LLaDA**: [GSAI-ML/LLaDA](https://github.com/GSAI-ML/LLaDA)
- **SMDM**: [GSAI-ML/SMDM](https://github.com/GSAI-ML/SMDM)

## Acknowledgments

This implementation is inspired by and adapted from:

1. The original LLaDA paper and implementation by GSAI-ML
2. The SMDM simplified approach for educational purposes
3. The Hugging Face Transformers library for model loading and tokenization
4. The Weights & Biases platform for experiment tracking

## License

This project is released under the same license as the original LLaDA implementation. Please refer to the original repositories for specific licensing information.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this educational implementation.

## Disclaimer

This is an educational implementation intended for learning and research purposes. For production use, please refer to the official LLaDA implementation by GSAI-ML. 
