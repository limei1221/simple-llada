# Modified from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import Config
from model import TransEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="Path to checkpoint file to load.",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=["34M", "85M", "1B"],
        required=True,
        help="Model size - must match the checkpoint.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, there was a little girl",
        help="Text prompt for generation.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Number of sampling steps.",
    )
    p.add_argument(
        "--gen_length",
        type=int,
        default=128,
        help="Generated answer length.",
    )
    p.add_argument(
        "--block_length",
        type=int,
        default=32,
        help="Block length for semi-autoregressive remasking.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    p.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale.",
    )
    p.add_argument(
        "--remasking",
        type=str,
        choices=["low_confidence", "random"],
        default="low_confidence",
        help="Remasking strategy.",
    )
    return p.parse_args()


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (
                num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
    print(f"Using device: {device}")

    config = Config.from_name(f"Diff_LLaMA_{args.model}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    model = TransEncoder(config).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    model.eval()

    input_ids = tokenizer(args.prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    print(f"Prompt: {args.prompt}")

    print("Generating...")
    out = generate(
        model,
        input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        mask_id=config.vocab_size
    )

    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"Generated: {generated_text}")


if __name__ == '__main__':
    main()
