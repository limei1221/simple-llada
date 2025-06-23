# Modified from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

import argparse
from pathlib import Path
from typing import Optional

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
        "--max_length",
        type=int,
        default=128,
        help="Maximum length of the generated sequence.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of highest probability tokens to consider for sampling.",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Cumulative probability threshold for nucleus sampling.",
    )
    p.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Whether to use sampling or greedy decoding.",
    )
    p.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate per prompt.",
    )
    return p.parse_args()


@torch.no_grad()
def generate(
    model: TransEncoder,
    input_ids: torch.LongTensor,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    num_return_sequences: int = 1,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.LongTensor:
    """
    Generate text using the autoregressive model with temperature and top-k sampling.

    Args:
        model: Autoregressive model.
        input_ids: Input token IDs of shape (batch_size, seq_len)
        max_length: Maximum length of the generated sequence
        temperature: Temperature for sampling (higher = more random)
        top_k: Number of highest probability tokens to consider for sampling
        top_p: Cumulative probability threshold for nucleus sampling
        do_sample: Whether to use sampling or greedy decoding
        pad_token_id: ID of the padding token
        eos_token_id: ID of the end-of-sequence token
        num_return_sequences: Number of sequences to generate per prompt
        attention_mask: Optional attention mask

    Returns:
        Generated token IDs of shape (batch_size * num_return_sequences, max_length)
    """
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", 0)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", pad_token_id)

    batch_size = input_ids.shape[0]
    device = input_ids.device

    if num_return_sequences > 1:
        input_ids = input_ids.repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)

    # Initialize the generated sequence with input_ids
    generated = input_ids.clone()

    # Create attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    for _ in range(max_length - input_ids.shape[1]):
        # Get logits for the current sequence
        logits = model(generated)  # (B, T, V)

        # Get next token logits
        next_token_logits = logits[:, -1, :] / temperature

        if do_sample:
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append the next token to the generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Update attention mask
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size * num_return_sequences, 1),
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )

        # Check if all sequences have reached EOS token
        if (generated == eos_token_id).any(dim=1).all():
            break

    return generated


def main():
    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )
    print(f"Using device: {device}")

    config = Config.from_name(f"LLaMA_{args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    model = TransEncoder(config, is_causal=True).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(device)
    print(f"Prompt: {args.prompt}")

    print("Generating...")
    out = generate(
        model,
        input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=args.num_return_sequences,
    )

    generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)

    if args.num_return_sequences == 1:
        print(f"Generated: {generated_text[0]}")
    else:
        for i, text in enumerate(generated_text):
            print("-" * 100)
            print(f"Generated {i + 1}: {text}")


if __name__ == "__main__":
    main()
