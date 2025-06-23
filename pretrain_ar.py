# Modified from https://github.com/ML-GSAI/SMDM/blob/main/pretrain/train_mdm_rl.py


import argparse
import datetime
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional, Tuple

import wandb
from config import Config
from model import TransEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        choices=["34M", "85M", "1B"],
        required=True,
        help="Name suffix - see Config.from_name for mapping.",
    )
    p.add_argument(
        "--flops",
        type=float,
        default=0.1,
        help="Target training compute in 10^18 FLOPs.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=5,
        help="Number of steps to accumulate gradients before updating weights.",
    )
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("checkpoints_ar"),
        help="Directory where checkpoints will be saved.",
    )
    p.add_argument(
        "--resume_from",
        type=Path,
        help="Path to checkpoint file to resume training from.",
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log training progress every N steps.",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=1000,
        help="Run evaluation and save checkpoint every N steps.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_message(message: str, log_file: Path) -> None:
    """Log a message both to console and file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_latest: bool = True) -> None:
    """Delete old checkpoint files, keeping only the latest one if keep_latest is True."""
    if not checkpoint_dir.exists():
        return

    checkpoint_files = list(checkpoint_dir.glob("step_*.pth"))

    if len(checkpoint_files) <= 1:
        return

    if keep_latest:

        def get_step_number(file_path):
            filename = file_path.stem  # Remove extension
            step_str = filename.replace("step_", "")
            return int(step_str)

        checkpoint_files.sort(key=get_step_number)

        for checkpoint_file in checkpoint_files[:-1]:
            checkpoint_file.unlink()
            print(f"Deleted old checkpoint: {checkpoint_file}")
    else:
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()
            print(f"Deleted checkpoint: {checkpoint_file}")


def cleanup_gpu_memory():
    """Clean up GPU memory to prevent fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
        }
    return None


def load_checkpoint(
    checkpoint_path: Path,
    model: TransEncoder,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    """Load model and optimizer state from checkpoint, return the step number."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)

    return step


def save_checkpoint(
    model: TransEncoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_path: Path,
) -> None:
    """Save model, optimizer state, and current step to checkpoint.

    This function is called after completed accumulation cycles, ensuring
    the model state is consistent and ready for resuming training.
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch, *, seq_len: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="pt",
    )


def make_loader(ds, *, batch_size: int, seq_len: int) -> DataLoader:
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs={"seq_len": seq_len},
        remove_columns=ds.column_names,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # shuffle=False since we're using an IterableDataset
        pin_memory=True,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        },
    )


def count_total_parameters(model: TransEncoder) -> int:
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_embedding_parameters(model: TransEncoder) -> int:
    """Count parameters in the embedding layer only."""
    return model.embed_tokens.weight.numel() + model.lm_head.weight.numel()


def get_lr(
    step: int, warmup: int, total: int, base_lr: float = 5e-4, min_lr: float = 5e-5
) -> float:
    if step < warmup:
        return base_lr * step / warmup
    ratio = (step - warmup) / max(1, total - warmup)
    assert 0 <= ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def train():
    args = parse_args()
    set_seed(args.seed)

    assert args.log_every % args.gradient_accumulation_steps == 0
    assert args.eval_every % args.gradient_accumulation_steps == 0

    log_file = args.out_dir / "log.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        run_name = f"ar_{args.model}_flops{args.flops}_bs{args.batch_size}"
        if args.resume_from:
            run_name += f"_resume_{args.resume_from}"

        wandb.init(
            project="simple-llada",
            name=run_name,
            config={
                "model_size": f"{args.model}",
                "flops": args.flops,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "out_dir": str(args.out_dir),
                "resume_from": str(args.resume_from) if args.resume_from else None,
            },
        )

    log_message(
        f"Starting training with model={args.model}, flops={args.flops}*10^18, batch_size={args.batch_size}",
        log_file,
    )

    config = Config.from_name(f"Diff_LLaMA_{args.model}")
    log_message(f"[INFO] Block size: {config.block_size}", log_file)
    config.vocab_size = tokenizer.vocab_size
    log_message(f"[INFO] Vocab size: {config.vocab_size}", log_file)

    model = TransEncoder(config, is_causal=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )
    model.to(device)
    log_message(f"[INFO] Using device: {device}", log_file)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)

    total_params = count_total_parameters(model)
    embedding_params = count_embedding_parameters(model)
    non_embedding_params = total_params - embedding_params

    if not args.no_wandb:
        wandb.config.update(
            {
                "total_params": total_params,
                "non_embedding_params": non_embedding_params,
                "embedding_params": embedding_params,
                "device": device,
                "vocab_size": config.vocab_size,
                "block_size": config.block_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "n_query_groups": config.n_query_groups,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }
        )

    log_message(f"[INFO] Total parameters: {total_params / 1e6:.2f}M", log_file)
    log_message(f"[INFO] Non-embedding parameters: {non_embedding_params / 1e6:.2f}M", log_file)

    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True, token=True)
    # dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

    train_loader = make_loader(
        dataset, batch_size=args.batch_size, seq_len=config.block_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1
    )

    est_tokens_per_step = args.batch_size * config.block_size
    est_flops_per_token = 6 * non_embedding_params  # transformer rule-of-thumb
    total_steps = int((args.flops * 1e18) / (est_flops_per_token * est_tokens_per_step))
    total_steps = max(total_steps, 1000)

    warmup = max(100, total_steps // 100)
    log_message(
        f"[INFO] Training for ~{total_steps:,} steps (warm-up {warmup})", log_file
    )

    if not args.no_wandb:
        wandb.config.update(
            {
                "total_steps": total_steps,
                "warmup_steps": warmup,
                "est_tokens_per_step": est_tokens_per_step,
                "est_flops_per_token": est_flops_per_token,
                "flops_calculation": {
                    "non_embedding_params": non_embedding_params,
                    "tokens_per_step": est_tokens_per_step,
                    "flops_per_token": est_flops_per_token,
                    "total_flops": est_flops_per_token * est_tokens_per_step * total_steps,
                },
            }
        )

    step = 0
    if args.resume_from:
        try:
            step = load_checkpoint(args.resume_from, model, optimizer, device)
            log_message(f"[INFO] Resuming training from {args.resume_from}", log_file)
        except Exception as e:
            log_message(f"[ERROR] Failed to load checkpoint: {e}", log_file)
            raise

    optimizer.zero_grad()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            lr = get_lr(step, warmup, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            if torch.rand(1) < 0.01:  # variable length input augmentation
                length = torch.randint(1, config.block_size + 1, (1,))
                input_ids = input_ids[:, :length]

            logits = model(input_ids)  # (B,T,V)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss = loss / args.gradient_accumulation_steps

            loss.backward()

            step += 1

            completed_accumulation = step % args.gradient_accumulation_steps == 0

            if completed_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                cleanup_gpu_memory()

                if step % args.log_every == 0:
                    memory_info = get_gpu_memory_info()

                    if not args.no_wandb:
                        log_data = {
                            "train/loss": loss.item() * args.gradient_accumulation_steps,
                            "train/learning_rate": lr,
                            "train/step": step,
                        }

                        if memory_info:
                            log_data.update(
                                {
                                    "train/gpu_allocated_gb": memory_info[
                                        "allocated_gb"
                                    ],
                                    "train/gpu_reserved_gb": memory_info["reserved_gb"],
                                    "train/gpu_free_gb": memory_info["free_gb"],
                                }
                            )

                        wandb.log(log_data)

                    log_message(
                        f"step {step:>6} | loss {loss.item() * args.gradient_accumulation_steps:.4f} | lr {lr:.2e}",
                        log_file,
                    )

                if step % args.eval_every == 0:
                    validate(model, device, config.vocab_size, log_file, args.no_wandb)

                    cleanup_old_checkpoints(args.out_dir, keep_latest=True)

                    ckpt_path = args.out_dir / f"step_{step:07d}.pth"
                    save_checkpoint(model, optimizer, step, ckpt_path)
                    log_message(f"[ckpt] saved model to {ckpt_path}", log_file)

    if not args.no_wandb:
        wandb.finish()


def validate(model, device, vocab_size: int, log_file: Path, no_wandb: bool) -> None:
    """Generate sample text using the current model state."""
    model.eval()
    with torch.no_grad():
        prompt_text = "Once upon a time,"

        tokens = tokenizer(prompt_text, return_tensors="pt")

        out = generate(
            model,
            input_ids=tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
            max_length=128,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(out[0], skip_special_tokens=True)

        log_message(f"[val] Prompt: {prompt_text}", log_file)
        log_message(f"[val] Generated: {generated_text}", log_file)

        if not no_wandb:
            wandb.log(
                {
                    "val/generated_text": generated_text,
                }
            )

    model.train()

def generate(
    model,
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
    **kwargs
) -> torch.LongTensor:
    """
    Generate text using the model with temperature and top-k sampling.

    Args:
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
        pad_token_id = getattr(model.config, 'pad_token_id', 0)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, 'eos_token_id', pad_token_id)

    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Handle multiple sequences per prompt
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
        logits = model(generated)  # (B, T, V+1)
        
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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train()
