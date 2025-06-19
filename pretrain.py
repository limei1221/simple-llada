"""Minimal end-to-end training loop for a *Masked Diffusion Model* (MDM).

The script intentionally mirrors the logic in
`pretrain/train_mdm.py` from the SMDM repository, **but**:
    * uses a *vastly* smaller default model defined in ``model.py``
    * removes every distributed/TPU/flash-attention optimisation
    * fits comfortably on a single GPU (or even CPU for toy runs)

Usage
-----
python train_mdm.py --model 6M --batch_size 32 --memory_efficient
python train_mdm.py --model 6M --batch_size 32 --memory_efficient --resume_from checkpoints/step_0001000.pth
"""

from __future__ import annotations

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

import wandb
from generate import generate
from model import TransEncoder


# -------------------------------------------------------------------------
# Argument parsing & reproducibility
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        choices=["6M", "19M", "34M", "1B"],
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
        default=Path("checkpoints"),
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
    p.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Enable memory efficient training with gradient checkpointing and cleanup.",
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

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("step_*.pth"))

    if len(checkpoint_files) <= 1:
        return  # No need to clean up if there's only one or no checkpoints

    if keep_latest:
        # Sort by step number (extract step number from filename)
        def get_step_number(file_path):
            filename = file_path.stem  # Remove extension
            step_str = filename.replace("step_", "")
            return int(step_str)

        checkpoint_files.sort(key=get_step_number)

        # Keep the latest (last in sorted list) and delete the rest
        for checkpoint_file in checkpoint_files[:-1]:
            checkpoint_file.unlink()
            print(f"Deleted old checkpoint: {checkpoint_file}")
    else:
        # Delete all checkpoints
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
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
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

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Get the step number
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


# -------------------------------------------------------------------------
# Config - identical to the one in the original codebase, but *single file*
# -------------------------------------------------------------------------
class Config:
    def __init__(
        self,
        n_layer: int,
        n_head: int,
        n_embd: int,
        *,
        n_query_groups: int = 1,
        block_size: int = 1024,
        bias: bool = False,
        vocab_size: int = 32_000,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_query_groups = n_query_groups
        self.block_size = block_size
        self.bias = bias
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.norm_eps = norm_eps

        # derived
        self.head_dim = n_embd // n_head
        self.intermediate_size = 4 * n_embd

    # ---- presets ---------------------------------------------------------
    @classmethod
    def from_name(cls, name: str) -> "Config":
        if (
            name == "Diff_LLaMA_6M"
        ):  # 4.73M non-embedding parameters (6M model in SMDM codebase)
            return cls(n_layer=6, n_head=4, n_embd=256, n_query_groups=1)
        if (
            name == "Diff_LLaMA_19M"
        ):  # 14M non-embedding parameters (19M model in SMDM paper, Table 8)
            return cls(n_layer=8, n_head=6, n_embd=384, n_query_groups=1)
        if (
            name == "Diff_LLaMA_34M"
        ):  # 25M non-embedding parameters (34M model in SMDM paper, Table 8)
            return cls(n_layer=8, n_head=8, n_embd=512, n_query_groups=1)
        if (
            name == "Diff_LLaMA_1B"
        ):  # 946M non-embedding parameters (1B model in LLaDA paper, Table 5)
            return cls(n_layer=22, n_head=32, n_embd=2048, n_query_groups=8)
        raise ValueError(name)


# -------------------------------------------------------------------------
# Model wrapper for compatibility with generate function
# -------------------------------------------------------------------------
class ModelWrapper:
    """Wrapper to make TransEncoder compatible with the generate function."""

    def __init__(self, model: TransEncoder):
        self.model = model
        # The mask_id should be vocab_size (the last token in the vocabulary)
        self.mask_id = model.config.vocab_size

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __call__(self, x):
        # Return an object with logits attribute like HuggingFace models
        class Output:
            def __init__(self, logits):
                self.logits = logits

        logits = self.model(x)
        return Output(logits)


# -------------------------------------------------------------------------
# Forward diffusion (token masking) - identical to the one in the paper
# -------------------------------------------------------------------------
def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


# -------------------------------------------------------------------------
# Data --------------------------------------------------------------------
# -------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch, *, seq_len: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=seq_len,
        padding="max_length",
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
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack(
                [torch.tensor(x["attention_mask"]) for x in batch]
            ),
        },
    )


# -------------------------------------------------------------------------
# Model parameter counting
# -------------------------------------------------------------------------


def count_total_parameters(model: TransEncoder) -> int:
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_embedding_parameters(model: TransEncoder) -> int:
    """Count parameters in the embedding layer only."""
    return model.embed_tokens.weight.numel() + model.lm_head.weight.numel()


# -------------------------------------------------------------------------
# Training loop -----------------------------------------------------------
# -------------------------------------------------------------------------


def get_lr(
    step: int, warmup: int, total: int, base_lr: float = 2e-4, min_lr: float = 2e-5
) -> float:
    if step < warmup:
        return base_lr * step / warmup
    # cosine
    ratio = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def train():
    args = parse_args()
    set_seed(args.seed)

    assert args.log_every % args.gradient_accumulation_steps == 0
    assert args.eval_every % args.gradient_accumulation_steps == 0

    # Create log file
    log_file = args.out_dir / "log.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not args.no_wandb:
        run_name = f"mdm_{args.model}_flops{args.flops}_bs{args.batch_size}"
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

    # Log training configuration
    log_message(
        f"Starting training with model={args.model}, flops={args.flops}*10^18, batch_size={args.batch_size}",
        log_file,
    )

    # ---------------------------- model -----------------------------------
    config = Config.from_name(f"Diff_LLaMA_{args.model}")
    config.vocab_size = tokenizer.vocab_size
    log_message(f"[INFO] Vocab size: {config.vocab_size}", log_file)

    model = TransEncoder(config)
    model.apply(model._init_weights)  # simple initialisation

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )
    model.to(device)
    log_message(f"[INFO] Using device: {device}", log_file)

    # Memory optimization settings
    if args.memory_efficient:
        # Set memory fraction to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use at most 90% of GPU memory
        log_message("[INFO] Enabled memory efficient training", log_file)

    # Log model info to wandb
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
                "memory_efficient": args.memory_efficient,
            }
        )

    log_message(f"[INFO] Total parameters: {total_params / 1e6:.2f}M", log_file)
    log_message(f"[INFO] Embedding parameters: {embedding_params / 1e6:.2f}M", log_file)
    log_message(
        f"[INFO] Non-embedding parameters: {non_embedding_params / 1e6:.2f}M", log_file
    )

    # ---------------------------- data ------------------------------------
    # dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True, token=True)
    dataset = load_dataset(
        "openwebtext", split="train", streaming=True, trust_remote_code=True
    )

    train_loader = make_loader(
        dataset, batch_size=args.batch_size, seq_len=config.block_size
    )

    # ---------------------------- optimiser -------------------------------
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Compute a *very* rough step budget from requested FLOPs
    est_tokens_per_step = args.batch_size * config.block_size
    est_flops_per_token = 6 * non_embedding_params  # transformer rule-of-thumb
    total_steps = int((args.flops * 1e18) / (est_flops_per_token * est_tokens_per_step))
    total_steps = max(total_steps, 1000)  # always train a bit

    warmup = max(100, total_steps // 100)
    log_message(
        f"[INFO] Training for ~{total_steps:,} steps (warm-up {warmup})", log_file
    )

    # Log training schedule to wandb
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
                    "total_flops": est_flops_per_token
                    * est_tokens_per_step
                    * total_steps,
                },
            }
        )

    # ---------------------------- main loop -------------------------------
    step = 0

    # Load checkpoint if resuming
    if args.resume_from:
        try:
            step = load_checkpoint(args.resume_from, model, optimiser, device)
            log_message(f"[INFO] Resuming training from {args.resume_from}", log_file)
        except Exception as e:
            log_message(f"[ERROR] Failed to load checkpoint: {e}", log_file)
            raise

    # Ensure gradients are zeroed at start
    optimiser.zero_grad()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            # lr schedule --------------------------------------------------
            lr = get_lr(step, warmup, total_steps)
            for pg in optimiser.param_groups:
                pg["lr"] = lr

            # forward ------------------------------------------------------
            input_ids = batch["input_ids"].to(device)
            if torch.rand(1) < 0.01:  # variable length input
                length = torch.randint(1, config.block_size + 1, (1,))
                input_ids = input_ids[:, :length]
            noisy_input, mask_indices, p_mask = forward_process(
                input_ids, total_dim=config.vocab_size
            )

            logits = model(noisy_input)  # (B,T,V)
            loss = (
                F.cross_entropy(
                    logits[mask_indices],
                    input_ids[mask_indices],
                    reduction="none",
                )
                / p_mask[mask_indices]
            )
            # loss = loss.mean()
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps

            # backward -----------------------------------------------------
            loss.backward()

            # Increment step first
            step += 1

            # Check if we just completed an accumulation cycle
            # This ensures we only perform optimizer steps and save checkpoints
            # after the gradients have been accumulated for the full cycle
            completed_accumulation = step % args.gradient_accumulation_steps == 0

            if completed_accumulation:
                # Perform optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad()

                # Memory cleanup after optimizer step
                if args.memory_efficient:
                    cleanup_gpu_memory()

                # Logging and checkpointing only after completed accumulation cycles
                if step % args.log_every == 0:
                    # Get memory info
                    memory_info = get_gpu_memory_info()

                    # Log to wandb
                    if not args.no_wandb:
                        log_data = {
                            "train/loss": loss.item() * args.gradient_accumulation_steps,  # Scale back for logging
                            "train/learning_rate": lr,
                            "train/step": step,
                            "train/mask_ratio": mask_indices.float().mean().item(),
                        }

                        # Add memory info if available
                        if memory_info:
                            log_data.update(
                                {
                                    "train/gpu_allocated_gb": memory_info["allocated_gb"],
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

                    # ---------------- checkpoint ----------------------------
                    # Save checkpoint after completed accumulation cycle
                    # This ensures the model state is consistent and ready for resuming
                    cleanup_old_checkpoints(args.out_dir, keep_latest=True)

                    ckpt_path = args.out_dir / f"step_{step:07d}.pth"
                    save_checkpoint(model, optimiser, step, ckpt_path)
                    log_message(f"[ckpt] saved model to {ckpt_path}", log_file)

    # Final wandb log
    if not args.no_wandb:
        wandb.finish()


def validate(model, device, vocab_size: int, log_file: Path, no_wandb: bool) -> None:
    """Generate sample text using the current model state."""
    model.eval()
    with torch.no_grad():
        prompt_text = "Once upon a time, "

        # Tokenize the prompt
        tokens = tokenizer(prompt_text, return_tensors="pt")
        prompt = tokens.input_ids.to(device)

        # Wrap the model for compatibility with generate function
        wrapped_model = ModelWrapper(model)

        out = generate(
            wrapped_model,
            prompt,
            steps=128,
            gen_length=128,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=wrapped_model.mask_id,
        )
        generated_text = tokenizer.batch_decode(
            out[:, prompt.shape[0] :], skip_special_tokens=True
        )[0]

        log_message(f"[val] Prompt: {prompt_text}", log_file)
        log_message(f"[val] Generated: {generated_text}", log_file)

        # Log to wandb
        if not no_wandb:
            wandb.log(
                {
                    "val/generated_text": wandb.Html(
                        f"<p><strong>Prompt:</strong> {prompt_text}</p><p><strong>Generated:</strong> {generated_text}</p>"
                    ),
                }
            )

    model.train()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train()
