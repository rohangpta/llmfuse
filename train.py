"""
Distributed training script for Qwen models using SFTTrainer.

This script is executed by torchrun for multi-GPU distributed training.
Based on Modal's multinode training guide patterns.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        print(f"Initializing process group for rank {os.environ.get('RANK', 0)}")
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}"),
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"Rank {os.environ.get('RANK', 0)} using GPU {torch.cuda.current_device()}")


def load_training_data(train_split=0.9, output_dir="/root/output_models"):
    """Load and format training data with train/test split."""
    training_data_path = "/root/training_data.jsonl"

    if not os.path.exists(training_data_path):
        raise FileNotFoundError(
            "Training data not found. Please run prepare_training_data first."
        )

    examples = []
    with open(training_data_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))

    print(f"✅ Loaded {len(examples)} total examples")

    # Split into train/test
    import random

    random.seed(42)  # For reproducible splits
    random.shuffle(examples)

    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    print(
        f"📊 Split: {len(train_examples)} train, {len(test_examples)} test ({train_split:.0%}/{1 - train_split:.0%})"
    )

    # Save test set for later evaluation (to the output volume)
    test_data_path = f"{output_dir}/../test_data.jsonl"
    with open(test_data_path, "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")
    print(f"💾 Saved test set to {test_data_path}")

    return Dataset.from_list(train_examples)


def main():
    """Main distributed training function."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    # Setup distributed training
    setup_distributed()

    # Model mapping
    model_mapping = {
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3-14b": "Qwen/Qwen3-14B",
        "qwen3-32b": "Qwen/Qwen3-32B",
    }

    full_model_name = model_mapping.get(args.model_name, args.model_name)

    print(f"🚀 Starting distributed SFT training for {args.model_name}")
    print(f"📦 Model: {full_model_name}")

    if args.use_wandb and int(os.environ.get("RANK", 0)) == 0:
        import wandb

        wandb.init(
            project="qwen3-filesystem-sft",
            name=f"qwen-sft-{args.model_name}-{args.num_epochs}epochs-distributed",
            config={
                "model_name": args.model_name,
                "full_model_name": full_model_name,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "distributed": True,
                "num_gpus": torch.cuda.device_count(),
            },
        )

    # Load training data
    print("📁 Loading training data...")
    dataset = load_training_data(output_dir=args.output_dir)
    print(f"📊 Dataset size: {len(dataset)}")

    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        full_model_name,
        trust_remote_code=True,
        cache_dir="/root/models",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        trust_remote_code=True,
        cache_dir="/root/models",
        torch_dtype=torch.bfloat16,
    )

    print(f"📊 Model parameters: {model.num_parameters():,}")

    # Disable KV caching for training (from Modal guide)
    model.config.use_cache = False

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Calculate global batch size for consistent training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = args.batch_size * world_size * 4  # 4 is gradient accumulation

    print(f"🔧 Training configuration:")
    print(f"   World size: {world_size}")
    print(f"   Per-device batch size: {args.batch_size}")
    print(f"   Gradient accumulation: 4")
    print(f"   Global batch size: {global_batch_size}")

    # Training configuration using SFTConfig
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        save_strategy="steps",
        bf16=True,
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen-sft-{args.model_name}-{args.num_epochs}epochs-distributed"
        if args.use_wandb
        else None,
        # Distributed training settings
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Initialize SFTTrainer
    print("🏋️ Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )

    # Start training
    print("🚀 Starting distributed training...")
    trainer.train()

    # Save model (only on rank 0)
    if int(os.environ.get("RANK", 0)) == 0:
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        print(f"✅ Training completed! Model saved to: {args.output_dir}")

    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
