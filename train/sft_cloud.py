#!/usr/bin/env python3
"""
Cloud-optimized training script for Qwen models using SFTTrainer.
Designed for Modal's distributed training environment.
"""

import argparse
import json
import os
import sys
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


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


def load_training_data(training_data_path, train_split=0.9, output_dir="/root/output_models"):
    """Load and format training data with train/test split."""
    
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data not found: {training_data_path}")

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

    print(f"📊 Split: {len(train_examples)} train, {len(test_examples)} test ({train_split:.0%}/{1 - train_split:.0%})")

    # Save test set for later evaluation
    os.makedirs(output_dir, exist_ok=True)
    test_data_path = os.path.join(output_dir, "test_data.jsonl")
    with open(test_data_path, "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")
    print(f"💾 Saved test set to {test_data_path}")

    return Dataset.from_list(train_examples)


def main():
    """Main distributed training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True, help="Path to training data JSONL file")
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
        "qwen3-0.6b": "Qwen/Qwen2.5-0.5B",          # Base model (no instruct)
        "qwen3-1.7b": "Qwen/Qwen2.5-1.5B",          # Base model  
        "qwen3-3b": "Qwen/Qwen2.5-3B",              # Base model
        "qwen3-4b": "Qwen/Qwen3-4B",                # Actual Qwen3-4B model
        "qwen3-8b": "Qwen/Qwen2.5-7B",              # Base model
        "qwen3-14b": "Qwen/Qwen2.5-14B",            # Base model
        "qwen3-32b": "Qwen/Qwen2.5-32B",            # Base model
    }

    if args.model_name not in model_mapping:
        raise ValueError(f"Model {args.model_name} not supported. Choose from: {list(model_mapping.keys())}")

    model_id = model_mapping[args.model_name]
    print(f"🤖 Using model: {model_id}")

    # Load training data
    print("📁 Loading training data...")
    dataset = load_training_data(training_data_path=args.training_data, output_dir=args.output_dir)
    print(f"📊 Dataset size: {len(dataset)}")

    # Load model and tokenizer
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        use_fast=False,
        padding_side="right",  # Required for causal LM
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For distributed training, don't use device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Disable cache for training
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
        max_seq_length=512,  # Reduced for memory efficiency
        dataset_text_field="text",
        packing=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen-sft-{args.model_name}-{args.num_epochs}epochs-distributed"
        if args.use_wandb
        else None,
        # Distributed training settings
        dataloader_pin_memory=False,
        remove_unused_columns=True,  # Remove prompt/completion after creating text field
    )

    # Transform dataset to have 'text' field that SFTTrainer expects
    def add_text_field(example):
        # Combine prompt and completion into single text field
        example['text'] = f"{example['prompt']}\n{example['completion']}"
        return example
    
    print("🔄 Transforming dataset for SFTTrainer...")
    dataset = dataset.map(add_text_field)
    
    # Initialize SFTTrainer
    print("🏋️ Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
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