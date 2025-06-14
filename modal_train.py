"""
Modal training script for fine-tuning Qwen models using SFTTrainer.

This script uses Supervised Fine-Tuning Trainer (SFTTrainer) from the TRL library
to fine-tune Qwen models on filesystem operation data.

Usage:
    modal run modal_train.py::train_qwen --model-name="qwen3-8b"
    modal run modal_train.py::train_qwen --model-name="qwen3-8b" --use-wandb
"""

import modal
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Define the Modal app
app = modal.App("qwen3-sft-training")

# Create Modal Volumes
model_cache = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
training_data_volume = modal.Volume.from_name(
    "qwen3-training-data", create_if_missing=True
)
trained_models_volume = modal.Volume.from_name(
    "qwen3-trained-models", create_if_missing=True
)

# Mount paths
MODEL_CACHE_PATH = "/root/models"
TRAINING_DATA_PATH = "/root/training_data"
OUTPUT_MODEL_PATH = "/root/output_models"

# Qwen3 model sizes mapping (same as in modal_run.py)
QWEN3_MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

# Create the training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "trl>=0.7.0",  # For SFTTrainer
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "wandb>=0.16.0",  # For experiment tracking
            "huggingface_hub[hf_transfer]>=0.19.0",
            "jsonlines>=4.0.0",
        ]
    )
    .apt_install(["git"])
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file("training_data.jsonl", "/root/training_data.jsonl")
)


def format_training_data(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format training data for SFTTrainer."""
    formatted_data = []

    for example in examples:
        # Create a conversation format for chat fine-tuning
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")

        # Format as conversation
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

        formatted_data.append(
            {
                "messages": conversation,
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>",
            }
        )

    return formatted_data


@app.function(
    image=training_image,
    volumes={
        MODEL_CACHE_PATH: model_cache,
        TRAINING_DATA_PATH: training_data_volume,
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=60,  # 1 hour for data preparation
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prepare_training_data():
    """Prepare and upload training data to the volume."""

    print("📁 Preparing training data...")

    # Check if training_data.jsonl exists locally and upload it
    local_data_file = "training_data.jsonl"
    remote_data_file = f"{TRAINING_DATA_PATH}/training_data.jsonl"

    if not os.path.exists(local_data_file):
        print(f"❌ {local_data_file} not found locally")
        return "Error: training_data.jsonl not found"

    # Copy local file to volume
    import shutil

    os.makedirs(TRAINING_DATA_PATH, exist_ok=True)
    shutil.copy(local_data_file, remote_data_file)

    # Validate the data
    training_examples = []
    with open(remote_data_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                if "prompt" in example and "completion" in example:
                    training_examples.append(example)
                else:
                    print(f"⚠️  Line {line_num}: Missing 'prompt' or 'completion' field")
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: Invalid JSON - {e}")

    print(f"✅ Loaded {len(training_examples)} valid training examples")

    # Commit changes to volume
    training_data_volume.commit()

    return f"Successfully prepared {len(training_examples)} training examples"


@app.function(
    image=training_image,
    gpu="H100",  # Single H100 for training
    volumes={
        MODEL_CACHE_PATH: model_cache,
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=4 * 60 * 60,  # 4 hours for training
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_qwen(
    model_name: str = "qwen3-8b",
    use_wandb: bool = True,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_checkpointing: bool = True,
):
    """Fine-tune a Qwen model using SFTTrainer."""

    print(f"🚀 Starting SFT training for {model_name}")
    print("=" * 60)

    # Convert model name to full HF model name
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"📦 Model: {full_model_name}")

    # Import training libraries
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
    import json

    # Setup W&B if requested
    if use_wandb:
        import wandb

        run_name = f"qwen-sft-{model_name}-{num_epochs}epochs"
        os.environ["WANDB_PROJECT"] = "qwen3-filesystem-sft"
        os.environ["WANDB_RUN_NAME"] = run_name
        print(f"📊 W&B Project: qwen3-filesystem-sft, Run: {run_name}")

        # Load training data
    print("📁 Loading training data...")
    training_data_file = "/root/training_data.jsonl"  # File baked into image

    if not os.path.exists(training_data_file):
        raise FileNotFoundError(f"Training data not found: {training_data_file}")

    # Load JSONL data
    training_examples = []
    with open(training_data_file, "r") as f:
        for line in f:
            if line.strip():
                training_examples.append(json.loads(line.strip()))

    print(f"✅ Loaded {len(training_examples)} training examples")

    # Format data for SFTTrainer
    formatted_data = format_training_data(training_examples)

    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    print(f"📊 Dataset size: {len(dataset)}")

    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        full_model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_PATH
    )

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for full fine-tuning
    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"📊 Model parameters: {model.num_parameters():,}")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Training configuration using SFTConfig
    output_dir = f"{OUTPUT_MODEL_PATH}/{model_name}-sft-{num_epochs}epochs"

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,  # Save checkpoints more frequently
        save_total_limit=5,  # Keep more checkpoints
        save_strategy="steps",
        bf16=True,
        max_seq_length=2048,  # This is supported in SFTConfig
        dataset_text_field="text",  # Specify the text field
        packing=False,  # Don't pack sequences for better quality
        report_to="wandb" if use_wandb else "none",
        run_name=f"qwen-sft-{model_name}-{num_epochs}epochs" if use_wandb else None,
    )

    # Initialize SFTTrainer
    print("🏋️ Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )

    # Start training
    print("🚀 Starting training...")
    print(f"   Model: {full_model_name}")
    print(f"   Examples: {len(dataset)}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Full fine-tuning: Yes")
    print(f"   Output: {output_dir}")

    # Train the model
    trainer.train()

    # Save the final model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Commit to volume
    trained_models_volume.commit()

    print(f"✅ Training completed! Model saved to: {output_dir}")

    # Training statistics
    train_logs = trainer.state.log_history
    final_loss = train_logs[-1].get("train_loss", "N/A") if train_logs else "N/A"

    return {
        "model_name": model_name,
        "full_model_name": full_model_name,
        "output_dir": output_dir,
        "training_examples": len(dataset),
        "epochs": num_epochs,
        "final_loss": final_loss,
        "training_type": "full_fine_tuning",
    }


@app.function(
    image=training_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=30 * 60,  # 30 minutes for testing
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_trained_model(model_path: str, test_prompts: Optional[List[str]] = None):
    """Test a trained model with sample prompts."""

    print(f"🧪 Testing trained model: {model_path}")

    if test_prompts is None:
        test_prompts = [
            "What happens when I run 'mkdir /home/user/test' on a filesystem?",
            "Explain the difference between 'rm' and 'rmdir' commands.",
            "How do I change file permissions using chmod?",
        ]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    full_model_path = f"{OUTPUT_MODEL_PATH}/{model_path}"

    if not os.path.exists(full_model_path):
        return f"Error: Model not found at {full_model_path}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_model_path)

    # Load model (check if it's a PEFT model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error loading model: {e}"

    model.eval()

    results = []

    for prompt in test_prompts:
        print(f"\n📝 Testing prompt: {prompt[:50]}...")

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        ).strip()

        results.append({"prompt": prompt, "response": response})

        print(f"🤖 Response: {response[:100]}...")

    return {"model_path": model_path, "test_results": results}


@app.local_entrypoint()
def main():
    """Local entrypoint showing available training functions."""
    print("Available Modal training functions:")
    print(
        "  modal run modal_train.py::prepare_training_data                     # Upload training data"
    )
    print(
        "  modal run modal_train.py::train_qwen --model-name='qwen3-8b'        # Train with SFTTrainer"
    )
    print(
        "  modal run modal_train.py::train_qwen --model-name='qwen3-8b' --use-wandb  # Train with W&B"
    )
    print(
        "  modal run modal_train.py::test_trained_model --model-path='qwen3-8b-sft-3epochs'  # Test model"
    )

    print("\nTraining options:")
    print(
        "  --model-name: qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b"
    )
    print("  --num-epochs: Number of training epochs (default: 3)")
    print("  --batch-size: Per-device batch size (default: 4)")
    print("  --learning-rate: Learning rate (default: 2e-5)")
    print("  --use-lora: Use LoRA fine-tuning (default: True)")
    print("  --use-wandb: Enable W&B logging (default: False)")


if __name__ == "__main__":
    main()
