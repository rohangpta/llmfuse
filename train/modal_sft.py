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
    .add_local_file("data/training_data_100.jsonl", "/root/training_data.jsonl")
    .add_local_file("train/train.py", "/root/train.py")
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
    gpu="H100:8",  # 8x H100 GPUs for distributed training
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
    batch_size: int = 2,  # Reduced per-device batch size for multi-GPU
    learning_rate: float = 2e-5,
    gradient_checkpointing: bool = True,
):
    """
    Distributed training of Qwen models using torchrun and SFTTrainer.

    Args:
        model_name: Model size to train (qwen3-4b, qwen3-8b, etc.)
        use_wandb: Enable Weights & Biases logging
        num_epochs: Number of training epochs
        batch_size: Per-device batch size (global batch size = batch_size * num_gpus * grad_accum)
        learning_rate: Learning rate for training
        gradient_checkpointing: Enable gradient checkpointing to save memory
    """
    import subprocess
    import tempfile
    import os

    print(f"🚀 Starting distributed SFT training for {model_name}")
    print(f"🔧 Configuration:")
    print(f"   GPUs: 8x H100")
    print(f"   Per-device batch size: {batch_size}")
    print(f"   Global batch size: {batch_size * 8 * 4}")  # 8 GPUs * 4 grad accum
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   W&B logging: {use_wandb}")

    # Use the train.py file that's baked into the image
    training_script_path = "/root/train.py"

    if not os.path.exists(training_script_path):
        raise FileNotFoundError(f"Training script not found: {training_script_path}")

    try:
        # Set up output directory
        output_dir = (
            f"{OUTPUT_MODEL_PATH}/{model_name}-sft-{num_epochs}epochs-distributed"
        )

        # Set up environment for W&B
        env = os.environ.copy()
        if use_wandb:
            env["WANDB_PROJECT"] = "qwen3-filesystem-sft"

        # Prepare torchrun arguments
        torchrun_args = [
            "torchrun",
            "--nnodes=1",  # Single node
            "--nproc-per-node=8",  # 8 GPUs
            "--master-addr=localhost",
            "--master-port=29500",
            training_script_path,
            "--model_name",
            model_name,
            "--output_dir",
            output_dir,
            "--num_epochs",
            str(num_epochs),
            "--batch_size",
            str(batch_size),
            "--learning_rate",
            str(learning_rate),
        ]

        if use_wandb:
            torchrun_args.append("--use_wandb")

        if gradient_checkpointing:
            torchrun_args.append("--gradient_checkpointing")

        print(f"🔥 Executing torchrun with command:")
        print(f"   {' '.join(torchrun_args)}")

        # Run distributed training (output streams by default)
        result = subprocess.run(torchrun_args, env=env, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Training failed with return code {result.returncode}")

        # Commit trained model to volume
        trained_models_volume.commit()

        print("✅ Distributed training completed successfully!")

        return {
            "model_name": model_name,
            "output_dir": output_dir,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "global_batch_size": batch_size * 8 * 4,
            "learning_rate": learning_rate,
            "training_type": "distributed_sft",
            "num_gpus": 8,
        }

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


@app.function(
    image=training_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=30 * 60,  # 30 minutes for test evaluation
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_trained_model_on_test_set(model_path: str):
    """
    Evaluate the trained model on the held-out test set (10% of data).

    Args:
        model_path: Path to the trained model (e.g., "qwen3-4b-sft-3epochs-distributed")
    """
    import sys
    import os
    import json
    from datetime import datetime

    print(f"🧪 Evaluating trained model on test set: {model_path}")

    full_model_path = f"{OUTPUT_MODEL_PATH}/{model_path}"
    test_data_path = f"{OUTPUT_MODEL_PATH}/test_data.jsonl"  # Saved during training

    if not os.path.exists(full_model_path):
        raise ValueError(f"Trained model not found at: {full_model_path}")

    if not os.path.exists(test_data_path):
        raise ValueError(
            f"Test data not found at: {test_data_path}. Make sure you trained with the split dataset."
        )

    # Load test data
    test_examples = []
    with open(test_data_path, "r") as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line.strip()))

    print(f"📊 Loaded {len(test_examples)} test examples")

    # Set up the model for evaluation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("🔤 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(full_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        full_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Evaluate each test example
    results = []
    correct = 0

    for i, example in enumerate(test_examples):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(test_examples)}")

        # Create prompt
        prompt = example["prompt"]
        expected = example["completion"]

        try:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer(
                formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,  # Deterministic for evaluation
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode response
            predicted = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            ).strip()

            # Simple similarity check (you can make this more sophisticated)
            similarity = calculate_simple_similarity(predicted, expected)
            is_correct = similarity > 0.7

            if is_correct:
                correct += 1

            results.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": predicted,
                    "similarity": similarity,
                    "correct": is_correct,
                }
            )

        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
            results.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": f"Error: {str(e)}",
                    "similarity": 0.0,
                    "correct": False,
                }
            )

    # Calculate final metrics
    accuracy = correct / len(test_examples)
    avg_similarity = sum(r["similarity"] for r in results) / len(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_eval_{model_path.replace('/', '_')}_{timestamp}.json"
    results_path = f"{OUTPUT_MODEL_PATH}/{results_filename}"

    eval_results = {
        "model_path": model_path,
        "test_examples": len(test_examples),
        "correct_predictions": correct,
        "accuracy": accuracy,
        "average_similarity": avg_similarity,
        "detailed_results": results,
    }

    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # Commit results to volume
    trained_models_volume.commit()

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"📦 Model: {model_path}")
    print(f"📊 Test examples: {len(test_examples)}")
    print(f"✅ Correct predictions: {correct}")
    print(f"🎯 Test accuracy: {accuracy:.1%}")
    print(f"📈 Average similarity: {avg_similarity:.3f}")

    # Show some failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n❌ Example failures (showing first 3 of {len(failures)}):")
        for i, failure in enumerate(failures[:3]):
            print(f"  {i + 1}. Expected: {failure['expected'][:100]}...")
            print(f"     Predicted: {failure['predicted'][:100]}...")
            print(f"     Similarity: {failure['similarity']:.3f}")

    print(f"\n💾 Detailed results saved to: {results_filename}")

    return eval_results


def calculate_simple_similarity(predicted: str, expected: str) -> float:
    """Simple similarity calculation for evaluation."""
    if not predicted and not expected:
        return 1.0

    if not predicted or not expected:
        return 0.0

    # Try exact match first
    if predicted.strip() == expected.strip():
        return 1.0

    # Normalize whitespace and compare
    pred_normalized = " ".join(predicted.split())
    exp_normalized = " ".join(expected.split())

    if pred_normalized == exp_normalized:
        return 1.0

    # Character-level similarity as fallback
    pred_chars = set(pred_normalized.lower())
    exp_chars = set(exp_normalized.lower())

    if not pred_chars and not exp_chars:
        return 1.0

    intersection = len(pred_chars.intersection(exp_chars))
    union = len(pred_chars.union(exp_chars))

    return intersection / union if union > 0 else 0.0


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
def test_trained_model(model_path: str, test_prompts=None):
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


@app.function(
    image=training_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=60 * 60,  # 1 hour for comprehensive evaluation
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_trained_model_comprehensive(model_path: str, max_examples: int = 50):
    """
    Comprehensive evaluation of trained model using the existing eval.py system.

    Args:
        model_path: Path to the trained model (e.g., "qwen3-4b-sft-3epochs")
        max_examples: Maximum number of examples to evaluate
    """
    import sys
    import os
    from datetime import datetime

    # Add src to path to import eval modules
    sys.path.append("/root")

    print(f"🧪 Comprehensive evaluation of trained model: {model_path}")
    print(f"📊 Max examples: {max_examples}")

    full_model_path = f"{OUTPUT_MODEL_PATH}/{model_path}"

    if not os.path.exists(full_model_path):
        raise ValueError(f"Trained model not found at: {full_model_path}")

    # Set up the model for evaluation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("🔤 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(full_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        full_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Create a temporary model response function that uses our trained model
    def get_trained_model_response(prompt: str, temperature: float = 0.0) -> str:
        """Get response from the trained model."""
        try:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer(
                formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode response
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            ).strip()

            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    # Monkey patch the model response function in eval module
    # First, we need to create the eval modules in the container
    print("📝 Setting up evaluation environment...")

    # Create a simple eval script that uses our trained model
    eval_script = f'''
import json
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Any

# Import the trained model response function
get_model_response = None

def set_model_response_function(func):
    global get_model_response
    get_model_response = func

def extract_result_from_llm_output(output: str) -> str:
    """Extract the actual result from LLM output."""
    # Remove common prefixes and clean up
    lines = output.split('\\n')
    result_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip common LLM response prefixes
        if line.lower().startswith(('result:', 'output:', 'answer:', 'the result is')):
            line = line.split(':', 1)[-1].strip()
        result_lines.append(line)
    
    return '\\n'.join(result_lines).strip()

def calculate_similarity(predicted: str, expected: str, operation_type: str = "unknown") -> float:
    """Calculate similarity between predicted and expected results."""
    if not predicted and not expected:
        return 1.0
    
    if not predicted or not expected:
        return 0.0
    
    # Try exact match first
    if predicted.strip() == expected.strip():
        return 1.0
    
    # Normalize whitespace and compare
    pred_normalized = ' '.join(predicted.split())
    exp_normalized = ' '.join(expected.split())
    
    if pred_normalized == exp_normalized:
        return 1.0
    
    # Character-level similarity as fallback
    pred_chars = set(pred_normalized.lower())
    exp_chars = set(exp_normalized.lower())
    
    if not pred_chars and not exp_chars:
        return 1.0
    
    intersection = len(pred_chars.intersection(exp_chars))
    union = len(pred_chars.union(exp_chars))
    
    return intersection / union if union > 0 else 0.0

def evaluate_single_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single example using the trained model."""
    try:
        # Create prompt for the trained model
        prompt = f"""Given the filesystem state and operation, predict the result.

Initial filesystem state:
{{example['initial_state']}}

Operation: {{example['operation']}}

Expected result format: {{'filesystem state tree' if example['operation_type'] == 'state_change' else 'command output'}}

Result:"""
        
        predicted = get_model_response(prompt, temperature=0.0)
        predicted = extract_result_from_llm_output(predicted)
        
        expected = example['result']
        similarity = calculate_similarity(predicted, expected, example['operation_type'])
        correct = similarity > 0.7  # 70% similarity threshold
        
        return {{
            'correct': correct,
            'similarity': similarity,
            'predicted': predicted,
            'expected': expected,
            'operation': example['operation'],
            'operation_type': example['operation_type']
        }}
        
    except Exception as e:
        return {{
            'correct': False,
            'similarity': 0.0,
            'predicted': f"Error: {{str(e)}}",
            'expected': example['result'],
            'operation': example['operation'],
            'operation_type': example['operation_type'],
            'error': str(e)
        }}

def evaluate_dataset(examples: List[Dict[str, Any]], max_examples: int = 50) -> Dict[str, Any]:
    """Evaluate the model on a dataset."""
    if len(examples) > max_examples:
        examples = examples[:max_examples]
    
    print(f"Evaluating {{len(examples)}} examples...")
    
    results = []
    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"Progress: {{i + 1}}/{{len(examples)}}")
        
        result = evaluate_single_example(example)
        results.append(result)
    
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    avg_similarity = sum(r['similarity'] for r in results) / total if total > 0 else 0.0
    
    # Statistics by operation type
    by_type = defaultdict(list)
    for result in results:
        by_type[result['operation_type']].append(result)
    
    type_stats = {{}}
    for op_type, type_results in by_type.items():
        type_correct = sum(1 for r in type_results if r['correct'])
        type_avg_sim = sum(r['similarity'] for r in type_results) / len(type_results)
        type_stats[op_type] = {{
            'total': len(type_results),
            'correct': type_correct,
            'accuracy': type_correct / len(type_results),
            'avg_similarity': type_avg_sim
        }}
    
    return {{
        'total_examples': total,
        'correct_predictions': correct,
        'overall_accuracy': correct / total,
        'average_similarity': avg_similarity,
        'accuracy_by_type': type_stats,
        'detailed_results': results
    }}
'''

    # Write the eval script to a temporary file and execute it
    with open("/tmp/trained_model_eval.py", "w") as f:
        f.write(eval_script)

    # Import our custom eval module
    sys.path.insert(0, "/tmp")
    import trained_model_eval

    # Set our model response function
    trained_model_eval.set_model_response_function(get_trained_model_response)

    # Load the training data for evaluation
    print("📁 Loading evaluation dataset...")
    training_data_path = "/root/training_data.jsonl"

    if not os.path.exists(training_data_path):
        raise FileNotFoundError(
            "Training data not found. Please run prepare_training_data first."
        )

    # Load HuggingFace format data and convert back to structured format
    examples = []
    with open(training_data_path, "r") as f:
        for line in f:
            if line.strip():
                hf_example = json.loads(line.strip())

                # Parse the prompt to extract components
                prompt = hf_example["prompt"]
                completion = hf_example["completion"]

                # Simple parsing to extract initial_state and operation
                lines = prompt.split("\n")
                initial_state = ""
                operation = ""

                for i, line in enumerate(lines):
                    if "Current filesystem state:" in line:
                        # Find the state section
                        state_lines = []
                        for j in range(i + 1, len(lines)):
                            if lines[j].startswith("Operation:"):
                                operation = lines[j].replace("Operation:", "").strip()
                                break
                            state_lines.append(lines[j])
                        initial_state = "\n".join(state_lines).strip()
                        break

                # Determine operation type
                operation_type = "state_change"
                if any(
                    op in operation.lower()
                    for op in ["readdir", "ls", "stat", "getattr"]
                ):
                    operation_type = "query"

                examples.append(
                    {
                        "initial_state": initial_state,
                        "operation": operation,
                        "result": completion,
                        "operation_type": operation_type,
                    }
                )

    print(f"📊 Loaded {len(examples)} examples from training data")

    # Run evaluation
    results = trained_model_eval.evaluate_dataset(examples, max_examples)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"trained_model_comprehensive_eval_{model_path.replace('/', '_')}_{timestamp}.json"
    results_path = f"{OUTPUT_MODEL_PATH}/{results_filename}"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Commit results to volume
    trained_models_volume.commit()

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"📦 Model: {model_path}")
    print(f"📊 Total examples: {results['total_examples']}")
    print(f"✅ Correct predictions: {results['correct_predictions']}")
    print(f"🎯 Overall accuracy: {results['overall_accuracy']:.1%}")
    print(f"📈 Average similarity: {results['average_similarity']:.3f}")

    print(f"\n📋 Accuracy by operation type:")
    for op_type, stats in results["accuracy_by_type"].items():
        print(
            f"  {op_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']}) - Avg similarity: {stats['avg_similarity']:.3f}"
        )

    # Show some example failures
    failures = [r for r in results["detailed_results"] if not r["correct"]]
    if failures:
        print(f"\n❌ Example failures (showing first 3 of {len(failures)}):")
        for i, failure in enumerate(failures[:3]):
            print(f"  {i + 1}. Operation: {failure['operation']}")
            print(f"     Expected: {failure['expected'][:100]}...")
            print(f"     Predicted: {failure['predicted'][:100]}...")
            print(f"     Similarity: {failure['similarity']:.3f}")

    print(f"\n💾 Detailed results saved to: {results_filename}")

    return results


@app.function(
    image=training_image,
    volumes={
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=10 * 60,  # 10 minutes
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def create_test_split():
    """
    Manually create the 90/10 train/test split and save test data.
    Use this if you need to create the test split without retraining.
    """
    import json
    import random
    import os

    print("📊 Creating 90/10 train/test split...")

    # Load the full dataset
    training_data_path = "/root/training_data.jsonl"

    if not os.path.exists(training_data_path):
        raise FileNotFoundError(
            "Training data not found. The data should be baked into the image."
        )

    examples = []
    with open(training_data_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))

    print(f"✅ Loaded {len(examples)} total examples")

    # Split into train/test
    random.seed(42)  # Same seed as training
    random.shuffle(examples)

    train_split = 0.9
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    print(
        f"📊 Split: {len(train_examples)} train, {len(test_examples)} test ({train_split:.0%}/{1 - train_split:.0%})"
    )

    # Save test set
    test_data_path = f"{OUTPUT_MODEL_PATH}/test_data.jsonl"
    with open(test_data_path, "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")

    print(f"💾 Saved test set to {test_data_path}")

    # Also save train set for reference
    train_data_path = f"{OUTPUT_MODEL_PATH}/train_data.jsonl"
    with open(train_data_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    print(f"💾 Saved train set to {train_data_path}")

    # Commit to volume
    trained_models_volume.commit()

    print("✅ Train/test split created successfully!")

    return {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "test_data_path": test_data_path,
        "train_data_path": train_data_path,
    }


@app.function(
    image=training_image,
    volumes={
        OUTPUT_MODEL_PATH: trained_models_volume,
    },
    timeout=5 * 60,  # 5 minutes
)
def list_volume_contents():
    """List contents of the trained models volume for debugging."""
    import os

    print(f"📁 Contents of {OUTPUT_MODEL_PATH}:")

    if not os.path.exists(OUTPUT_MODEL_PATH):
        print(f"❌ Directory {OUTPUT_MODEL_PATH} does not exist!")
        return []

    contents = []
    for root, dirs, files in os.walk(OUTPUT_MODEL_PATH):
        level = root.replace(OUTPUT_MODEL_PATH, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size} bytes)")
            contents.append(file_path)

    return contents


@app.local_entrypoint()
def main():
    """Local entrypoint showing available training functions."""
    print("Available Modal training functions:")
    print(
        "  modal run modal_train.py::prepare_training_data                     # Upload training data"
    )
    print(
        "  modal run modal_train.py::train_qwen --model-name='qwen3-8b'        # Distributed training with 8x H100"
    )
    print(
        "  modal run modal_train.py::train_qwen --model-name='qwen3-8b' --use-wandb  # Train with W&B"
    )
    print(
        "  modal run modal_train.py::test_trained_model --model-path='qwen3-8b-sft-3epochs-distributed'  # Test model"
    )
    print(
        "  modal run modal_train.py::eval_trained_model_comprehensive --model-path='qwen3-8b-sft-3epochs-distributed'  # Full eval"
    )
    print(
        "  modal run modal_train.py::eval_trained_model_on_test_set --model-path='qwen3-8b-sft-3epochs-distributed'     # Test set eval"
    )

    print("\nDistributed training features:")
    print("  🚀 8x H100 GPUs with torchrun")
    print("  📊 Global batch size: per-device × 8 GPUs × 4 grad_accum")
    print("  💾 Gradient checkpointing for memory efficiency")
    print("  📈 W&B integration with distributed logging")

    print("\nTraining options:")
    print(
        "  --model-name: qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b"
    )
    print("  --num-epochs: Number of training epochs (default: 3)")
    print("  --batch-size: Per-device batch size (default: 2, global = 2×8×4 = 64)")
    print("  --learning-rate: Learning rate (default: 2e-5)")
    print("  --use-wandb: Enable W&B logging (default: True)")
    print("  --gradient-checkpointing: Enable gradient checkpointing (default: True)")


if __name__ == "__main__":
    main()
