import os
import json
from datetime import datetime
import modal

# Support both package and script execution
try:
    from .runner import evaluate_model_local
except Exception:
    from eval.runner import evaluate_model_local

# Modal app and volumes
app = modal.App("llmfuse-eval")

MODEL_CACHE = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
TRAINED_MODELS = modal.Volume.from_name("qwen3-trained-models", create_if_missing=True)

MODEL_CACHE_PATH = "/root/models"
OUTPUT_MODEL_PATH = "/root/output_models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.24.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.4.3",
        "vllm>=0.6.2",
    ])
    .workdir("/root")
    .add_local_dir("eval", "/root/eval")
    .add_local_dir("data", "/root/data")
)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: MODEL_CACHE,
        OUTPUT_MODEL_PATH: TRAINED_MODELS,
    },
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_on_dataset(model_path: str, dataset_path: str, max_examples: int | None = None):
    """Evaluate a trained model stored in Modal volume on a dataset in the repo.

    Args:
      model_path: e.g. "qwen3-4b-sft-1epochs-distributed" (folder under /root/output_models)
      dataset_path: e.g. "data/train/fuse_100_1754254154.jsonl" (mounted under /root)
    """
    # Resolve paths
    full_model_dir = model_path
    if not model_path.startswith("/"):
        full_model_dir = f"{OUTPUT_MODEL_PATH}/{model_path}"
    full_dataset_path = dataset_path
    if not dataset_path.startswith("/"):
        full_dataset_path = f"/root/{dataset_path}"

    if not os.path.exists(full_model_dir):
        raise FileNotFoundError(f"Model not found at {full_model_dir}")
    if not os.path.exists(full_dataset_path):
        raise FileNotFoundError(f"Dataset not found at {full_dataset_path}")

    print(f"🧪 Evaluating {full_model_dir} on {full_dataset_path}")
    results = evaluate_model_local(full_model_dir, full_dataset_path, limit=max_examples)

    ts = results.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"custom_eval_{os.path.basename(full_model_dir)}_{ts}.json"
    out_path = f"{OUTPUT_MODEL_PATH}/{out_name}"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    TRAINED_MODELS.commit()

    print("\n" + "=" * 60)
    print("🎯 DATASET EVAL RESULTS")
    print("=" * 60)
    print(f"📦 Model: {model_path}")
    print(f"🗂 Dataset: {dataset_path}")
    print(f"📊 Examples: {results['total_examples']}")
    print(f"✅ Accuracy (>0.7 sim): {results['accuracy']:.1%}")
    print(f"🎯 Exact match: {results['exact_accuracy']:.1%}")
    print(f"📈 Avg similarity: {results['average_similarity']:.3f}")
    print(f"💾 Saved to: {out_name}")

    return {"results_file": out_name, **{k: results[k] for k in ("accuracy", "exact_accuracy", "average_similarity")}}


@app.function(
    image=image,
    volumes={OUTPUT_MODEL_PATH: TRAINED_MODELS},
    timeout=5 * 60,
)
def _read_results_file(filename: str) -> bytes:
    path = f"{OUTPUT_MODEL_PATH}/{filename}"
    if not os.path.exists(path):
        available = []
        if os.path.exists(OUTPUT_MODEL_PATH):
            available = os.listdir(OUTPUT_MODEL_PATH)
        raise FileNotFoundError(f"{filename} not found. Available: {available}")
    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def download_eval_results_file(filename: str, local_dir: str = "./eval_results"):
    os.makedirs(local_dir, exist_ok=True)
    data = _read_results_file.remote(filename)
    local_path = os.path.join(local_dir, filename)
    with open(local_path, "wb") as f:
        f.write(data)
    print(f"✅ Saved to {local_path}")
    return local_path
