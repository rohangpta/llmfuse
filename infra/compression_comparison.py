"""
Compare compression ratios between base Qwen3-4B and fine-tuned model on XML filesystem data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("compression-comparison")

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_PACKAGE_ROOT = "/root/llmfuse_pkg"

MODEL_CACHE = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
TRAINED_MODELS = modal.Volume.from_name("qwen3-trained-models", create_if_missing=True)

MODEL_CACHE_PATH = "/root/models"
OUTPUT_MODEL_PATH = "/root/output_models"

BASE_MODEL = "Qwen/Qwen3-4B"
FINETUNED_MODEL_NAME = "qwen3-4b-sft-8epochs-distributed"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "accelerate>=0.24.0",
            "tokenizers>=0.15.0",
            "numpy",
        ]
    )
    .add_local_dir(str(REPO_ROOT / "common"), f"{REMOTE_PACKAGE_ROOT}/common", copy=True)
    .add_local_dir(str(REPO_ROOT / "llmfuse"), f"{REMOTE_PACKAGE_ROOT}/llmfuse", copy=True)
    .add_local_dir(str(REPO_ROOT / "llmencode"), f"{REMOTE_PACKAGE_ROOT}/llmencode", copy=True)
    .env({"PYTHONPATH": REMOTE_PACKAGE_ROOT, "HF_HOME": MODEL_CACHE_PATH})
    .workdir("/root")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 30,
    volumes={
        MODEL_CACHE_PATH: MODEL_CACHE,
        OUTPUT_MODEL_PATH: TRAINED_MODELS,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def compare_compression(test_texts: List[str]) -> Dict[str, Any]:
    """
    Compare compression between base Qwen3-4B and fine-tuned model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np

    from llmencode.arithmetic_coding import ArithmeticCoder

    results = []

    # Load tokenizer (same for both models - fine-tuned uses base tokenizer)
    print(f"Loading tokenizer from: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load base model
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda().eval()

    # Load fine-tuned model
    finetuned_path = f"{OUTPUT_MODEL_PATH}/{FINETUNED_MODEL_NAME}"
    print(f"Loading fine-tuned model: {finetuned_path}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda().eval()

    def get_probs(model, tokenizer, context: str) -> np.ndarray:
        """Get next-token probability distribution."""
        input_ids = tokenizer.encode(context, return_tensors="pt").cuda()
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        prob_floor = 1e-15
        probs = np.maximum(probs, prob_floor)
        probs /= probs.sum()
        return probs

    def compress_with_model(text: str, model, model_name: str) -> Dict[str, Any]:
        """Compress text using arithmetic coding with model probabilities."""
        input_ids = tokenizer.encode(text, add_special_tokens=False)

        probs_per_token = []
        start_token = "<START>"

        for i, token_id in enumerate(input_ids):
            if i == 0:
                context = start_token
            else:
                context = start_token + tokenizer.decode(input_ids[:i], skip_special_tokens=True)
            probs = get_probs(model, tokenizer, context)
            probs_per_token.append(probs.tolist())

        coder = ArithmeticCoder()
        compressed_bytes = coder.encode(input_ids, probs_per_token)

        return {
            "model": model_name,
            "original_bytes": len(text.encode("utf-8")),
            "compressed_bytes": len(compressed_bytes),
            "num_tokens": len(input_ids),
            "compression_ratio": len(text.encode("utf-8")) / len(compressed_bytes) if compressed_bytes else 0,
        }

    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {len(text)} chars")
        print(f"{'='*60}")

        # Compress with base model
        print(f"Compressing with base model...")
        base_result = compress_with_model(text, base_model, "Qwen/Qwen3-4B")
        print(f"  Base: {base_result['original_bytes']} -> {base_result['compressed_bytes']} bytes ({base_result['compression_ratio']:.2f}x)")

        # Compress with fine-tuned model
        print(f"Compressing with fine-tuned model...")
        ft_result = compress_with_model(text, ft_model, FINETUNED_MODEL_NAME)
        print(f"  Fine-tuned: {ft_result['original_bytes']} -> {ft_result['compressed_bytes']} bytes ({ft_result['compression_ratio']:.2f}x)")

        improvement = (base_result['compressed_bytes'] - ft_result['compressed_bytes']) / base_result['compressed_bytes'] * 100 if base_result['compressed_bytes'] else 0
        print(f"  Improvement: {improvement:.1f}% smaller with fine-tuned model")

        results.append({
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "text_length": len(text),
            "base": base_result,
            "finetuned": ft_result,
            "improvement_pct": improvement,
        })

    return {"results": results}


@app.local_entrypoint()
def main():
    """Run compression comparison on sample XML filesystem data."""

    test_texts = [
        # Sample 1: The XML from the user
        '''<filesystem><directory path="/" name="/" mode="755" owner="root" group="root" mtime="2025-01-01T00:00:00"><directory path="testdir" name="testdir" mode="755" owner="root" group="root" mtime="2025-01-01T00:00:00" /><file path="testfile.txt" name="testfile.txt" mode="644" owner="root" group="root" mtime="2025-01-01T00:00:01" size="14"><body>hello llmfuse
</body></file></directory></filesystem>''',

        # Sample 2: Larger filesystem tree
        '''<filesystem><directory path="/" name="/" mode="755" owner="root" group="root" mtime="2025-01-01T00:00:00"><directory path="etc" name="etc" mode="755" owner="root" group="root" mtime="2025-01-01T00:00:00"><file path="etc/passwd" name="passwd" mode="644" owner="root" group="root" mtime="2025-01-01T00:00:01" size="42"><body>root:x:0:0:root:/root:/bin/bash
</body></file><file path="etc/hosts" name="hosts" mode="644" owner="root" group="root" mtime="2025-01-01T00:00:01" size="27"><body>127.0.0.1 localhost
</body></file></directory><directory path="var" name="var" mode="755" owner="root" group="root" mtime="2025-01-01T00:00:00"><directory path="var/log" name="log" mode="755" owner="root" group="root" mtime="2025-01-01T00:00:00" /></directory></directory></filesystem>''',

        # Sample 3: Plain English text (control - should favor base model)
        '''The quick brown fox jumps over the lazy dog. This is a simple English sentence that should be well-predicted by the base model since it was trained on general text data.''',
    ]

    print("Running compression comparison on Modal...")
    print(f"Test texts: {len(test_texts)}")

    result = compare_compression.remote(test_texts)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for i, r in enumerate(result["results"]):
        print(f"\nTest {i+1}: {r['text_preview'][:60]}...")
        print(f"  Original: {r['text_length']} bytes")
        print(f"  Base Qwen3-4B:    {r['base']['compressed_bytes']} bytes ({r['base']['compression_ratio']:.2f}x)")
        print(f"  Fine-tuned:       {r['finetuned']['compressed_bytes']} bytes ({r['finetuned']['compression_ratio']:.2f}x)")
        print(f"  Improvement:      {r['improvement_pct']:.1f}%")
