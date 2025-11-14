import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch

from llmfuse.utils import STATE_STOP_TOKEN

from .postprocess import (
    sanitize_by_expected,
    is_state_expected,
    strip_state_stop_token,
)
from .metrics import exact_match, simple_similarity


MAX_CONTEXT_TOKENS = int(os.environ.get("LLMFUSE_MAX_CONTEXT", 8192))
MAX_GENERATED_TOKENS = int(os.environ.get("LLMFUSE_MAX_GENERATED", 4096))


def _detect_operation_line(prompt: str) -> str:
    """Return the first meaningful line describing the operation.

    Prompts are typically of the form:
      <W> or <R> on the first line, then an operation like 'readdir(...)' on the second.
    """
    if not prompt:
        return ""
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip the first tag-only line like <W> or <R>
        if stripped.startswith("<") and stripped.endswith(">") and len(stripped) <= 4:
            continue
        return stripped
    return ""


def build_prompt_with_contract(original_prompt: str) -> str:
    """Prepend a strict output contract to drive exact formatting.

    The contract varies by operation type (readdir/read vs tree-producing ops).
    """
    header = (
        "You are a pure function. Output exactly the required result and nothing else.\n"
        "No explanations, no code fences, no repeated outputs, no prefixes or suffixes.\n"
    )

    op_line = _detect_operation_line(original_prompt).lower()
    if "readdir(" in op_line:
        contract = (
            "Return only a JSON array of names with double quotes, on one line,\n"
            "formatted exactly like: [\".\", \"..\", \"name1\", \"name2\"].\n"
            "Use a single space after each comma. Output nothing else.\n"
            "Sort the names in lexicographic order (\".\", then \"..\", then others ascending)."
        )
    elif op_line.startswith("read(") or " read(" in op_line:
        contract = (
            "Return only the exact file content. If the file is empty, return an empty string\n"
            "(no characters). Output nothing else."
        )
    else:
        contract = (
            "Return exactly one <filesystem> XML document that represents the full state.\n"
            "It must start with <filesystem> and end with </filesystem> with no commentary,\n"
            f"and after emitting </filesystem> you must append the literal token {STATE_STOP_TOKEN} on its own line.\n"
            "Output nothing after that sentinel and do not repeat or summarize the tree."
        )

    return f"{header}{contract}\n\n{original_prompt}"


def evaluate_model_local(model_dir: str, dataset_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    # Load dataset
    with open(dataset_path, 'r') as f:
        examples_raw = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        examples_raw = examples_raw[:limit]

    prompts: List[str] = [ex["prompt"] for ex in examples_raw]
    expecteds: List[str] = [ex["completion"] for ex in examples_raw]

    results: List[Dict[str, Any]] = []
    num_exact = 0
    num_correct = 0

    # Prefer vLLM if available; otherwise fallback to transformers
    use_vllm = False
    try:
        import vllm  # noqa: F401
        use_vllm = True
    except Exception:
        use_vllm = False

    if use_vllm:
        from vllm import LLM, SamplingParams
        if torch.cuda.is_available():
            print(f"[EVAL] CUDA available: True, device_count={torch.cuda.device_count()}")
        else:
            print("[EVAL] CUDA available: False (running on CPU)")

        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            dtype="auto",
            max_model_len=MAX_CONTEXT_TOKENS,
            enforce_eager=False,
        )

        base_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_GENERATED_TOKENS,
            n=1,
            stop=[STATE_STOP_TOKEN],
        )

        try:
            batch_size = int(os.environ.get("VLLM_EVAL_BATCH_SIZE", 64))
        except Exception:
            batch_size = 64

        total = len(prompts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_prompts = prompts[start:end]
            batch_expecteds = expecteds[start:end]

            wrapped_prompts = [build_prompt_with_contract(p) for p in batch_prompts]

            # Use base greedy params (no aggressive stop sequences) to avoid truncation harming similarity
            outputs = llm.generate(wrapped_prompts, base_sampling_params)

            for i, output in enumerate(outputs):
                prompt = batch_prompts[i]
                expected = batch_expecteds[i]
                predicted = output.outputs[0].text if output.outputs else ""
                predicted = sanitize_by_expected(predicted, expected)
                expected_for_metrics = strip_state_stop_token(expected) if is_state_expected(expected) else expected

                sim = simple_similarity(predicted, expected_for_metrics)
                em = exact_match(predicted, expected_for_metrics)
                if sim > 0.7:
                    num_correct += 1
                if em:
                    num_exact += 1

                results.append({
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": predicted,
                    "similarity": sim,
                    "correct": sim > 0.7,
                    "exact": em,
                })

            done = min(end, total)
            if done % max(10, batch_size) == 0 or done == total:
                print(f"Progress: {done}/{total}")
    else:
        print("[EVAL] vLLM not available; falling back to transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        for idx, (prompt, expected) in enumerate(zip(prompts, expecteds)):
            wrapped = build_prompt_with_contract(prompt)
            inputs = tokenizer(
                wrapped,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT_TOKENS,
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_GENERATED_TOKENS,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the wrapped prompt portion; best-effort by slicing by input length
            predicted = decoded[len(wrapped):].strip()
            predicted = sanitize_by_expected(predicted, expected)
            expected_for_metrics = strip_state_stop_token(expected) if is_state_expected(expected) else expected

            sim = simple_similarity(predicted, expected_for_metrics)
            em = exact_match(predicted, expected_for_metrics)
            if sim > 0.7:
                num_correct += 1
            if em:
                num_exact += 1

            results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "similarity": sim,
                "correct": sim > 0.7,
                "exact": em,
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(prompts):
                print(f"Progress: {idx + 1}/{len(prompts)}")

    total_eval = len(results)
    acc = num_correct / total_eval if total_eval else 0.0
    exact_acc = num_exact / total_eval if total_eval else 0.0
    avg_sim = sum(r["similarity"] for r in results) / total_eval if total_eval else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "total_examples": total_eval,
        "accuracy": acc,
        "exact_accuracy": exact_acc,
        "average_similarity": avg_sim,
        "detailed_results": results,
        "timestamp": ts,
        "model_dir": model_dir,
        "dataset_path": dataset_path,
    }
    return out
