import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .postprocess import sanitize_by_expected
from .metrics import exact_match, simple_similarity


def evaluate_model_local(model_dir: str, dataset_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    with open(dataset_path, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        examples = examples[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results: List[Dict[str, Any]] = []
    num_exact = 0
    num_correct = 0

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        expected = ex["completion"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = decoded[len(prompt):].strip()
        predicted = sanitize_by_expected(predicted, expected)

        sim = simple_similarity(predicted, expected)
        em = exact_match(predicted, expected)
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

    total = len(examples)
    acc = num_correct / total if total else 0.0
    exact_acc = num_exact / total if total else 0.0
    avg_sim = sum(r["similarity"] for r in results) / total if total else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "total_examples": total,
        "accuracy": acc,
        "exact_accuracy": exact_acc,
        "average_similarity": avg_sim,
        "detailed_results": results,
        "timestamp": ts,
        "model_dir": model_dir,
        "dataset_path": dataset_path,
    }
    return out 