import argparse
import json
import os
from .runner import evaluate_model_local


def main():
    parser = argparse.ArgumentParser(description="Evaluate a local model folder on a JSONL dataset")
    parser.add_argument("--model-path", required=True, help="Path to model directory (folder with safetensors)")
    parser.add_argument("--dataset-path", required=True, help="Path to JSONL dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-to", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_model_local(args.model_path, args.dataset_path, limit=args.limit)

    print(json.dumps({
        "total_examples": results["total_examples"],
        "accuracy": results["accuracy"],
        "exact_accuracy": results["exact_accuracy"],
        "average_similarity": results["average_similarity"],
    }, indent=2))

    out_path = args.save_to or f"eval_results_{results['timestamp']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main() 