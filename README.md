# LLM-FUSE

A research experiment testing whether language models can perform robust, stateful reasoning by acting as a filesystem.

## Overview

This project trains an LLM to function as a complete filesystem implementation. A mountable FUSE filesystem serves as the testbed. The core approach is "full state rewrite": on each operation (e.g., `mkdir /foo`), the model receives the entire filesystem state and must output the complete new state.

This tests a fundamental question: can LLMs maintain consistent state across many sequential operations without drift or hallucination?

## How It Works

1. **Training data generation**: A reference FUSE filesystem logs real operations and their resulting states
2. **Supervised fine-tuning**: The model learns to predict state transitions from (state, operation) pairs
3. **Inference**: The trained model backs a real FUSE mount, handling all filesystem operations

State is represented as XML:

```xml
<filesystem>
  <directory path="/" name="/" mode="755" owner="root" group="root" mtime="...">
    <file path="hello.txt" name="hello.txt" mode="644" size="12">
      <body>hello world</body>
    </file>
  </directory>
</filesystem>
```

## Repository Structure

```
llmfuse/       # FUSE filesystem backed by LLM inference
llmencode/     # Arithmetic coding compression using LLM probabilities
train/         # Data generation and fine-tuning scripts
eval/          # Evaluation pipelines
infra/         # Modal deployment for GPU inference
common/        # Shared model utilities
scripts/       # Helper scripts
```

## Usage

### Generate Training Data

```bash
docker-compose run --rm datagen python3 -m train.generate_data \
  --num_examples 10000 \
  --output_dir /app/data/train
```

### Train on Modal

```bash
modal run train/sft_modal.py::train_qwen \
  --model-name "qwen3-4b" \
  --training-data "path/to/data.jsonl" \
  --num-epochs 8
```

### Evaluate

```bash
modal run eval/modal_eval.py::eval_on_dataset \
  --model-path "your-model-checkpoint" \
  --dataset-path "path/to/eval.jsonl"
```

### Run the Filesystem

Deploy the inference service:

```bash
modal run infra/modal_llmfuse.py::deploy --model-path your-model-checkpoint
```

Mount the filesystem (requires Linux/Docker for FUSE):

```bash
export LLMFUSE_REMOTE_ENDPOINT="https://your-modal-endpoint/generate"
bash scripts/run_llmfuse.sh
```

## LLMEncode

The `llmencode` package demonstrates prediction-compression equivalence using arithmetic coding guided by model probabilities. This explores the theoretical connection between language modeling and data compression.

```python
from llmencode import LLMEncode
encoder = LLMEncode(model_name="qwen3-4b")
result = encoder.test_roundtrip("Hello world", verbose=True)
```

## Requirements

- Docker (for FUSE operations and consistent environments)
- Modal account (for GPU training and inference)
- Python 3.11+ with dependencies in `requirements.txt`

## Related Work

This project draws inspiration from work on LLM reasoning, state tracking, and the theoretical connections between prediction and compression.
