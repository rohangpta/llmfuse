## LLM-FUSE: A Filesystem for Stateful LLM Reasoning

This project trains a Large Language Model to act as a filesystem, testing whether LLMs can perform robust, stateful reasoning. A mountable filesystem (via FUSE) serves as a controlled testbed. The core idea, "Full State Rewrite," tasks the LLM with applying a pure state transition: on each operation (e.g., `mkdir /new_dir`), the model receives the entire filesystem state (State_t) and must output the complete new state (State_{t+1}).

**Current Performance:** A fine-tuned `qwen3-4b` model achieves **78% accuracy** on filesystem operations after training on 2,000 diverse examples.

> **For AI Agents:** See [`AGENTS.md`](./AGENTS.md) for a comprehensive guide to the training pipeline, data format, and current state of the project.

## Quickstart (Docker)

Most workflows run inside Docker for FUSE support and consistent environments.

- **Prerequisites**: Docker

### Generate training data

**Recommended (docker-compose):**

```bash
# Generate 5000 high-quality examples (takes 2-3 hours)
docker-compose run --rm datagen python3 -m train.generate_data \
  -n 5000 \
  --output_dir /app/data/train \
  --targeted_prob 0.05
```

**Alternative (docker run):**

```bash
# Build the image
docker build -f Dockerfile.datagen -t llmfuse:latest .

# Generate examples (writes JSONL into ./data/train)
docker run --rm \
  --privileged \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --security-opt apparmor:unconfined \
  -e PYTHONUNBUFFERED=1 \
  -v "$(pwd)/data:/app/data" \
  llmfuse:latest \
  python -m train.generate_data --num_examples 5000 --output_dir /app/data/train --targeted_prob 0.05
```

**Key parameters:**
- `-n / --num_examples`: Number of examples to generate (recommended: 5000+)
- `--targeted_prob`: Fraction of examples targeting edge cases (default: 0.25, recommended: 0.05)
- `--output_dir`: Output directory inside container

**Data quality features:**
- **Deterministic content**: Same filepath always generates identical content (enables exact match learning)
- **Complete operations**: Write operations include full data, all operations reference existing files
- **Diverse coverage**: 12 operation types with balanced distribution
- **Edge case targeting**: Optional targeted examples for empty filesystems, nested directories

Output files are JSONL with deterministic naming: `fuse_{num_samples}_{unixtime}.jsonl` in `data/train/`.

### Evaluate a model on a dataset

Set an API key if evaluating Gemini models:

```bash
export GEMINI_API_KEY="your-api-key"
```

- Local evaluation (outside Docker):

```bash
python -m eval.cli --model-path ./models/qwen3-0.6b-test \
  --dataset-path data/train/fuse_200_1730000000.jsonl \
  --limit 50 \
  --save-to eval_results/local_eval.json
```

- Docker Compose evaluation (mounts your `models/` and `data/`):

```bash
# Build services
docker-compose build

# Run eval (override env with your paths)
MODEL_PATH=./models/qwen3-0.6b-test \
DATASET_PATH=./data/train/fuse_200_1730000000.jsonl \
LIMIT=50 \
SAVE_TO=./data/eval/local_eval.json \
docker-compose run --rm eval
```

- Modal evaluation (evaluate a trained model saved in a Modal volume on a dataset in the repo):

```bash
modal run eval/modal_eval.py::eval_on_dataset \
  --model-path "qwen3-4b-sft-1epochs-distributed" \
  --dataset-path "data/train/fuse_200_1730000000.jsonl" \
  --max-examples 50
```

Notes:
- Results are saved under Modal volume `/root/output_models` (use `eval/modal_eval.py::download_eval_results_file` to download),
  and local eval saves to `eval_results/` by default when you pass `--save-to`.

## Running the FUSE filesystem (Linux only)

Running the LLM-backed filesystem itself requires Linux with FUSE:

```bash
sudo apt-get update && sudo apt-get install -y fuse3 libfuse3-dev
python -m pip install -r requirements.txt

mkdir -p /tmp/fuse_mount
python llmfuse/llmfuse.py /tmp/fuse_mount

# Unmount when done
fusermount -u /tmp/fuse_mount
```

- macOS: run data generation and evaluation in Docker. Mounting the FUSE filesystem directly on macOS is not supported in this repo.

## Local development (optional)

If you prefer local installs for non-FUSE tasks (e.g., editing, unit tests):

```bash
python -m pip install -r requirements.txt
```

You can also use `uv` if you like, but Docker is recommended for anything requiring FUSE.

## Training on Modal

Train models on Modal Labs with distributed GPU support:

```bash
# Train a model (8x H100 GPUs)
modal run train/sft_modal.py::train_qwen \
  --model-path "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --training-data "data/train/fuse_2000_*.jsonl" \
  --num-epochs 3 \
  --batch-size 4
```

The training script uses:
- Hugging Face TRL's `SFTTrainer`
- Fully Sharded Data Parallel (FSDP) via `torchrun`
- Automatic 90/10 train/test split
- Strict output contracts for consistent formatting

Results are saved to Modal volume `/root/output_models/`.

## Data generation details

**Source files:**
- `train/generate_data.py` - Main generation script with deterministic content
- `train/reference_fuse.py` - Ground truth loopback FUSE filesystem that logs operations

**Generation methodology:**
Produces training triples: `(initial_state, operation) → result` where:
- **State-changing ops** (`<W>`): Complete filesystem tree (State T → State T+1)
- **Query ops** (`<R>`): Specific response (directory listing, file content, attributes)

**Key improvements (Oct 2025):**
1. **Deterministic content**: Files generate identical content based on filepath hash
   - Enables model to learn exact content copying
   - Same `/config.py` always has same content across all examples
2. **Complete operation info**: Write operations include full `data` parameter
3. **Valid references**: All operations only reference files that exist in initial state
4. **Balanced distribution**: 12 operation types with ~8-15% each

**Quality metrics (5K sample dataset):**
- ✅ 100% write operations include data parameter
- ✅ 94% rename operations reference valid source files
- ✅ 5.9% empty filesystem operations (edge case coverage)
- ✅ 3.2% deeply nested mkdir operations
- ✅ 1.1% error responses (realistic failure scenarios)

**Command-line flags:**
- `-n / --num_examples`: Number of examples (default: 100, recommended: 5000+)
- `--output_dir`: Output directory (default: `/app/data/train`)
- `--targeted_prob`: Edge case probability (default: 0.25, recommended: 0.05)

**Output format:** JSONL files with `prompt` and `completion` fields, using `<W>`/`<R>` markers.

## Evaluation

Robust evaluators with format-aware similarity:

- Local CLI (see above): `eval/cli.py`
- Modal functions: `eval/modal_eval.py::eval_on_dataset`, `eval/modal_eval.py::download_eval_results_file`

Highlights:
- Handles mixed completion formats (filesystem trees, stat-like outputs, directory listings)
- Uses same strict prompt contracts as training for consistency
- Similarity-based scoring (threshold: 0.7) with exact match tracking
- Results saved as JSON with detailed per-example analysis

## Architecture

- **Reference FUSE (ground truth)**: `train/reference_fuse.py`
  - Loopback to a real directory, logs every FUSE call used to build datasets.
- **Data Generation**: `train/generate_data.py`
  - Generates diverse training examples with randomized file content (code, text, configs, logs).
- **Training**: `train/sft_cloud.py` and `train/sft_modal.py`
  - Distributed fine-tuning on Modal Labs using strict prompt contracts.
- **LLM-backed FUSE**: `llmfuse/llmfuse.py`
  - Maintains state as a text tree via `llmfuse/fs_state.py` and prompts an LLM to compute the next full state for each operation.
- **Evaluation**: `eval/runner.py` (local) and `eval/modal_eval.py` (Modal)
  - Applies same prompt contracts as training for accurate evaluation.
- **Utilities**: `llmfuse/utils.py` (tree formatting constants, helpers)

## Compression (LLM-guided arithmetic coding)

The `llmencode` package demonstrates prediction–compression equivalence using arithmetic coding guided by model probabilities.

CLI examples:

```bash
# Roundtrip test (recommended):
python -m llmencode.llmencode test "Hello world" --model qwen3-4b

# Encode to hex:
python -m llmencode.llmencode encode "Some text" --model qwen3-4b --output-format hex
```

Notes:
- The standalone `decode` subcommand requires encoding metadata and will exit with an error message. Use the `test` subcommand for an end-to-end roundtrip in one process, or use the Python API to persist metadata.

Minimal Python API roundtrip:

```python
from llmencode import LLMEncode
encoder = LLMEncode(model_name="qwen3-4b")
stats = encoder.test_roundtrip("Hello world", verbose=True)
```

## Repository layout

- `llmfuse/`: LLM-driven filesystem and state representation
- `train/`: Reference FUSE and data generation
- `eval/`: Evaluation pipelines (local and Modal support)
- `llmencode/`: Arithmetic coding + LLM-guided compression
- `common/`: Shared model utilities
- `data/`: Generated datasets and evaluation outputs

## Requirements

Core dependencies are listed in `requirements.txt` (notably `fusepy`, `torch`, `transformers`). Gemini-based evaluation requires `google-generativeai` and a `GEMINI_API_KEY`.

## Current Results & Future Work

### Performance (October 2025)
- **Model:** `qwen3-4b` (Qwen2.5-Coder-3B-Instruct fine-tuned)
- **Training Data (old):** 2,000 examples with random content
- **Accuracy:** 78% (similarity > 0.7)
- **Exact Match:** 52%
- **Average Similarity:** 0.866

### Recent Improvements (Oct 19, 2025)
**Major data quality fixes:**
- ✅ Deterministic content generation (filepath-based hashing)
- ✅ Write operations include full data parameter
- ✅ All operations reference existing files only
- ✅ Removed compound commands creating inline files

**Expected impact:**
- Accuracy: ~78-80% (structural understanding already good)
- **Exact Match: 75-95%** (up from 52% - model can now copy content correctly)
- Gap reduction: From 26pp to <10pp

**New training data ready:**
- 5,000 high-quality examples with deterministic content
- File: `data/train/fuse_5000_1760902673.jsonl`
- Ready for retraining to validate improvements

### Path to 95%+ Accuracy
1. ✅ Fix data quality issues (deterministic content, complete operations)
2. 🔄 Retrain on 5K examples with fixes
3. 📊 Validate exact match improvement
4. Consider scaling to 10K examples if needed
5. Experiment with larger models (8B-14B params)

See [`AGENTS.md`](./AGENTS.md) for detailed analysis and research directions.

## Troubleshooting

- **FUSE errors (macOS/Windows)**: Use Docker for data generation/evaluation. The live FUSE filesystem is supported on Linux only.
- **Modal training/eval**: Ensure you have Modal credentials configured (`modal token set --token-id YOUR_ID --token-secret YOUR_SECRET`).
- **Modal eval results**: Download files via `eval/modal_eval.py::download_eval_results_file --filename <name>`.
- **Decode CLI**: Use `test` for roundtrip; standalone `decode` needs metadata from the same session.

