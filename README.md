## LLM-FUSE: A Filesystem for Stateful LLM Reasoning

This project trains a Large Language Model to act as a filesystem, testing whether LLMs can perform robust, stateful reasoning. A mountable filesystem (via FUSE) serves as a controlled testbed. The core idea, "Full State Rewrite," tasks the LLM with applying a pure state transition: on each operation (e.g., `mkdir /new_dir`), the model receives the entire filesystem state (State_t) and must output the complete new state (State_{t+1}).

**Current Performance (Oct 2025):** The latest `qwen3-4b-sft-8epochs-distributed` checkpoint trained on a 10 000-sample targeted dataset reaches **98 % accuracy**, **68 % exact match**, and **0.977 average similarity** on the focused 100-example eval split (`data/eval/fuse_100_1760993121.jsonl`). A broader 200-example regression set sits at 87 % / 54 % with the same model. The remaining 32/100 near misses are dominated by literal-copy drift (e.g. READs paraphrasing headings or TRUNCATE ops swapping `[TRACE]` for `[INFO]` even though byte lengths match), plus two real failures: one READ collapsing a five-line log to `[ERROR]` and one READDIR hallucinating an extra entry.

> **For AI Agents:** See [`AGENTS.md`](./AGENTS.md) for a comprehensive guide to the training pipeline, data format, and current state of the project.

## Quickstart (Docker)

Most workflows run inside Docker for FUSE support and consistent environments.

- **Prerequisites**: Docker

### Generate training data

**Recommended (docker-compose):**

```bash
# Generate 10k high-quality examples with targeted coverage (~5 hours)
docker-compose run --rm datagen python3 -m train.generate_data \
  --num_examples 10000 \
  --output_dir /app/data/train \
  --targeted_prob 0.3
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
  python -m train.generate_data --num_examples 10000 --output_dir /app/data/train --targeted_prob 0.3
```

**Key parameters:**
- `--num_examples`: Number of examples to generate (current best: 10 000)
- `--targeted_prob`: Fraction of edge-case prompts (current best: 0.3 for read/readdir heavy mix)
- `--output_dir`: Output directory inside container

**Data quality features:**
- **Deterministic content**: Same filepath always generates identical content (enables exact match learning)
- **Complete operations**: Write operations include full data, all operations reference existing files
- **Diverse coverage**: 12 operation types with balanced distribution
- **Edge case targeting**: Optional targeted examples for empty filesystems, nested directories

Output files are JSONL with deterministic naming: `fuse_{num_samples}_{unixtime}.jsonl` in `data/train/`.

### Evaluate a model on a dataset

Evaluation is now Modal-first. Use the pre-generated 100-example stress test for quick regression checks:

```bash
modal run eval/modal_eval.py::eval_on_dataset \
  --model-path "qwen3-4b-sft-8epochs-distributed" \
  --dataset-path "data/eval/fuse_100_1760993121.jsonl" \
  --max-examples 100
```

To download the JSON metrics afterwards:

```bash
modal run eval/modal_eval.py::download_eval_results_file \
  --filename "custom_eval_qwen3-4b-sft-8epochs-distributed_<timestamp>.json" \
  --local-dir ./eval_results
```

Modal keeps results under `/root/output_models`; the helper above copies them locally. The evaluator enforces a 1024-token generation cap and reports accuracy, exact match, and average similarity.

## Running the FUSE filesystem

The mountable filesystem is still work-in-progress in this trimmed build. The `llmfuse` package is preserved so the integration points remain stable, but the runtime entrypoint raises `NotImplementedError` until the Qwen-backed backend is wired in. Data generation, training, and evaluation remain fully supported.

## Local development (optional)

If you prefer local installs for non-FUSE tasks (e.g., editing, unit tests):

```bash
python -m pip install -r requirements.txt
```

You can also use `uv` if you like, but Docker is recommended for anything requiring FUSE.

> **Heads-up:** Both `llmfuse` and `llmencode` are currently placeholders kept for API compatibility. They raise `NotImplementedError` until the online inference path is reintroduced.

## Training on Modal

Train the Qwen3-4B pipeline directly on Modal (8× H100 GPUs recommended):

```bash
# Fine-tune Qwen3-4B for 8 epochs on the 10k targeted corpus
modal run train/sft_modal.py::train_qwen \
  --model-name "qwen3-4b" \
  --training-data "data/train/fuse_10000_1760983124.jsonl" \
  --num-epochs 8 \
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

**Highlights (Oct 2025 refresh):**
1. **Deterministic content**: File bodies are keyed off the path hash, so every `/config/foo.json` is a byte-for-byte copy across samples—critical for exact match.
2. **Complete operation payloads**: Writes, truncates, and renames always include the full post-op state.
3. **Targeted sampling**: `--targeted_prob 0.3` oversamples read/readdir scenarios, nested mkdir chains, and empty-tree edges that previously tripped the model.
4. **Operation coverage (10 000-sample run):**
   - `mkdir`: 1 758 • `write`: 1 652 • `readdir`: 1 137 • `read`: 880
   - `create`: 836 • `unlink`: 751 • `rmdir`: 715 • `chown`: 585 • `chmod`: 573

**Quality snapshots (10 k targeted set):**
- ✅ 880 read completions match byte length expectations (85 are intentional zero-byte reads).
- ✅ 1 137 readdir outputs parse as JSON arrays and mirror the on-tree entries exactly.
- ✅ No tree-formatting regressions detected in the generated completions.

**Command-line flags:**
- `--num_examples`: Number of examples to emit (use 10 000 for the current best run).
- `--targeted_prob`: Edge-case sampling rate (0.3 matches the latest training/eval runs).
- `--output_dir`: Output directory (default: `/app/data/train` inside Docker).

**Output format:** JSONL files with `prompt` and `completion` fields, using `<W>`/`<R>` markers.

## Evaluation

Modal hosts the evaluation pipeline. The primary entrypoints are:

- `eval/modal_eval.py::eval_on_dataset` – run inference with vLLM (1024-token cap, greedy decoding).
- `eval/modal_eval.py::download_eval_results_file` – pull the JSON report back to disk.

Highlights:
- Enforces the same contracts used for supervised fine-tuning, preventing format drift.
- Reports accuracy (similarity > 0.7), exact match, average similarity, and per-sample breakdowns.
- Works out-of-the-box with the curated 100-sample regression dataset or any JSONL produced by `train/generate_data.py`.

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
- **Model:** `qwen3-4b-sft-3epochs-distributed` (base: `Qwen/Qwen2.5-Coder-3B-Instruct`)
- **Training Data:** 5,000 deterministic examples with logged write contents
- **Accuracy:** 75% (similarity > 0.7)
- **Exact Match:** 64.5%
- **Average Similarity:** 0.854

### Recent Improvements (Oct 19, 2025)
**Major data quality fixes:**
- ✅ Deterministic content generation (filepath-based hashing)
- ✅ Write operations include full data parameter
- ✅ All operations reference existing files only
- ✅ Removed compound commands creating inline files

**Observed impact (Oct 19, 2025):**
- Accuracy: 75% (similarity > 0.7)
- **Exact Match:** 64.5% (up from 52% baseline)
- Average similarity: 0.854

**Next training opportunities:**
- Scale beyond 5,000 examples (e.g., 10K+) to push exact match higher
- File: `data/train/fuse_5000_1760902673.jsonl` (latest dataset)
- Ready for additional training runs or larger models

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
