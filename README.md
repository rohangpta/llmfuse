## LLM-FUSE: A Filesystem for Stateful LLM Reasoning

This project tests whether a Large Language Model (LLM) can perform robust, stateful reasoning. A mountable filesystem (via FUSE) serves as a controlled testbed. The core idea, "Full State Rewrite," tasks the LLM with applying a pure state transition: on each operation (e.g., `mkdir /new_dir`), the model receives the entire filesystem state (State_t) and must output the complete new state (State_{t+1}).

## Quickstart (Docker)

Most workflows run inside Docker for FUSE support and consistent environments.

- **Prerequisites**: Docker and Docker Compose.

### Generate training data

```bash
# Build and run the FUSE-enabled data generator (writes to ./data/train)
docker-compose up --build datagen

# Or override defaults (examples count and output dir)
docker-compose run --rm --privileged \
  -e PYTHONUNBUFFERED=1 \
  datagen python -m train.generate_data --num_examples 200 --output_dir /app/data/train
```

Output files are JSONL with deterministic naming: `fuse_{num_samples}_{unixtime}.jsonl` in `data/train/`.

### Evaluate a model on a dataset

Set an API key if evaluating Gemini models:

```bash
export GEMINI_API_KEY="your-api-key"
```

Run the robust evaluator (points to a dataset generated above):

```bash
docker-compose run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY \
  eval python eval/eval_main.py --data /app/data/train/fuse_200_1730000000.jsonl \
  --model gemini --max-examples 50
```

Notes:
- The compose service `eval` uses the same image as `datagen` and mounts project code. Pass `--data` explicitly.
- Results are saved under `/app/data/eval` by default (mapped to `./data/eval`).

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

## Data generation details

- Source: `train/generate_data.py`
- Reference FUSE (ground truth): `train/reference_fuse.py` (loopback FS that logs operations)
- Produces triples: `(initial_state, operation) -> result` where result is either the complete next filesystem tree (for state-changing ops) or a query response (e.g., directory listing, getattr fields).

Key flags:
- `--num_examples` (`-n`): number of examples to generate (default: 100)
- `--output_dir`: output directory inside the container (default: `/app/data/train`)

## Evaluation

Robust evaluator with format-aware similarity:

```bash
python eval/eval_main.py --data data/train/fuse_100_1730000000.jsonl \
  --model gemini --max-examples 50
```

Highlights:
- Handles mixed completion formats (filesystem trees, stat-like outputs, directory listings)
- More tolerant similarity for format differences
- Saves results to `data/eval/` by default (when run via Docker compose)

Typical small-sample results (10 examples):
- 60% overall accuracy
- 100% on state-changing operations (mkdir, rmdir, chmod, chown)
- Lower accuracy on query-heavy tasks (e.g., readdir) due to formatting difficulty

## Architecture

- **Reference FUSE (ground truth)**: `train/reference_fuse.py`
  - Loopback to a real directory, logs every FUSE call used to build datasets.
- **LLM-backed FUSE**: `llmfuse/llmfuse.py`
  - Maintains state as a text tree via `llmfuse/fs_state.py` and prompts an LLM to compute the next full state for each operation.
- **Utilities**: `llmfuse/utils.py` (tree formatting constants, helpers)
- **Evaluation**: `eval/eval_main.py` and helpers under `eval/`

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

## Troubleshooting

- **FUSE errors (macOS/Windows)**: Use Docker for data generation/evaluation. The live FUSE filesystem is supported on Linux only.
- **Evaluation errors about `--data`**: Ensure you pass `--data` to the evaluator path inside the container (`/app/data/...`).
- **Decode CLI**: Use `test` for roundtrip; standalone `decode` needs metadata from the same session.

