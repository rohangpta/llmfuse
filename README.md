# LLM-FUSE

LLM-FUSE turns filesystem operations into prompts. It mounts a FUSE filesystem,
sends each operation plus the current XML tree to a language model, and treats the
model response as the filesystem result.

Write-like operations such as `mkdir`, `write`, `rename`, and `unlink` must return
the complete next `<filesystem>` XML document. Read-like operations such as
`read` and `readdir` return only the requested content or listing. There is no
normal backing store; the live tree is the XML state the model last produced.

That makes state drift easy to see. File contents, directory structure,
permissions, and metadata either survive a sequence of mutations or they do not.

## How it works

1. `train/reference_fuse.py` runs a normal FUSE filesystem and logs real calls.
2. `train/generate_data.py` drives shell operations against that mount, snapshots
   the resulting tree, and writes prompt/completion JSONL.
3. State-changing examples use `<W>` prompts and expect a full XML tree followed
   by `<END_FS>`. Query examples use `<R>` prompts and expect the exact value.
4. `train/sft_modal.py` fine-tunes Qwen3-4B on those examples via Modal.
5. `infra/modal_llmfuse.py` serves the trained model as a `/generate` endpoint.
6. `llmfuse/llmfuse.py` mounts the LLM-backed filesystem and forwards operations
   to that endpoint.

## Repository layout

```text
common/       Qwen3-4B model loading and helpers
eval/         Evaluation runner, metrics, and postprocessing
infra/        Modal HTTP serving for LLM-FUSE and compression experiments
llmencode/    Separate compression playground; not part of the filesystem path
llmfuse/      FUSE implementation and XML filesystem state model
scripts/      Operational helpers such as dump_fs_xml and run_llmfuse
train/        Reference FUSE, data generation, and SFT scripts
```

## Setup

Use Python 3.11 or 3.12 for the project environment. FUSE data generation and
mounting need Linux; the Docker commands below assume a host with `/dev/fuse`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Modal training, evaluation, or serving:

```bash
pip install modal
modal setup
```

## Generate training data

Data generation runs in Docker because it needs FUSE privileges.

```bash
mkdir -p data/train

docker compose run --rm datagen python -m train.generate_data \
  --num_examples 1000 \
  --output_dir /app/data/train
```

Each generated row is a prompt/completion pair. To inspect the XML rendering for a
real directory:

```bash
python3 scripts/dump_fs_xml.py ./some-directory
```

## Train

The maintained model target is Qwen3-4B. Pass it explicitly; the trimmed training
pipeline rejects other model names.

```bash
DATASET=$(find data/train -type f -name 'fuse_*.jsonl' 2>/dev/null | sort | tail -1)
test -n "$DATASET" || { echo "Generate data first"; exit 1; }

modal run train/sft_modal.py::train_qwen \
  --model-name qwen3-4b \
  --training-data "$DATASET" \
  --num-epochs 8
```

Training downloads the base model with the Modal secret `huggingface-secret`. The
Modal function also declares `wandb-secret` because W&B logging is enabled by
default. The command above writes a Modal volume folder named
`qwen3-4b-sft-8epochs-distributed`.

## Evaluate

Use the model folder from training. The command below matches the 8-epoch example
above; replace the folder name if you trained with different settings.

```bash
DATASET=$(find data/train -type f -name 'fuse_*.jsonl' 2>/dev/null | sort | tail -1)
test -n "$DATASET" || { echo "Generate data first"; exit 1; }

modal run eval/modal_eval.py::eval_on_dataset \
  --model-path qwen3-4b-sft-8epochs-distributed \
  --dataset-path "$DATASET" \
  --max-examples 200
```

The evaluator reports exact match rate, average string similarity, and the share
of examples above the similarity threshold.

## Serve and mount

Deploy the trained model as a Modal endpoint, using the same model folder name:

```bash
modal run infra/modal_llmfuse.py::deploy \
  --model-path qwen3-4b-sft-8epochs-distributed
```

Set the endpoint URL printed by Modal, then start the local Dockerized FUSE mount:

```bash
export LLMFUSE_REMOTE_ENDPOINT="https://<modal-endpoint>/generate"
bash scripts/run_llmfuse.sh
```

The helper builds `Dockerfile.llmfuse`, mounts at `./mount`, and forwards
filesystem calls to the remote model. Try normal shell operations against the
mount:

```bash
mkdir -p mount/proj/docs
echo "hello world" > mount/proj/README.md
cat mount/proj/README.md
ls mount/proj
mv mount/proj/README.md mount/proj/notes.md
rm mount/proj/notes.md
```

## LLMEncode

`llmencode/` is a separate compression playground. It uses next-token
probabilities from a language model to drive arithmetic coding; it does not train
or serve the LLM-FUSE filesystem.

```bash
python3 -m llmencode.llmencode test "Hello world" --model qwen3-4b --verbose
```

## Limits

- Not a production filesystem. The FUSE surface is intentionally small and not
  POSIX-complete.
- Every state-changing operation rewrites the full XML tree, so prompt size grows
  with the number of files.
- File bodies are included only when they fit within configured size limits.
- Malformed model output, including bad XML or extra prose, fails the operation.
- Qwen3-4B is the only maintained model target in this repo.

## Tests

Run the local tests without loading model weights or mounting FUSE:

```bash
python3 - <<'PY'
import random
import pytest

random.seed(0)
raise SystemExit(pytest.main(["llmfuse/", "train/"]))
PY
```

End-to-end FUSE tests need a running model backend and Linux FUSE support. The
`fuse_integration` Docker Compose service is the path for that when configured.
