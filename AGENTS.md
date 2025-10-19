# LLM-FUSE: Agent-Friendly Guide

## What This Project Does

**LLM-FUSE** is a research project that trains a Large Language Model to act as a filesystem. Instead of traditional data structures, the LLM maintains the entire filesystem state in its context and predicts how that state changes after each operation.

### The Core Concept: "Full State Rewrite"

The model learns a pure state transition function:
```
State_t + Operation → State_{t+1}
```

For example:
- **Input:** Current filesystem tree + `mkdir('/docs')`
- **Output:** Updated filesystem tree with new `/docs` directory

This tests whether an LLM can perform **stateful, multi-step reasoning** with perfect consistency.

---

## Current State (October 2025)

### Performance
- **Model:** `qwen3-4b` fine-tuned for 3 epochs
- **Training Data:** 2,000 diverse filesystem operation examples
- **Test Accuracy:** **78%** (similarity > 0.7 threshold)
- **Average Similarity:** 0.866
- **Exact Match:** 52%

### What Works Well (>90% accuracy)
- Simple file operations (create, delete, chmod, chown)
- Directory operations (mkdir, rmdir)
- File reads with correct path resolution
- Permission changes
- Most readdir listings

### Known Failure Modes (causing the 22% gap)
1. **Nested directory creation** - Deep `mkdir -p` style operations
2. **Read operation confusion** - Occasionally reads wrong file from tree
3. **Empty filesystem edge cases** - Rare training data leakage
4. **Complex symlink operations** - File content tracking after symlink creation

---

## Project Structure

```
llmfuse/
├── train/
│   ├── generate_data.py      # Generates training data via reference FUSE
│   ├── reference_fuse.py     # Ground truth filesystem (logs operations)
│   ├── sft_cloud.py          # Core training script (runs on Modal)
│   └── sft_modal.py          # Modal orchestration (distributed training)
├── eval/
│   ├── runner.py             # Evaluation logic with prompt contracts
│   ├── modal_eval.py         # Modal-based evaluation functions
│   └── postprocess.py        # Results analysis
├── llmfuse/
│   ├── llmfuse.py           # LLM-backed FUSE filesystem
│   └── fs_state.py          # Filesystem state representation
├── data/
│   └── train/               # Generated JSONL datasets
└── common/
    └── model.py             # Shared model utilities
```

---

## Training Pipeline

### 1. Data Generation
Uses a **reference FUSE filesystem** to generate synthetic examples:
- Mounts a real loopback filesystem
- Executes random operations (mkdir, write, chmod, etc.)
- Logs: `(initial_state, operation) → final_state`
- Produces JSONL format with prompt/completion pairs

**Key features:**
- Highly randomized file content (code, text, structured data, logs)
- Diverse operation types (50+ different FUSE operations)
- Realistic permission modes and timestamps

### 2. Prompt Contract System
Each prompt is wrapped with strict output contracts to enforce format:

```python
# For tree-modifying operations (mkdir, write, etc.):
"You are a pure function. Output exactly the required result and nothing else.
Return exactly one filesystem tree in the format shown below."

# For readdir operations:
"Return only a JSON array of names: [\".\", \"..\", \"file1\", \"file2\"]"

# For read operations:
"Return only the exact file content without any prefixes or explanations."
```

This contract system is applied **consistently** during both training and evaluation.

### 3. Distributed Training (Modal)
- **Platform:** Modal Labs (8x H100 GPUs)
- **Framework:** Hugging Face TRL (`SFTTrainer`)
- **Distribution:** `torchrun` with FDDP (Fully Sharded Data Parallel)
- **Model:** `Qwen/Qwen2.5-Coder-3B-Instruct` (4B params)

Training script: `train/sft_cloud.py` (executed inside Modal)

### 4. Evaluation
Three evaluation methods:

1. **Local:** `eval/runner.py` - Quick testing on local machine
2. **Modal (test set):** Evaluate on the 10% held-out test split
3. **Modal (custom dataset):** `eval/modal_eval.py::eval_on_dataset`

All evaluations use the **same prompt contracts** as training.

**Metrics:**
- **Accuracy:** % of predictions with similarity > 0.7
- **Exact Match:** % of character-perfect predictions
- **Average Similarity:** Mean similarity across all samples

---

## Data Format

Training data is JSONL with this structure:

```json
{
  "prompt": "<W>\nmkdir('/docs', mode=0o755)\n---\n/                dir  755 root:root Oct 19 00:54\n└── test.txt                 file 644 root:root Oct 19 00:54 0 B\n        │",
  "completion": "/                dir  755 root:root Oct 19 00:54\n├── docs                     dir  755 root:root Oct 19 00:54\n└── test.txt                 file 644 root:root Oct 19 00:54 0 B\n        │"
}
```

- `<W>` marker: Write operation (tree-modifying)
- `<R>` marker: Read operation (query)
- `---` separator: Divides operation from current state
- Tree format: ASCII tree with permissions, ownership, timestamps

---

## Key Files for Agents

### Training
- `train/generate_data.py` - Data generation entry point
- `train/sft_cloud.py` - Lines 193-222: `build_prompt_with_contract()` function
- `train/sft_modal.py` - Lines 179-294: `train_qwen()` function

### Evaluation  
- `eval/runner.py` - `evaluate_model_local()` and `build_prompt_with_contract()`
- `eval/modal_eval.py` - Modal evaluation functions
- `eval/postprocess.py` - Results analysis

### Filesystem Logic
- `llmfuse/fs_state.py` - State representation
- `train/reference_fuse.py` - Ground truth implementation

---

## Common Operations

### Generate Training Data
```bash
# Using Docker (recommended)
docker-compose run --rm datagen python -m train.generate_data -n 2000

# Direct (requires FUSE support)
python -m train.generate_data --num_examples 2000 --output_dir ./data/train
```

### Train Model on Modal
```bash
modal run train/sft_modal.py::train_qwen \
  --model-path "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --training-data "data/train/fuse_2000_*.jsonl" \
  --num-epochs 3 \
  --batch-size 4
```

### Evaluate Trained Model
```bash
# Evaluate on test set (stored in Modal volume)
modal run eval/modal_eval.py::eval_on_dataset \
  --model-path "qwen3-4b-sft-3epochs-distributed" \
  --dataset-path "data/train/test_set_200.jsonl" \
  --max-examples 50
```

### Download Evaluation Results
```bash
modal run eval/modal_eval.py::download_eval_results_file \
  --filename "custom_eval_qwen3-4b-sft-3epochs-distributed_20251019_032839.json"
```

---

## Technical Details

### Why This Is Hard
1. **Long context:** Filesystem trees can be hundreds of lines
2. **Precise formatting:** Small errors break the tree structure
3. **State consistency:** Must track all file metadata (size, perms, timestamps)
4. **Complex operations:** Symlinks, nested directories, file renames
5. **No partial credit:** Unlike classification, small errors compound

### Training Challenges Solved
- **Data quality:** Sanitized NUL bytes, ensured valid UTF-8
- **Content diversity:** Custom randomizer generates varied file content
- **Format consistency:** Strict prompt contracts enforced everywhere
- **Distributed training:** Handled test set generation bug (overwriting in multi-GPU)
- **Memory constraints:** Switched from 8B to 4B model to fit in H100 memory

### Path to 100% Accuracy
Based on failure analysis, reaching 100% would require:
1. **More training data** (5K-10K samples vs current 2K)
2. **More training epochs** (5-10 vs current 3)
3. **Targeted examples** for edge cases:
   - Nested directory creation (`mkdir -p a/b/c/d`)
   - Empty filesystem operations
   - Complex symlink chains
4. **Larger model** (8B or 14B params if GPU memory allows)
5. **Post-processing** to catch duplicate outputs or format errors

---

## Quick Start for Agents

**Goal:** Train and evaluate your own LLM-FUSE model

1. **Generate data:**
   ```bash
   docker-compose run --rm datagen python -m train.generate_data -n 2000
   ```

2. **Analyze dataset:**
   ```bash
   python -m eval.postprocess data/train/fuse_2000_*.jsonl
   ```

3. **Train on Modal:**
   ```bash
   modal run train/sft_modal.py::train_qwen \
     --training-data "data/train/fuse_2000_*.jsonl"
   ```

4. **Evaluate:**
   ```bash
   modal run eval/modal_eval.py::eval_on_dataset \
     --model-path "qwen3-4b-sft-3epochs-distributed" \
     --dataset-path "data/train/test_set_200.jsonl" \
     --max-examples 50
   ```

5. **Download results:**
   ```bash
   modal run eval/modal_eval.py::download_eval_results_file \
     --filename "custom_eval_<model>_<timestamp>.json"
   ```

---

## Research Questions

This project explores:
1. **Can LLMs maintain perfect state consistency?** (Currently: 78% yes)
2. **How much training data is needed?** (2K → 78%, need 5K-10K for 95%+)
3. **Do prompt contracts improve accuracy?** (Yes, dramatically - 50% → 78%)
4. **What's the context limit?** (Works well up to ~500 lines of filesystem tree)
5. **Can this scale to real filesystems?** (Maybe - but need hierarchical approaches)

---

## Dependencies

- **Python 3.11+**
- **PyTorch 2.1+**
- **Transformers 4.36+**
- **Modal** (for distributed training)
- **FUSE** (Linux only, for data generation)
- **Docker** (recommended for consistent environments)

See `requirements.txt` for complete list.

---

## Contact & Contributing

This is a research project. Key areas for contribution:
- Improving training data diversity
- Optimizing prompt contracts for edge cases
- Scaling to larger filesystems
- Alternative model architectures

See the main `README.md` for setup instructions.
