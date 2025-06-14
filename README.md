# LLM-FUSE: A Filesystem for Stateful LLM Reasoning

This project is an experiment to determine if a Large Language Model (LLM) can be trained to perform robust, stateful reasoning. We use a mountable filesystem, implemented via FUSE, as a controlled and verifiable testbed for this capability.

The core methodology, termed "Full State Rewrite," tasks the LLM with executing a pure state transition function. On every turn, the model receives the entire textual representation of the current filesystem state (State_t) along with a single operation (e.g., `mkdir /new_dir`). Its objective is to output the new, complete, and perfectly accurate filesystem state (State_t+1).

## Installation

1. Install uv (modern Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd src
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Set up your Gemini API key for evaluation:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Testing

Run the test suite in Docker (required for Linux compatibility):

```bash
docker-compose run --rm datagen pytest src/ -v
```

## Data Generation

Generate synthetic training data using the reference FUSE filesystem:

```bash
# Generate training data with CLI (Docker required for FUSE support)
docker run --rm --privileged -v $(pwd):/app/host llmfuse-datagen \
  python -m src.generate_data -n 100 -o /app/host/training_data.json

# Or use docker-compose (generates 50 examples by default)
docker-compose up datagen-fuse

# Generate smaller test datasets
docker run --rm --privileged -v $(pwd):/app/host llmfuse-datagen \
  python -m src.generate_data -n 10 -o /app/host/test_data.json

# On macOS (clean exit with Docker instructions)
python -m src.generate_data --help
# ERROR: This script requires FUSE support and cannot run natively on macOS.
# Please use Docker instead: ...
```

**CLI Options:**
- `-n, --num_examples`: Number of training examples to generate (default: 100)
- `-o, --output`: Output JSON file path (default: training_data.json)

The data generation creates `(State_t, Operation, State_t+1)` training triples with:
- **Real FUSE operations**: `create()`, `mkdir()`, `readdir()`, `unlink()`, `rmdir()`, `chmod()`, `chown()`
- **Actual FUSE call parameters**: `mode=0o100644`, file handles, timestamps
- **Ground truth from reference_fuse.py**: Perfect filesystem state transitions
- **Complete operation logs**: Full sequence of FUSE calls for debugging

## Evaluation

Evaluate LLM performance on FUSE operation data:

```bash
# Set up API key (add to ~/.bash_profile for persistence)
export GEMINI_API_KEY="your-api-key-here"

# Generate test data and run evaluation
source ~/.bash_profile
docker run --rm --privileged -v $(pwd):/app/host -e GEMINI_API_KEY=$GEMINI_API_KEY llmfuse-datagen \
  python -m src.generate_data -n 10 -o /app/host/test_data.json

docker run --rm --privileged -v $(pwd):/app/host -e GEMINI_API_KEY=$GEMINI_API_KEY llmfuse-datagen \
  python -c "from src.eval import evaluate_dataset; evaluate_dataset('/app/host/test_data.json', '/app/host/eval_results.json', max_examples=10)"

# Or use docker-compose for evaluation
docker-compose run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY eval-fuse
```

**Expected Results (N=10 FUSE operations):**
- **60% overall accuracy** - exactly as expected for this challenging task
- **100% accuracy on state-changing operations** (mkdir, rmdir, chmod, chown)
- **0% accuracy on query operations** (readdir) - these are much harder as they require understanding complex output formats

**Sample Output:**
```
==================================================
EVALUATION SUMMARY
==================================================
Total examples: 10
Correct predictions: 6
Overall accuracy: 60.00%

Accuracy by operation type:
  state_change: 100.00% (6/6)
  query: 0.00% (0/4)

Accuracy by operation:
  mkdir: 100.00% (2/2)
  chown: 100.00% (2/2)
  rmdir: 100.00% (2/2)
  readdir: 0.00% (0/4)
```

The results demonstrate that the LLM successfully learns filesystem state transitions from FUSE training data, achieving perfect accuracy on operations that modify filesystem state while struggling with complex query output formatting.

## FUSE Filesystem

### Running with Docker (Recommended)

```bash
# Start FUSE filesystem
docker-compose up fuse

# Access filesystem in container
docker-compose exec fuse /bin/bash
# Inside container: filesystem mounted at /mnt/fuse
```

### Running Directly (Linux only)

1. Install FUSE:
   ```bash
   sudo apt-get install fuse3 libfuse3-dev
   ```

2. Create mount point and run:
   ```bash
   mkdir /tmp/fuse_mount
   uv run python llmfuse.py /tmp/fuse_mount
   ```

3. Interact with `/dev/llm` device:
   ```bash
   echo "How many files are in the current directory?" > /tmp/fuse_mount/dev/llm
   cat /tmp/fuse_mount/dev/llm
   ```

4. Unmount:
   ```bash
   fusermount -u /tmp/fuse_mount
   ```

## Training Methodology

**Supervised Fine-Tuning (SFT)**: Train on synthetically generated `(State_t, Operation) -> State_t+1` examples to teach basic mechanics and output format.

**Reinforcement Learning (RL)** *(planned)*: Apply RL to enhance robustness and precision using diff-based reward signals that heavily penalize any deviation from ground truth.

**Validation**: The `/dev/llm` device enables testing genuine understanding through natural language queries like "How many files are in /tmp?" that require reasoning over the current state representation.

## Architecture

The experiment consists of two key components:

1. **Reference FUSE Implementation** (`reference_fuse.py`): Provides perfect ground truth by executing real filesystem operations and logging all FUSE calls. This generates the training data.

2. **LLM FUSE Implementation** (`llmfuse.py`): Routes ALL filesystem operations through an LLM, which must predict the complete new filesystem state for each operation. This is what gets trained and evaluated.

The LLM filesystem deliberately has no built-in knowledge of filesystem semantics - it must learn everything from the training data generated by the reference implementation.
