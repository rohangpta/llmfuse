# Model Interface Usage Guide

The updated `model.py` interface now supports both **Gemini** and **Hugging Face** models, with special support for all Qwen3 model sizes.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Hugging Face models, ensure you have sufficient GPU memory
# or use smaller models like qwen3-0.6b or qwen3-1.7b
```

### Basic Usage

```python
from src.model import get_model_response

# Use Gemini (default)
response = get_model_response("What is a filesystem?")

# Use Qwen3-8B
response = get_model_response("What is a filesystem?", model_name="qwen3-8b")

# Use any Hugging Face model
response = get_model_response("What is a filesystem?", model_name="Qwen/Qwen3-32B")
```

## Available Qwen3 Models

The interface provides shortcuts for all Qwen3 model sizes:

| Shortcut Name | Full Hugging Face Name | Model Size | Memory Req. |
|---------------|------------------------|------------|-------------|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.6B params | ~1.5GB |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 1.7B params | ~3.5GB |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 4B params | ~8GB |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8B params | ~16GB |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 14B params | ~28GB |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 32B params | ~64GB |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B` | 30B (MoE) | ~20GB |
| `qwen3-235b-a22b` | `Qwen/Qwen3-235B-A22B` | 235B (MoE) | ~80GB |

**Base models** are also available by adding `-base` suffix (e.g., `qwen3-8b-base`).

## Model Provider Detection

The interface automatically detects which provider to use:

- **Gemini**: Model names starting with `models/gemini` or `gemini`
- **Hugging Face**: 
  - Qwen3 shortcuts (`qwen3-8b`, etc.)
  - Full HF names with `/` (e.g., `microsoft/DialoGPT-medium`)
  - Any other model name not matching Gemini patterns

## Advanced Usage

### Compare Model Sizes

```python
from src.model import get_model_response

prompt = "Explain filesystem permissions in Linux"

# Test different Qwen3 sizes
models = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]

for model in models:
    print(f"\n=== {model.upper()} ===")
    response = get_model_response(
        prompt=prompt,
        model_name=model,
        temperature=0.1,
        max_output_tokens=300
    )
    print(response)
```

### Qwen3 Thinking Mode

Qwen3 models support a "thinking mode" for complex reasoning:

```python
# Enable thinking mode (shows reasoning process)
response = get_model_response(
    "Solve this complex filesystem problem: ...",
    model_name="qwen3-8b"
    # The HF implementation sets enable_thinking=False by default
    # Modify the code to enable thinking if needed
)
```

### Custom Generation Parameters

```python
response = get_model_response(
    prompt="Creative filesystem metaphor",
    model_name="qwen3-8b",
    temperature=0.8,  # More creative
    max_output_tokens=500,
    # Additional HF parameters
    top_k=50,
    repetition_penalty=1.1
)
```

## Testing Script

Run the included test script to compare different models:

```bash
python test_qwen3_models.py
```

This will:
1. Show all available models
2. Test different Qwen3 sizes on filesystem-related prompts
3. Compare Gemini vs Qwen3 responses (if Gemini API key is configured)

## Environment Variables

- `GEMINI_API_KEY`: Required for Gemini models
- No additional setup needed for Hugging Face models (models download automatically)

## Memory Considerations

- **Small models** (0.6B-1.7B): Can run on most GPUs or even CPU
- **Medium models** (4B-8B): Require dedicated GPU with 8-16GB VRAM
- **Large models** (14B+): Require high-end GPUs or multiple GPUs
- **MoE models**: More efficient than equivalent dense models

## Error Handling

The interface provides clear error messages:

- Missing dependencies: "Hugging Face transformers not installed..."
- Insufficient memory: Model loading will fail with memory error
- API issues: Detailed error messages for debugging

## Tips for Best Performance

1. **Start small**: Begin with `qwen3-0.6b` or `qwen3-1.7b` for testing
2. **Monitor memory**: Use `nvidia-smi` to check GPU memory usage
3. **Use quantized models**: Consider GGUF or AWQ versions for production
4. **Batch processing**: Process multiple prompts together when possible

## Example Applications

### Filesystem Operation Prediction

```python
# Predict what happens with filesystem commands
commands = [
    "mkdir -p /tmp/test/deep/path",
    "chmod 755 script.sh",
    "ln -s /path/to/file symlink"
]

for cmd in commands:
    prediction = get_model_response(
        f"What does this command do? {cmd}",
        model_name="qwen3-4b"
    )
    print(f"Command: {cmd}")
    print(f"Prediction: {prediction}\n")
```

### Model Performance Comparison

```python
# Compare different models on the same task
prompt = "Explain the difference between hard links and soft links"

models = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
results = {}

for model in models:
    results[model] = get_model_response(prompt, model_name=model)
    
# Analyze results...
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `transformers>=4.51.0` is installed
2. **CUDA errors**: Check GPU memory and CUDA compatibility
3. **Model download failures**: Check internet connection and disk space
4. **Slow inference**: Consider using smaller models or quantized versions

### Performance Optimization

- Use `torch.compile()` for faster inference (requires PyTorch 2.0+)
- Enable `device_map="auto"` for multi-GPU setups
- Consider using `accelerate` for advanced distributed inference 