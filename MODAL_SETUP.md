# Modal Setup Instructions

This guide explains how to set up and run the Qwen3 model testing on Modal's cloud infrastructure.

## Prerequisites

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal token new
   ```

## Optional: Gemini API Setup

If you want to compare Gemini with Qwen3 models, set up the Gemini API secret:

1. **Get a Gemini API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Add the secret to Modal**:
   ```bash
   modal secret create gemini-api-key GEMINI_API_KEY=your_api_key_here
   ```

   Replace `your_api_key_here` with your actual Gemini API key.

## Running the Tests

### Main Test Suite (Recommended)
```bash
modal run modal_run.py::test_qwen3
```
This runs a comprehensive test comparing different Qwen3 model sizes.

### Individual Test Functions

**Test different model sizes:**
```bash
modal run modal_run.py::test_qwen3_sizes
```

**Test a single model:**
```bash
modal run modal_run.py::test_qwen3_single --model-name="qwen3-8b"
```

**Compare Gemini vs Qwen3:**
```bash
modal run modal_run.py::compare_gemini_qwen3
```

## What Each Test Does

### `test_qwen3` (Main Function)
- Lists available models
- Tests multiple Qwen3 sizes (0.6B, 1.7B, 4B, 8B) on filesystem-related prompts
- Provides a comprehensive comparison and summary
- **Runtime**: ~30-45 minutes
- **Cost**: ~$5-10 depending on model usage

### `test_qwen3_sizes`
- Focuses on comparing different model sizes
- Tests 3 filesystem-related prompts across 4 model sizes
- **Runtime**: ~20-30 minutes
- **Cost**: ~$3-7

### `test_qwen3_single`
- Tests a single model with 4 detailed filesystem prompts
- Good for focused testing of a specific model size
- **Runtime**: ~10-15 minutes
- **Cost**: ~$1-3

### `compare_gemini_qwen3`
- Side-by-side comparison of Gemini and Qwen3-8B
- Requires Gemini API key
- **Runtime**: ~5-10 minutes
- **Cost**: ~$1-2 plus Gemini API costs

## Hardware Configuration

The Modal functions are configured with:
- **GPU**: A100 40GB (optimal for models up to Qwen3-8B)
- **Timeout**: 1-3 hours depending on function
- **Image**: Debian slim with Python 3.11 + ML dependencies

## Cost Estimation

Modal pricing (approximate):
- A100 40GB: ~$4-6/hour
- Most tests complete in 30-60 minutes
- Total cost per full test run: ~$2-10

## Troubleshooting

### Common Issues

1. **"Secret not found" error**: 
   - The Gemini API secret is optional
   - Qwen3 tests will run without it

2. **GPU memory errors**:
   - Try smaller models first (qwen3-0.6b, qwen3-1.7b)
   - The A100 40GB should handle up to qwen3-8b comfortably

3. **Timeout errors**:
   - Tests are configured with generous timeouts
   - If you see timeouts, try testing fewer models at once

4. **Import errors**:
   - These are expected in local linting
   - The app runs correctly on Modal

### Model Size Recommendations

**For experimentation:**
- Start with `qwen3-0.6b` and `qwen3-1.7b`
- Fast and cheap to test

**For quality evaluation:**
- Use `qwen3-4b` and `qwen3-8b`
- Good balance of performance and cost

**For production comparison:**
- Test `qwen3-8b` as the recommended size
- Compare against your current Gemini usage

## Monitoring

You can monitor your Modal runs at:
- [Modal Dashboard](https://modal.com/apps)
- View logs, costs, and performance metrics in real-time

## Customization

To modify the tests:

1. **Change models tested**: Edit the `models_to_test` arrays in each function
2. **Add custom prompts**: Modify the `test_prompts` arrays
3. **Adjust parameters**: Change `temperature`, `max_output_tokens`, etc.
4. **Change GPU**: Modify the `gpu=modal.gpu.A100(size="40GB")` parameter

## Example Output

The tests will show:
- Model loading progress
- Response comparisons
- Performance summaries
- Success/failure status for each model

Example:
```
🚀 COMPREHENSIVE QWEN3 TESTING ON MODAL
===============================================================================

📋 AVAILABLE MODELS:
Available Qwen3 models:
  qwen3-0.6b           -> Qwen/Qwen3-0.6B
  qwen3-1.7b           -> Qwen/Qwen3-1.7B
  ...

🧪 RUNNING MODEL SIZE COMPARISON...
===============================================================================

📝 TESTING PROMPT: What happens when I run 'chmod 755 script.sh'?
--------------------------------------------------------------------------------

🤖 Testing qwen3-0.6b...
✅ qwen3-0.6b: The command `chmod 755 script.sh` changes the file permissions...

🤖 Testing qwen3-1.7b...
✅ qwen3-1.7b: This command sets the file permissions for `script.sh` to 755...
``` 