"""
Common utilities and shared functionality for model evaluation.
Contains shared constants, Modal setup, and helper functions.
"""

import modal
import os
import json
import aiohttp
from typing import Dict, Any, List, Optional, Union

# Define the Modal app
app = modal.App("qwen3-vllm-testing")

# Create Modal Volumes for caching and results
model_cache = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen3-vllm-cache", create_if_missing=True)
eval_results_volume = modal.Volume.from_name(
    "qwen3-eval-results", create_if_missing=True
)

MODEL_CACHE_PATH = "/root/models"
VLLM_CACHE_PATH = "/root/.cache/vllm"
EVAL_RESULTS_PATH = "/root/eval_results"

# Create the vLLM image with all dependencies
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "vllm==0.9.1",
            "google-generativeai",
            "huggingface_hub[hf_transfer]==0.32.0",
            "flashinfer-python==0.2.6.post1",
            "openai>=1.0.0",  # For client
            "aiohttp>=3.8.0",  # For async requests
        ],
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .apt_install(["git"])
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",  # Use V1 engine for better performance
        }
    )
    .add_local_file("data/training_data_1000.jsonl", "/root/data/training_data_100.jsonl")
    .add_local_file("data/training_data_1000.jsonl", "/root/data/training_data_1000.jsonl")
)

# Model configuration constants
DEFAULT_MODEL = "models/gemini-2.5-flash-preview-05-20"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 512
API_KEY_ENV_VAR = "GEMINI_API_KEY"
VLLM_PORT = 8000
MINUTES = 60  # seconds

# Safety settings for unrestricted content generation
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
]

# Qwen3 model sizes mapping
QWEN3_MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}


def _is_huggingface_model(model_name: str) -> bool:
    """Determine if the model is a Hugging Face model based on naming patterns."""
    if model_name.lower() in QWEN3_MODELS:
        return True
    if "/" in model_name and not model_name.startswith("models/"):
        return True
    if model_name.startswith("models/gemini") or model_name.startswith("gemini"):
        return False
    return False


async def _get_vllm_response(
    server_url: str,
    prompt: str,
    model_name: str,
    temperature: float = 0.1,  # Lower temperature for more deterministic output
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    **kwargs,
) -> str:
    """Get response from vLLM server via OpenAI-compatible API."""

    # Convert simplified model names to full HF model names for display
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    messages = [{"role": "user", "content": prompt}]

    payload = {
        # "model": full_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "stream": False,
        # Add parameters to discourage reasoning tokens
        "top_p": 0.9,  # Slightly more focused sampling
        "frequency_penalty": 0.1,  # Discourage repetitive patterns
        "presence_penalty": 0.1,   # Encourage concise responses
        # Removed stop tokens as they were preventing any output
        **kwargs,
    }

    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return (
                        f"ERROR: vLLM server error (status {resp.status}): {error_text}"
                    )

                response_json = await resp.json()

                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"].strip()
                else:
                    return "ERROR: No response from vLLM server"

    except Exception as e:
        return f"ERROR: Failed to connect to vLLM server: {str(e)}"


def _get_gemini_response(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Get response from a Gemini model."""
    try:
        import google.generativeai as genai
    except ImportError:
        return "ERROR: google-generativeai not installed"

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"Environment variable {API_KEY_ENV_VAR} is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )

    try:
        response = model.generate_content(
            contents=prompt,
            generation_config=config,
            safety_settings=SAFETY_SETTINGS,
        )

        if not response.parts:
            error_msg = (
                "Model response was blocked (safety filters or other restrictions)"
            )
            print(f"ERROR: {error_msg}")
            return f"ERROR: {error_msg}"

        return response.text

    except Exception as e:
        error_msg = f"API call failed: {type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"ERROR: {error_msg}"


def validate_api_key() -> bool:
    """Validate that the required API key is available."""
    return os.environ.get(API_KEY_ENV_VAR) is not None


def get_available_qwen3_models() -> List[str]:
    """Get list of available Qwen3 model sizes."""
    return list(QWEN3_MODELS.keys())


def print_available_models() -> None:
    """Print available models and usage examples."""
    print("\n🤖 Available Qwen3 Models:")
    for model_key, model_path in QWEN3_MODELS.items():
        print(f"  {model_key:12} -> {model_path}")

    print("\n🚀 Usage Examples:")
    print("  modal run modal_eval.py::test_qwen3                        # Test all sizes")
    print("  modal run modal_eval.py::test_qwen3_single --model-name='qwen3-8b'  # Test single size")
    print("  modal run modal_eval.py::compare_gemini_qwen3              # Compare with Gemini")
    print("  modal run modal_eval.py::serve_qwen3             # Default (qwen3-8b)")
    print("  modal run modal_eval.py::serve_qwen3_0_6b        # Qwen3 0.6B")
    print("  modal run modal_eval.py::serve_qwen3_1_7b        # Qwen3 1.7B")
    print("  modal run modal_eval.py::serve_qwen3_4b          # Qwen3 4B")
    print("  modal run modal_eval.py::serve_qwen3_8b          # Qwen3 8B")
    print("  modal run modal_eval.py::serve_qwen3_14b         # Qwen3 14B")
    print("  modal run modal_eval.py::serve_qwen3_32b         # Qwen3 32B")


@app.function(
    image=vllm_image,
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=600,  # 10 minutes
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(model_name: str):
    """Download and cache a model."""
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"📥 Downloading {full_model_name}...")

    try:
        from huggingface_hub import snapshot_download

        # Download the model to our cache directory
        snapshot_download(
            repo_id=full_model_name,
            cache_dir=MODEL_CACHE_PATH,
            local_files_only=False,
        )

        print(f"✅ Model {full_model_name} downloaded successfully!")
        return f"Successfully downloaded {full_model_name}"

    except Exception as e:
        error_msg = f"Failed to download {full_model_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return f"ERROR: {error_msg}"


def parse_fuse_operation(operation_str: str) -> tuple[str, str, Dict[str, Any]]:
    """Parse a FUSE operation string into components."""
    parts = operation_str.split(":")
    if len(parts) >= 2:
        operation = parts[0].strip()
        path = parts[1].strip()
        
        # Try to parse additional parameters as JSON if present
        params = {}
        if len(parts) > 2:
            try:
                params = json.loads(":".join(parts[2:]))
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as a simple string parameter
                params = {"extra": ":".join(parts[2:])}
        
        return operation, path, params
    else:
        return operation_str, "", {}


def calculate_similarity(
    predicted: str, expected: str, operation_type: str = "unknown"
) -> float:
    """Calculate similarity between predicted and expected outputs."""
    if not predicted or not expected:
        return 0.0
    
    # For exact matches
    if predicted.strip() == expected.strip():
        return 1.0
    
    # For partial matches, use simple word overlap
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    
    if not exp_words:
        return 0.0
    
    intersection = pred_words.intersection(exp_words)
    return len(intersection) / len(exp_words)


def load_evaluation_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from file."""
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return []
    
    examples = []
    try:
        with open(dataset_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        example = json.loads(line.strip())
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Line {line_num}: Invalid JSON - {e}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []
    
    print(f"📊 Loaded {len(examples)} examples from {dataset_file}")
    return examples 