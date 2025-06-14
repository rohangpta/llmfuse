"""
Modal app for testing Qwen3 models using vLLM server on cloud infrastructure.

This app runs Qwen3 models via vLLM server with OpenAI-compatible API,
allowing efficient inference and comparison of different model sizes.

Usage:
    modal run modal_run.py::test_qwen3
    modal run modal_run.py::test_qwen3_single --model-name="qwen3-8b"
    modal run modal_run.py::compare_gemini_qwen3
    modal run modal_run.py::serve_qwen3 --model-name="qwen3-8b"
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
)

# Model configuration constants
DEFAULT_MODEL = "models/gemini-2.5-flash-preview-05-20"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192
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
    temperature: float = DEFAULT_TEMPERATURE,
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
        "model": full_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "stream": False,
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
    """Check if the Gemini API key is properly configured."""
    return API_KEY_ENV_VAR in os.environ


def get_available_qwen3_models() -> List[str]:
    """Get list of available Qwen3 model shortcuts."""
    return list(QWEN3_MODELS.keys())


def print_available_models() -> None:
    """Print all available model options."""
    print("Available Qwen3 models:")
    for short_name, full_name in QWEN3_MODELS.items():
        print(f"  {short_name:<20} -> {full_name}")

    print("\nGemini models:")
    print(f"  {DEFAULT_MODEL} (default)")
    print("  Or any other Gemini model name starting with 'models/gemini'")

    print(
        "\nYou can also use any Hugging Face model name directly (e.g., 'microsoft/DialoGPT-medium')"
    )


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
    """Download and cache a specific Qwen3 model to avoid downloading during inference."""
    from huggingface_hub import snapshot_download

    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"📦 Downloading and caching model: {full_model_name}")

    try:
        # Download model files using snapshot_download
        snapshot_download(
            full_model_name,
            ignore_patterns=["*.pt", "*.bin"],  # Use safetensors when available
        )

        # Ensure the volume persists the changes
        model_cache.commit()

        print(f"✅ Model {full_model_name} successfully cached!")
        return f"Model {full_model_name} cached successfully"

    except Exception as e:
        error_msg = f"Failed to cache model {full_model_name}: {e}"
        print(f"❌ {error_msg}")
        return f"ERROR: {error_msg}"


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def start_vllm_server(model_name: str = "qwen3-8b"):
    """Start a vLLM server for the specified model and return the process."""
    import subprocess
    import time

    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Start the server process
    process = subprocess.Popen(cmd)

    # Wait a bit for server to start
    print("⏳ Waiting for vLLM server to initialize...")
    time.sleep(30)  # Give server time to start

    # Keep the function running to maintain the server
    try:
        process.wait()
    except KeyboardInterrupt:
        print("🛑 Stopping vLLM server...")
        process.terminate()
        process.wait()

    return f"vLLM server for {full_model_name} stopped"


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_qwen3_0_6b():
    """Serve Qwen3 0.6B model using vLLM."""
    import subprocess

    model_name = "qwen3-0.6b"
    full_model_name = QWEN3_MODELS[model_name]

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_qwen3_1_7b():
    """Serve Qwen3 1.7B model using vLLM."""
    import subprocess

    model_name = "qwen3-1.7b"
    full_model_name = QWEN3_MODELS[model_name]

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_qwen3_4b():
    """Serve Qwen3 4B model using vLLM."""
    import subprocess

    model_name = "qwen3-4b"
    full_model_name = QWEN3_MODELS[model_name]

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_qwen3_8b():
    """Serve Qwen3 8B model using vLLM."""
    import subprocess

    model_name = "qwen3-8b"
    full_model_name = QWEN3_MODELS[model_name]

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_qwen3_14b():
    """Serve Qwen3 14B model using vLLM."""
    import subprocess

    model_name = "qwen3-14b"
    full_model_name = QWEN3_MODELS[model_name]

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_qwen3_32b():
    """Serve Qwen3 32B model using vLLM."""
    import subprocess

    model_name = "qwen3-32b"
    full_model_name = QWEN3_MODELS[model_name]

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "vllm",
        "serve",
        full_model_name,  # Model as positional argument
        "--served-model-name",
        full_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def test_qwen3_with_vllm():
    """Test Qwen3 models using direct vLLM inference (no server)."""

    print("🚀 QWEN3 DIRECT INFERENCE TESTING ON MODAL")
    print("=" * 80)

    # Test prompts related to filesystem operations
    test_prompts = [
        "What happens when I run 'ls -la' in a directory?",
        "Explain the difference between absolute and relative paths.",
        "What is the purpose of file permissions in Unix systems?",
    ]

    # Test different Qwen3 sizes
    models_to_test = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]

    results = {}

    for model_name in models_to_test:
        print(f"\n🤖 Testing {model_name.upper()}")
        print("-" * 60)

        # Convert simplified model names to full HF model names
        if model_name.lower() in QWEN3_MODELS:
            full_model_name = QWEN3_MODELS[model_name.lower()]
        else:
            full_model_name = model_name

        results[model_name] = {}

        # Load model directly for inference
        try:
            from vllm import LLM, SamplingParams

            print(f"📦 Loading {full_model_name}...")
            llm = LLM(
                model=full_model_name,
                trust_remote_code=True,
                enforce_eager=True,  # Faster startup
                gpu_memory_utilization=0.8,
            )

            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=200,
                top_p=0.9,
            )

            for prompt in test_prompts:
                print(f"\n📝 Prompt: {prompt}")

                try:
                    # Format prompt for Qwen3 chat format
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = llm.get_tokenizer().apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,  # Disable thinking mode
                    )

                    outputs = llm.generate([formatted_prompt], sampling_params)
                    response = outputs[0].outputs[0].text.strip()

                    # Clean up any remaining thinking tags
                    if response.startswith("<think>"):
                        # Find the end of thinking and extract the actual response
                        think_end = response.find("</think>")
                        if think_end != -1:
                            response = response[think_end + 8 :].strip()
                        else:
                            # If no closing tag, take everything after <think>
                            response = response[7:].strip()

                    results[model_name][prompt] = response
                    print(f"✅ Response: {response[:100]}...")

                except Exception as e:
                    error_msg = f"Error: {e}"
                    results[model_name][prompt] = error_msg
                    print(f"❌ {error_msg}")

            # Clean up model to free memory
            del llm
            import gc

            gc.collect()

        except Exception as e:
            error_msg = f"Failed to load model {full_model_name}: {e}"
            print(f"❌ {error_msg}")
            for prompt in test_prompts:
                results[model_name][prompt] = error_msg

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY:")
    print("=" * 80)

    for model_name, model_results in results.items():
        print(f"\n🤖 {model_name.upper()}:")
        for prompt, response in model_results.items():
            status = (
                "✅"
                if not response.startswith("Error")
                and not response.startswith("Failed")
                else "❌"
            )
            print(f"  {status} {prompt[:50]}...")
            if not response.startswith("Error") and not response.startswith("Failed"):
                print(f"      Response: {response[:100]}...")

    print(
        f"\n🎉 Testing completed! Tested {len(results)} models on {len(test_prompts)} prompts."
    )
    return results


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=1800,  # 30 minutes
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def test_qwen3_single_vllm(model_name: str = "qwen3-8b"):
    """Test a single Qwen3 model with direct vLLM inference."""

    print(f"🚀 Testing {model_name.upper()} with direct vLLM inference")
    print("=" * 60)

    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    # Test prompts
    prompts = [
        "Explain how file permissions work in Linux",
        "What's the difference between a hard link and a symbolic link?",
        "How does the 'find' command work and what are some useful options?",
        "Describe the Linux filesystem hierarchy and key directories",
    ]

    results = []

    try:
        from vllm import LLM, SamplingParams

        print(f"📦 Loading {full_model_name}...")
        llm = LLM(
            model=full_model_name,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )

        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=300,
            top_p=0.9,
        )

        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            print("-" * 60)

            try:
                # Format prompt for Qwen3 chat format
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = llm.get_tokenizer().apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Disable thinking mode
                )

                outputs = llm.generate([formatted_prompt], sampling_params)
                response = outputs[0].outputs[0].text.strip()

                # Clean up any remaining thinking tags
                if response.startswith("<think>"):
                    # Find the end of thinking and extract the actual response
                    think_end = response.find("</think>")
                    if think_end != -1:
                        response = response[think_end + 8 :].strip()
                    else:
                        # If no closing tag, take everything after <think>
                        response = response[7:].strip()

                print(f"🤖 {model_name} response:")
                print(response)
                results.append(
                    {"prompt": prompt, "response": response, "status": "success"}
                )

            except Exception as e:
                error_msg = f"Error: {e}"
                print(f"❌ {error_msg}")
                results.append(
                    {"prompt": prompt, "response": error_msg, "status": "error"}
                )

        # Clean up
        del llm
        import gc

        gc.collect()

    except Exception as e:
        error_msg = f"Failed to load model {full_model_name}: {e}"
        print(f"❌ {error_msg}")
        for prompt in prompts:
            results.append({"prompt": prompt, "response": error_msg, "status": "error"})

    return results


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=2400,  # 40 minutes
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def compare_gemini_qwen3_vllm():
    """Compare Gemini and Qwen3 responses using direct vLLM inference."""

    print("🔥 GEMINI vs QWEN3 (Direct vLLM) COMPARISON")
    print("=" * 80)

    prompt = "Describe what a filesystem is and give examples of common filesystem operations like creating directories, copying files, and setting permissions."

    print(f"📝 PROMPT: {prompt}")
    print("-" * 80)

    # Test Gemini
    if validate_api_key():
        print("\n🔥 GEMINI RESPONSE:")
        try:
            gemini_response = _get_gemini_response(prompt)
            print(gemini_response)
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
    else:
        print("\n⚠️  Gemini API key not configured. Skipping Gemini test.")

    # Test Qwen3 with direct vLLM
    print("\n🤖 QWEN3-8B (Direct vLLM) RESPONSE:")

    try:
        from vllm import LLM, SamplingParams

        model_name = "qwen3-8b"
        full_model_name = QWEN3_MODELS[model_name]

        print(f"📦 Loading {full_model_name}...")
        llm = LLM(
            model=full_model_name,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            top_p=1.0,
        )

        # Format prompt for Qwen3 chat format
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable thinking mode
        )

        outputs = llm.generate([formatted_prompt], sampling_params)
        qwen3_response = outputs[0].outputs[0].text.strip()

        # Clean up any remaining thinking tags
        if qwen3_response.startswith("<think>"):
            # Find the end of thinking and extract the actual response
            think_end = qwen3_response.find("</think>")
            if think_end != -1:
                qwen3_response = qwen3_response[think_end + 8 :].strip()
            else:
                # If no closing tag, take everything after <think>
                qwen3_response = qwen3_response[7:].strip()

        print(qwen3_response)

        # Clean up
        del llm
        import gc

        gc.collect()

    except Exception as e:
        print(f"❌ Qwen3 Error: {e}")

    print("\n" + "=" * 80)
    return "✅ Gemini vs Qwen3 (Direct vLLM) comparison completed!"


# Legacy function names for backward compatibility
test_qwen3 = test_qwen3_with_vllm
test_qwen3_single = test_qwen3_single_vllm
compare_gemini_qwen3 = compare_gemini_qwen3_vllm


@app.local_entrypoint()
def main():
    """Local entrypoint for development."""
    print("Available Modal functions:")
    print("  modal run modal_run.py::test_qwen3              # Full vLLM test suite")
    print(
        "  modal run modal_run.py::test_qwen3_single       # Test single model with vLLM"
    )
    print(
        "  modal run modal_run.py::compare_gemini_qwen3    # Compare Gemini vs Qwen3 (vLLM)"
    )
    print(
        "  modal run modal_run.py::eval_sizes              # Evaluate all Qwen3 sizes on filesystem ops"
    )
    print(
        "  modal run modal_run.py::analyze_eval_results --results-filename='qwen3_eval_results_1234567890.json'  # Analyze saved results"
    )
    print(
        "  modal run modal_run.py::download_model --model-name='qwen3-8b'  # Cache model"
    )
    print("\nAvailable vLLM servers:")
    print("  modal run modal_run.py::serve_qwen3             # Default (qwen3-8b)")
    print("  modal run modal_run.py::serve_qwen3_0_6b        # Qwen3 0.6B")
    print("  modal run modal_run.py::serve_qwen3_1_7b        # Qwen3 1.7B")
    print("  modal run modal_run.py::serve_qwen3_4b          # Qwen3 4B")
    print("  modal run modal_run.py::serve_qwen3_8b          # Qwen3 8B")
    print("  modal run modal_run.py::serve_qwen3_14b         # Qwen3 14B")
    print("  modal run modal_run.py::serve_qwen3_32b         # Qwen3 32B")


def parse_fuse_operation(operation_str: str) -> tuple[str, str, Dict[str, Any]]:
    """Parse a FUSE operation string into components."""
    if operation_str.startswith("FUSE "):
        operation_str = operation_str[5:]  # Remove 'FUSE ' prefix

    if "(" in operation_str and ")" in operation_str:
        op_name = operation_str.split("(")[0]
        params_part = operation_str[
            operation_str.find("(") + 1 : operation_str.rfind(")")
        ]

        path = ""
        params = {}

        if params_part:
            parts = [p.strip() for p in params_part.split(",")]
            if parts:
                path_part = parts[0].strip("'\"")
                path = path_part

                for part in parts[1:]:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        params[key] = value

        return op_name, path, params
    else:
        return "shell", operation_str, {}


def calculate_similarity(
    predicted: str, expected: str, operation_type: str = "unknown"
) -> float:
    """Calculate similarity between predicted and expected results."""
    if not predicted and not expected:
        return 1.0

    if not predicted or not expected:
        return 0.0

    if predicted.strip() == expected.strip():
        return 1.0

    pred_normalized = " ".join(predicted.split())
    exp_normalized = " ".join(expected.split())

    if pred_normalized == exp_normalized:
        return 1.0

    # Character-level similarity for filesystem operations
    pred_chars = set(pred_normalized.lower())
    exp_chars = set(exp_normalized.lower())

    if not pred_chars and not exp_chars:
        return 1.0

    intersection = len(pred_chars.intersection(exp_chars))
    union = len(pred_chars.union(exp_chars))

    return intersection / union if union > 0 else 0.0


def load_evaluation_dataset() -> List[Dict[str, Any]]:
    """Load evaluation dataset for filesystem operations."""
    # Sample evaluation dataset for filesystem operations
    return [
        {
            "initial_state": "/\n├── home/\n│   └── user/\n└── tmp/",
            "operation": "mkdir('/home/user/documents', mode='0o755')",
            "result": "/\n├── home/\n│   └── user/\n│       └── documents/\n└── tmp/",
            "operation_type": "state_change",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n│       ├── file1.txt\n│       └── file2.txt\n└── tmp/",
            "operation": "readdir('/home/user')",
            "result": "file1.txt\nfile2.txt",
            "operation_type": "query",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n│       └── test.txt\n└── tmp/",
            "operation": "unlink('/home/user/test.txt')",
            "result": "/\n├── home/\n│   └── user/\n└── tmp/",
            "operation_type": "state_change",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n│       └── docs/\n│           └── readme.txt\n└── tmp/",
            "operation": "chmod('/home/user/docs/readme.txt', mode='0o644')",
            "result": "/\n├── home/\n│   └── user/\n│       └── docs/\n│           └── readme.txt (644)",
            "operation_type": "state_change",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n│       ├── project/\n│       └── backup/\n└── tmp/",
            "operation": "readdir('/home/user')",
            "result": "project\nbackup",
            "operation_type": "query",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n└── tmp/\n    └── cache/",
            "operation": "rmdir('/tmp/cache')",
            "result": "/\n├── home/\n│   └── user/\n└── tmp/",
            "operation_type": "state_change",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n│       └── old_name.txt\n└── tmp/",
            "operation": "rename('/home/user/old_name.txt', '/home/user/new_name.txt')",
            "result": "/\n├── home/\n│   └── user/\n│       └── new_name.txt\n└── tmp/",
            "operation_type": "state_change",
        },
        {
            "initial_state": "/\n├── home/\n│   └── user/\n│       └── file.txt\n└── tmp/",
            "operation": "getattr('/home/user/file.txt')",
            "result": "type: file, size: 1024, mode: 644",
            "operation_type": "query",
        },
    ]


async def evaluate_model_on_dataset(
    llm, model_name: str, dataset: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate a single model on the dataset."""
    from vllm import SamplingParams

    results = []

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
    )

    for example in dataset:
        try:
            # Create evaluation prompt
            prompt = f"""You are simulating a filesystem. Given the current state and operation, predict the exact result.

Current filesystem state:
{example["initial_state"]}

Operation: {example["operation"]}

Provide only the result as it would appear after the operation:"""

            # Format for Qwen3 chat format
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = llm.get_tokenizer().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            outputs = llm.generate([formatted_prompt], sampling_params)
            predicted = outputs[0].outputs[0].text.strip()

            # Clean up any remaining thinking tags
            if predicted.startswith("<think>"):
                think_end = predicted.find("</think>")
                if think_end != -1:
                    predicted = predicted[think_end + 8 :].strip()
                else:
                    predicted = predicted[7:].strip()

            expected = example["result"]
            similarity = calculate_similarity(
                predicted, expected, example["operation_type"]
            )
            correct = similarity > 0.7  # 70% similarity threshold

            results.append(
                {
                    "operation": example["operation"],
                    "operation_type": example["operation_type"],
                    "predicted": predicted,
                    "expected": expected,
                    "similarity": similarity,
                    "correct": correct,
                }
            )

        except Exception as e:
            results.append(
                {
                    "operation": example["operation"],
                    "operation_type": example["operation_type"],
                    "predicted": f"Error: {str(e)}",
                    "expected": example["result"],
                    "similarity": 0.0,
                    "correct": False,
                }
            )

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Statistics by operation type
    by_type = {}
    for result in results:
        op_type = result["operation_type"]
        if op_type not in by_type:
            by_type[op_type] = []
        by_type[op_type].append(result)

    type_stats = {}
    for op_type, type_results in by_type.items():
        type_correct = sum(1 for r in type_results if r["correct"])
        type_stats[op_type] = {
            "total": len(type_results),
            "correct": type_correct,
            "accuracy": type_correct / len(type_results) if type_results else 0.0,
        }

    return {
        "model_name": model_name,
        "total_examples": total,
        "correct_predictions": correct,
        "overall_accuracy": accuracy,
        "accuracy_by_type": type_stats,
        "detailed_results": results,
    }


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=20 * MINUTES,  # 20 minutes per model (faster with H100)
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def eval_single_size(model_name: str) -> Dict[str, Any]:
    """Evaluate a single Qwen model size on filesystem operations."""

    print(f"🚀 Evaluating {model_name.upper()} on filesystem operations")
    print("-" * 60)

    # Load evaluation dataset
    dataset = load_evaluation_dataset()

    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    try:
        from vllm import LLM

        print(f"📦 Loading {full_model_name}...")
        llm = LLM(
            model=full_model_name,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )

        # Evaluate model on dataset
        result = await evaluate_model_on_dataset(llm, model_name, dataset)

        print(f"✅ {model_name}: {result['overall_accuracy']:.2%} accuracy")
        print(f"   Correct: {result['correct_predictions']}/{result['total_examples']}")

        # Clean up model to free memory
        del llm
        import gc

        gc.collect()

        return result

    except Exception as e:
        print(f"❌ Failed to evaluate {model_name}: {e}")
        return {"model_name": model_name, "error": str(e), "overall_accuracy": 0.0}


@app.function(
    image=vllm_image,
    volumes={EVAL_RESULTS_PATH: eval_results_volume},
    timeout=40 * MINUTES,  # 40 minutes total timeout (faster with H100s)
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def eval_sizes():
    """Evaluate all Qwen model sizes on filesystem operations in parallel."""

    print("🔥 EVALUATING ALL QWEN3 MODEL SIZES ON FILESYSTEM OPERATIONS (PARALLEL)")
    print("=" * 80)

    # Load evaluation dataset for info
    dataset = load_evaluation_dataset()
    print(f"📊 Loaded {len(dataset)} evaluation examples")

    # Models to evaluate (starting with smaller ones)
    # Note: Add "qwen3-32b" if you have enough GPU quota and time
    models_to_eval = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"]

    print(f"🚀 Launching {len(models_to_eval)} parallel evaluations...")
    print("   Each model will run on its own GPU instance")

    # Run evaluations in parallel using Modal's async map functionality
    results_list = []
    async for result in eval_single_size.map.aio(models_to_eval):
        results_list.append(result)
        print(f"📥 Received results for {result['model_name']}")

    # Convert list of results back to dictionary
    all_results = {result["model_name"]: result for result in results_list}

    print(f"\n✅ All {len(results_list)} parallel evaluations completed!")

    # Print comparison summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY - QWEN3 MODEL COMPARISON")
    print("=" * 80)

    print("\nOverall Accuracy by Model Size:")
    sorted_models = sorted(
        all_results.items(), key=lambda x: x[1].get("overall_accuracy", 0), reverse=True
    )

    for model_name, result in sorted_models:
        if "error" not in result:
            accuracy = result["overall_accuracy"]
            correct = result["correct_predictions"]
            total = result["total_examples"]
            print(f"  {model_name:<15}: {accuracy:.2%} ({correct}/{total})")

            # Show breakdown by operation type
            for op_type, stats in result["accuracy_by_type"].items():
                print(
                    f"    {op_type:<12}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})"
                )
        else:
            print(f"  {model_name:<15}: ERROR - {result['error']}")

    # Show detailed analysis
    print("\n" + "=" * 80)
    print("📈 DETAILED ANALYSIS")
    print("=" * 80)

    # Find best and worst performing models
    successful_results = [(k, v) for k, v in all_results.items() if "error" not in v]

    if successful_results:
        best_model, best_result = max(
            successful_results, key=lambda x: x[1]["overall_accuracy"]
        )
        worst_model, worst_result = min(
            successful_results, key=lambda x: x[1]["overall_accuracy"]
        )

        print(f"\n🏆 Best performing model: {best_model}")
        print(f"   Accuracy: {best_result['overall_accuracy']:.2%}")

        print(f"\n📉 Needs improvement: {worst_model}")
        print(f"   Accuracy: {worst_result['overall_accuracy']:.2%}")

        # Show examples where models differ
        print(f"\n🔍 Example differences between {best_model} and {worst_model}:")

        best_details = best_result["detailed_results"]
        worst_details = worst_result["detailed_results"]

        differences_shown = 0
        for i, (best_ex, worst_ex) in enumerate(zip(best_details, worst_details)):
            if best_ex["correct"] != worst_ex["correct"] and differences_shown < 2:
                print(f"\n  Example {i + 1}: {best_ex['operation']}")
                print(f"    Expected: {best_ex['expected'][:100]}...")
                print(
                    f"    {best_model}: {'✅' if best_ex['correct'] else '❌'} {best_ex['predicted'][:100]}..."
                )
                print(
                    f"    {worst_model}: {'✅' if worst_ex['correct'] else '❌'} {worst_ex['predicted'][:100]}..."
                )
                differences_shown += 1

    print(
        f"\n🎉 Evaluation completed! Tested {len(models_to_eval)} models on {len(dataset)} filesystem operations."
    )

    # Save detailed results to file for later inspection
    import time

    output_data = {
        "evaluation_summary": {
            "total_models": len(models_to_eval),
            "total_examples_per_model": len(dataset),
            "models_evaluated": [
                m for m in models_to_eval if "error" not in all_results.get(m, {})
            ],
            "models_failed": [
                m for m in models_to_eval if "error" in all_results.get(m, {})
            ],
            "best_model": best_model if successful_results else None,
            "best_accuracy": best_result["overall_accuracy"]
            if successful_results
            else None,
        },
        "detailed_results": all_results,
        "evaluation_dataset": dataset,
        "evaluation_timestamp": time.time(),
        "evaluation_config": {
            "similarity_threshold": 0.7,
            "temperature": 0.0,
            "max_tokens": 512,
            "gpu_memory_utilization": 0.8,
        },
    }

    # Save to JSON file in persistent volume
    timestamp = int(time.time())
    output_filename = f"qwen3_eval_results_{timestamp}.json"
    full_path = f"{EVAL_RESULTS_PATH}/{output_filename}"

    with open(full_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Commit changes to the volume to make them persistent
    eval_results_volume.commit()

    print(f"\n💾 Detailed results saved to persistent volume: {full_path}")
    print(
        "   This file contains all predictions, expected outputs, and failure details"
    )
    print(f"   Use analyze_eval_results('{output_filename}') to inspect failures later")
    print(f"   Or download from Modal volume: qwen3-eval-results")

    return all_results


@app.function(
    image=vllm_image,
    volumes={EVAL_RESULTS_PATH: eval_results_volume},
    timeout=10 * MINUTES,
)
def analyze_eval_results(results_filename: str):
    """Analyze saved evaluation results and show detailed failure analysis."""

    print("🔍 ANALYZING EVALUATION RESULTS")
    print("=" * 50)

    # Construct full path from filename
    results_file = f"{EVAL_RESULTS_PATH}/{results_filename}"

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        summary = data["evaluation_summary"]
        detailed_results = data["detailed_results"]

        print(f"📊 Evaluation Summary:")
        print(f"   Models evaluated: {len(summary['models_evaluated'])}")
        print(f"   Models failed: {len(summary['models_failed'])}")
        print(f"   Examples per model: {summary['total_examples_per_model']}")
        print(
            f"   Best model: {summary['best_model']} ({summary['best_accuracy']:.2%})"
        )

        # Detailed failure analysis
        print(f"\n❌ FAILURE ANALYSIS")
        print("=" * 50)

        for model_name, results in detailed_results.items():
            if "error" in results:
                print(f"\n🚫 {model_name}: FAILED TO LOAD")
                print(f"   Error: {results['error']}")
                continue

            failures = [r for r in results["detailed_results"] if not r["correct"]]
            if failures:
                print(
                    f"\n🔍 {model_name} failures ({len(failures)}/{results['total_examples']}):"
                )

                for i, failure in enumerate(failures):
                    print(f"\n  Failure {i + 1}: {failure['operation']}")
                    print(f"    Expected: {failure['expected']}")
                    print(f"    Predicted: {failure['predicted']}")
                    print(f"    Similarity: {failure.get('similarity', 'N/A'):.2f}")
                    print(f"    Type: {failure['operation_type']}")
            else:
                print(f"\n✅ {model_name}: Perfect score! No failures.")

        # Operation-specific analysis
        print(f"\n📈 OPERATION-SPECIFIC ANALYSIS")
        print("=" * 50)

        # Collect results by operation
        by_operation = {}
        for model_name, results in detailed_results.items():
            if "error" in results:
                continue
            for result in results["detailed_results"]:
                op = result["operation"]
                if op not in by_operation:
                    by_operation[op] = []
                by_operation[op].append((model_name, result))

        for operation, op_results in by_operation.items():
            print(f"\n🔧 {operation}")
            correct_count = sum(1 for _, r in op_results if r["correct"])
            total_count = len(op_results)
            print(
                f"   Overall success rate: {correct_count}/{total_count} ({correct_count / total_count:.2%})"
            )

            # Show which models failed this operation
            failures = [(model, r) for model, r in op_results if not r["correct"]]
            if failures:
                print(f"   Models that failed:")
                for model, failure in failures:
                    print(f"     {model}: {failure['predicted'][:100]}...")

        return f"Analysis complete. Found {sum(len([r for r in results['detailed_results'] if not r['correct']]) for results in detailed_results.values() if 'error' not in results)} total failures across all models."

    except Exception as e:
        return f"Error analyzing results: {e}"


if __name__ == "__main__":
    main()
