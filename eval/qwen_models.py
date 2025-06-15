"""
Qwen model serving functions for different model sizes.
Contains the individual serving functions for each Qwen model variant.
"""

from eval.common import (
    app, vllm_image, model_cache, vllm_cache, 
    MODEL_CACHE_PATH, VLLM_CACHE_PATH, VLLM_PORT, MINUTES,
    QWEN3_MODELS
)
import modal
import subprocess
import time


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
    """Start a vLLM server for the specified model."""
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")

    # Start the server
    process = subprocess.Popen(cmd)
    
    # Wait a bit for the server to start
    time.sleep(30)
    
    print(f"✅ vLLM server started for {full_model_name}")
    return f"Server started for {full_model_name}"


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
    """Serve Qwen3 0.6B model via vLLM with OpenAI-compatible API."""
    model_name = "qwen3-0.6b"
    full_model_name = QWEN3_MODELS[model_name]
    
    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
    """Serve Qwen3 1.7B model via vLLM with OpenAI-compatible API."""
    model_name = "qwen3-1.7b"
    full_model_name = QWEN3_MODELS[model_name]
    
    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
    """Serve Qwen3 4B model via vLLM with OpenAI-compatible API."""
    model_name = "qwen3-4b"
    full_model_name = QWEN3_MODELS[model_name]
    
    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
    """Serve Qwen3 8B model via vLLM with OpenAI-compatible API."""
    model_name = "qwen3-8b"
    full_model_name = QWEN3_MODELS[model_name]
    
    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
    """Serve Qwen3 14B model via vLLM with OpenAI-compatible API."""
    model_name = "qwen3-14b"
    full_model_name = QWEN3_MODELS[model_name]
    
    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
    """Serve Qwen3 32B model via vLLM with OpenAI-compatible API."""
    model_name = "qwen3-32b"
    full_model_name = QWEN3_MODELS[model_name]
    
    print(f"🚀 Starting vLLM server for {full_model_name}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"🔧 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# Alias for the default serving function
serve_qwen3 = serve_qwen3_8b 