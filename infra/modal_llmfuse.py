"""
Modal deployment for serving the llmfuse model over HTTP.

This module spins up a FastAPI app inside Modal with vLLM loaded once per GPU.
The filesystem daemon can then hit the exposed endpoint via HTTPS.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import modal

from llmfuse.utils import STATE_STOP_TOKEN

app = modal.App("llmfuse-runtime")

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_PACKAGE_ROOT = "/root/llmfuse_pkg"

MODEL_CACHE = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
TRAINED_MODELS = modal.Volume.from_name("qwen3-trained-models", create_if_missing=True)

MODEL_CACHE_PATH = "/root/models"
OUTPUT_MODEL_PATH = "/root/output_models"
DEFAULT_MODEL_NAME = "qwen3-4b-sft-8epochs-distributed"

MODEL_ENV_KEY = "LLMFUSE_MODEL_PATH"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "fastapi>=0.110.0",
            "uvicorn>=0.23.0",
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "accelerate>=0.24.0",
            "tokenizers>=0.15.0",
            "safetensors>=0.4.3",
            "vllm>=0.6.2",
        ]
    )
    .add_local_dir(
        str(REPO_ROOT / "llmfuse"),
        f"{REMOTE_PACKAGE_ROOT}/llmfuse",
        copy=True,
    )
    .env({"PYTHONPATH": REMOTE_PACKAGE_ROOT})
    .workdir("/root")
)


@app.function(
    image=image,
    gpu="H100",
    min_containers=1,
    timeout=60 * 60,
    volumes={
        MODEL_CACHE_PATH: MODEL_CACHE,
        OUTPUT_MODEL_PATH: TRAINED_MODELS,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
@modal.asgi_app()
def serve():
    from fastapi import Body, FastAPI, Header, HTTPException
    from vllm import LLM, SamplingParams

    model_name = os.environ.get(MODEL_ENV_KEY, DEFAULT_MODEL_NAME)
    full_model_path = (
        model_name
        if model_name.startswith("/")
        else f"{OUTPUT_MODEL_PATH}/{model_name}"
    )
    if not os.path.exists(full_model_path):
        raise RuntimeError(f"Model path not found: {full_model_path}")

    llm = LLM(
        model=full_model_path,
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=8192,
        enforce_eager=False,
    )

    sampling_cache: Dict[Tuple[float, int], SamplingParams] = {}
    expected_token = os.environ.get("LLMFUSE_RUNTIME_TOKEN", "").strip()

    def get_sampling_params(temp: float, max_tokens: int) -> SamplingParams:
        key = (temp, max_tokens)
        if key not in sampling_cache:
            sampling_cache[key] = SamplingParams(
                temperature=temp,
                top_p=1.0,
                max_tokens=max_tokens,
                n=1,
                stop=[STATE_STOP_TOKEN],
            )
        return sampling_cache[key]

    api = FastAPI()

    @api.post("/generate")
    async def generate(
        payload: Dict[str, Any] = Body(...),
        x_llmfuse_token: str | None = Header(default=None, alias="x-llmfuse-token"),
    ):
        if expected_token and (x_llmfuse_token or "") != expected_token:
            raise HTTPException(status_code=401, detail="invalid token")
        prompt = payload.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        temperature = float(payload.get("temperature", 0.0))
        max_tokens = int(payload.get("max_tokens", 1024))
        sampling_params = get_sampling_params(temperature, max_tokens)

        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        cleaned = text.strip()
        if STATE_STOP_TOKEN in cleaned:
            cleaned = cleaned.split(STATE_STOP_TOKEN, 1)[0].strip()
        return {"text": cleaned}

    return api


@app.local_entrypoint()
def deploy(model_path: str = DEFAULT_MODEL_NAME, runtime_token: str | None = None):
    """
    Deploy the Modal inference service and print its public URL.

    Override `model_path` to point at a different folder under /root/output_models.
    """
    env = {MODEL_ENV_KEY: model_path}
    if runtime_token:
        env["LLMFUSE_RUNTIME_TOKEN"] = runtime_token
    serve.deploy(environment=env)
    print(f"✅ Service deployed at: {serve.web_url}")


@app.local_entrypoint()
def show_endpoint():
    """Print the currently deployed endpoint URL."""
    print(f"🔗 Current endpoint: {serve.web_url}")
