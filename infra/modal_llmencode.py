"""
Modal deployment for serving LLMEncode (compression/decompression) over HTTP.

Exposes two endpoints:
- POST /encode: {text, model?, max_tokens?} -> {compressed_b64, metadata}
- POST /decode: {compressed_b64, metadata} -> {text}

Deploy: `modal run infra/modal_llmencode.py::deploy --model-name Qwen/Qwen3-4B --runtime-token <token>`
Auth (optional): set LLMENCODE_RUNTIME_TOKEN and provide x-llmencode-token header.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict

import modal

app = modal.App("llmencode-service")

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_PACKAGE_ROOT = "/root/llmfuse_pkg"

MODEL_CACHE = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)

MODEL_CACHE_PATH = "/root/models"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"

MODEL_ENV_KEY = "LLMENCODE_MODEL_NAME"
TOKEN_ENV_KEY = "LLMENCODE_RUNTIME_TOKEN"

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
        ]
    )
    # Minimal package set needed for llmencode/common utilities
    .add_local_dir(str(REPO_ROOT / "common"), f"{REMOTE_PACKAGE_ROOT}/common", copy=True)
    .add_local_dir(str(REPO_ROOT / "llmfuse"), f"{REMOTE_PACKAGE_ROOT}/llmfuse", copy=True)
    .add_local_dir(str(REPO_ROOT / "llmencode"), f"{REMOTE_PACKAGE_ROOT}/llmencode", copy=True)
    .env({"PYTHONPATH": REMOTE_PACKAGE_ROOT, "HF_HOME": MODEL_CACHE_PATH})
    .workdir("/root")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={
        MODEL_CACHE_PATH: MODEL_CACHE,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def serve():
    """FastAPI app that wraps LLMEncode encode/decode."""
    from fastapi import Body, FastAPI, Header, HTTPException

    from llmencode import LLMEncode

    model_name = os.environ.get(MODEL_ENV_KEY, DEFAULT_MODEL_NAME)
    expected_token = os.environ.get(TOKEN_ENV_KEY, "").strip()

    api = FastAPI()

    def _check_token(incoming: str | None) -> None:
        if expected_token and (incoming or "") != expected_token:
            raise HTTPException(status_code=401, detail="invalid token")

    def _encode_payload(text: str, max_tokens: int) -> Dict[str, Any]:
        encoder = LLMEncode(model_name=model_name, max_tokens=max_tokens)
        compressed_bytes = encoder.encode(text)
        metadata = encoder._last_metadata  # stored by encode()
        return {
            "compressed_b64": base64.b64encode(compressed_bytes).decode("ascii"),
            "metadata": metadata,
        }

    @api.post("/encode")
    async def encode(
        payload: Dict[str, Any] = Body(...),
        x_llmencode_token: str | None = Header(default=None, alias="x-llmencode-token"),
    ):
        _check_token(x_llmencode_token)
        text = payload.get("text", "")
        if not isinstance(text, str) or text == "":
            raise HTTPException(status_code=400, detail="'text' must be a non-empty string")
        max_tokens = int(payload.get("max_tokens", 1000))
        if max_tokens <= 0:
            raise HTTPException(status_code=400, detail="'max_tokens' must be positive")
        try:
            return _encode_payload(text, max_tokens)
        except Exception as exc:  # FastAPI consistently wraps exceptions
            raise HTTPException(status_code=500, detail=f"encode failed: {exc}")

    @api.post("/decode")
    async def decode(
        payload: Dict[str, Any] = Body(...),
        x_llmencode_token: str | None = Header(default=None, alias="x-llmencode-token"),
    ):
        _check_token(x_llmencode_token)
        compressed_b64 = payload.get("compressed_b64")
        metadata = payload.get("metadata")
        if not compressed_b64 or not metadata:
            raise HTTPException(status_code=400, detail="'compressed_b64' and 'metadata' are required")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="'metadata' must be an object")
        try:
            compressed_bytes = base64.b64decode(compressed_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64 in 'compressed_b64'")

        # Enforce model match to avoid unexpected cross-model decode attempts
        if metadata.get("model_name") and metadata["model_name"] != model_name:
            raise HTTPException(
                status_code=400,
                detail=f"metadata model_name ({metadata['model_name']}) "
                f"does not match server model ({model_name})",
            )

        try:
            encoder = LLMEncode(model_name=model_name, max_tokens=metadata.get("num_tokens", 1000))
            encoder._last_metadata = metadata
            text = encoder.decode(compressed_bytes)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"decode failed: {exc}")

    return api


@app.local_entrypoint()
def deploy(model_name: str = DEFAULT_MODEL_NAME, runtime_token: str | None = None):
    """
    Deploy the llmencode Modal service and print its public URL.

    - model_name: HF model id to use for logprobs (default Qwen/Qwen3-4B)
    - runtime_token: optional shared secret; set LLMENCODE_RUNTIME_TOKEN
    """
    env = {MODEL_ENV_KEY: model_name}
    if runtime_token:
        env[TOKEN_ENV_KEY] = runtime_token
    serve.deploy(environment=env)
    print(f"✅ Service deployed at: {serve.web_url}")


@app.local_entrypoint()
def show_endpoint():
    """Print the currently deployed endpoint URL."""
    print(f"🔗 Current endpoint: {serve.web_url}")
