"""State compression helpers for llmfuse.

We use the LLMEncode pipeline (locally or via a remote Modal endpoint) to compress
the canonical XML filesystem string, falling back to zlib/identity when the heavy
model path isn't available.
"""
from __future__ import annotations

import base64
import os
import requests
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

DEFAULT_LLMENCODE_REMOTE_ENDPOINT = "https://rl-for-fun--llmencode-service-serve.modal.run"
LLMENCODE_REMOTE_ENDPOINT_ENV = "LLMENCODE_REMOTE_ENDPOINT"

class StateCodecError(RuntimeError):
    """Raised when a codec cannot complete its operation."""


class BaseStateCodec:
    codec_id = "identity"

    def compress(self, text: str) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def decompress(self, payload: Dict[str, Any]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class IdentityStateCodec(BaseStateCodec):
    codec_id = "identity"

    def compress(self, text: str) -> Dict[str, Any]:
        return {"codec": self.codec_id, "blob": text}

    def decompress(self, payload: Dict[str, Any]) -> str:
        return payload.get("blob", "")


class ZlibStateCodec(BaseStateCodec):
    codec_id = "zlib"

    def compress(self, text: str) -> Dict[str, Any]:
        blob = base64.b64encode(zlib.compress(text.encode("utf-8"))).decode("ascii")
        return {"codec": self.codec_id, "blob": blob}

    def decompress(self, payload: Dict[str, Any]) -> str:
        blob = payload.get("blob")
        if blob is None:
            raise StateCodecError("Missing blob for zlib codec")
        data = zlib.decompress(base64.b64decode(blob.encode("ascii")))
        return data.decode("utf-8")


class LLMEncodeStateCodec(BaseStateCodec):
    codec_id = "llmencode"

    def __init__(self, model_name: str, max_tokens: int) -> None:
        try:
            from common.model import (
                compress_with_model_probs,
                decompress_with_model_probs,
            )
        except Exception as exc:  # pragma: no cover - import guard
            raise StateCodecError("LLMEncode dependencies unavailable") from exc

        self._compress_fn = compress_with_model_probs
        self._decompress_fn = decompress_with_model_probs
        self._model_name = model_name
        self._max_tokens = max_tokens

    def compress(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"codec": self.codec_id, "blob": "", "metadata": None}
        try:
            compressed_bytes, metadata = self._compress_fn(
                text,
                model_name=self._model_name,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - heavy path
            raise StateCodecError(f"LLMEncode compression failed: {exc}") from exc

        blob = base64.b64encode(compressed_bytes).decode("ascii")
        return {"codec": self.codec_id, "blob": blob, "metadata": metadata}

    def decompress(self, payload: Dict[str, Any]) -> str:
        blob = payload.get("blob")
        metadata = payload.get("metadata")
        if metadata is None:
            raise StateCodecError("Missing metadata for LLMEncode payload")
        try:
            compressed_bytes = base64.b64decode(blob.encode("ascii")) if blob else b""
            if not compressed_bytes:
                return ""
            return self._decompress_fn(compressed_bytes, metadata)
        except Exception as exc:  # pragma: no cover - heavy path
            raise StateCodecError(f"LLMEncode decompression failed: {exc}") from exc


class RemoteLLMEncodeStateCodec(BaseStateCodec):
    codec_id = "llmencode-remote"

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 20000,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._max_tokens = max_tokens

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["x-llmencode-token"] = self._token
        return headers

    def compress(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"codec": self.codec_id, "blob": "", "metadata": None}
        payload = {"text": text, "max_tokens": self._max_tokens}
        try:
            print(f"[LLMFUSE] Remote state compress -> {self._endpoint}/encode (max_tokens={self._max_tokens})")
            print(f"[LLMFUSE] Payload length: {len(text)}")
            resp = requests.post(
                f"{self._endpoint}/encode",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"[LLMFUSE] Remote state compress OK. blob_len={len(str(data.get('compressed_b64', '')))}")
        except Exception as exc:
            raise StateCodecError(f"Remote LLMEncode compression failed: {exc}") from exc

        blob = data.get("compressed_b64")
        metadata = data.get("metadata")
        if not blob or metadata is None:
            raise StateCodecError("Remote encode response missing blob or metadata")
        return {"codec": self.codec_id, "blob": blob, "metadata": metadata}

    def decompress(self, payload: Dict[str, Any]) -> str:
        blob = payload.get("blob")
        metadata = payload.get("metadata")
        if metadata is None:
            raise StateCodecError("Missing metadata for remote LLMEncode payload")
        try:
            print(f"[LLMFUSE] Remote state decompress -> {self._endpoint}/decode")
            resp = requests.post(
                f"{self._endpoint}/decode",
                json={"compressed_b64": blob, "metadata": metadata},
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            print("[LLMFUSE] Remote state decompress OK.")
        except Exception as exc:
            raise StateCodecError(f"Remote LLMEncode decompression failed: {exc}") from exc

        text = data.get("text")
        if text is None:
            raise StateCodecError("Remote decode response missing text")
        return text


class StateCodecManager:
    """Facade that picks the best available codec with transparent fallback."""

    def __init__(self) -> None:
        preferred = os.environ.get("LLMFUSE_STATE_CODEC", "llmencode-remote").lower().replace("_", "-")
        model_name = os.environ.get("LLMFUSE_STATE_MODEL", "qwen3-4b")
        max_tokens = 20000
        # Remote llmencode endpoint (e.g., Modal service)
        remote_endpoint = os.environ.get(LLMENCODE_REMOTE_ENDPOINT_ENV) or DEFAULT_LLMENCODE_REMOTE_ENDPOINT
        remote_token = None  # Token not used
        remote_timeout = 60.0
        remote_max_tokens = max_tokens

        self._codecs: Dict[str, BaseStateCodec] = {
            "identity": IdentityStateCodec(),
            "zlib": ZlibStateCodec(),
        }

        llmencode_requested = preferred == "llmencode" or os.environ.get(
            "LLMFUSE_STATE_CODEC_TRY_LLMENCODE", "0"
        ) == "1"

        if llmencode_requested:
            try:
                self._codecs["llmencode"] = LLMEncodeStateCodec(model_name, max_tokens)
            except StateCodecError as exc:
                print(f"[LLMFUSE] LLMEncode codec unavailable: {exc}. Falling back.")
                if preferred == "llmencode":
                    preferred = "zlib"

        if remote_endpoint:
            try:
                self._codecs["llmencode-remote"] = RemoteLLMEncodeStateCodec(
                    remote_endpoint,
                    token=remote_token,
                    timeout=remote_timeout,
                    max_tokens=remote_max_tokens,
                )
            except StateCodecError as exc:
                print(f"[LLMFUSE] Remote LLMEncode codec unavailable: {exc}. Falling back.")
                if preferred == "llmencode-remote":
                    preferred = "zlib"
        else:
            print("[LLMFUSE] Remote LLMEncode endpoint not configured; skipping llmencode-remote codec.")

        self._preferred = preferred if preferred in self._codecs else self._default_preference

    @property
    def _default_preference(self) -> str:
        if "llmencode-remote" in self._codecs:
            return "llmencode-remote"
        if "llmencode" in self._codecs:
            return "llmencode"
        return "zlib"

    def _codec_order(self) -> list[str]:
        order = [self._preferred]
        for name in ("llmencode-remote", "llmencode", "zlib", "identity"):
            if name not in order and name in self._codecs:
                order.append(name)
        return order

    def compress(self, text: str) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for name in self._codec_order():
            codec = self._codecs.get(name)
            if not codec:
                continue
            try:
                payload = codec.compress(text)
                payload.setdefault("codec", codec.codec_id)
                return payload
            except Exception as exc:
                last_error = exc
                print(f"[LLMFUSE] State codec {name} failed: {exc}")
        raise StateCodecError(f"No state codec succeeded: {last_error}")

    def decompress(self, payload: Dict[str, Any]) -> str:
        codec_name = (payload or {}).get("codec", "identity")
        codec = self._codecs.get(codec_name)
        if not codec:
            codec = self._codecs["identity"]
        return codec.decompress(payload)


@dataclass
class CompressedStateStore:
    codec_manager: StateCodecManager
    _payload: Optional[Dict[str, Any]] = None
    _cache: Optional[str] = None

    def update(self, xml_state: str) -> None:
        self._cache = xml_state
        self._payload = self.codec_manager.compress(xml_state)

    def get(self) -> str:
        if self._cache is not None:
            return self._cache
        if not self._payload:
            return ""
        self._cache = self.codec_manager.decompress(self._payload)
        return self._cache

    def clear_cache(self) -> None:
        self._cache = None
