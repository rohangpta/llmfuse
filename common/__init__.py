"""
LLMFuse Common - Shared utilities and model interfaces.

This package contains common functionality used across the LLM-FUSE project:
- Model interfaces for LLM communication
- Shared utilities and helper functions
"""

# Core modules for common functionality
# LLMEncode functionality moved to llmencode package
# LLMFuse functionality moved to llmfuse package
from .model import (
    get_model_response,
    get_model_logprobs,
    compress_with_model_probs,
    decompress_with_model_probs,
    query_model,
    validate_api_key,
    get_available_models,
    get_available_qwen3_models,
)

__version__ = "0.1.0"
__author__ = "LLMFuse Team"
__description__ = "Common utilities and model interfaces for LLM-FUSE"

__all__ = [
    # Model interface
    "get_model_response",
    "get_model_logprobs", 
    "compress_with_model_probs",
    "decompress_with_model_probs",
    "query_model",
    "validate_api_key",
    "get_available_models",
    "get_available_qwen3_models",
] 