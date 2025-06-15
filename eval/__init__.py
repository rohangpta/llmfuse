"""
Evaluation package for Qwen3 models.

This package contains:
- common: Shared utilities and Modal setup
- qwen_models: Model serving functions for different sizes
- evaluation: Evaluation and testing functions
- modal_eval: Main entrypoint
"""

from .modal_eval import (
    test_qwen3, test_qwen3_single, compare_gemini_qwen3,
    test_qwen3_with_vllm, test_qwen3_single_vllm, compare_gemini_qwen3_vllm
)

from .common import (
    app, print_available_models, get_available_qwen3_models, download_model
)

from .qwen_models import (
    serve_qwen3, serve_qwen3_0_6b, serve_qwen3_1_7b, 
    serve_qwen3_4b, serve_qwen3_8b, serve_qwen3_14b, serve_qwen3_32b
)

from .evaluation import (
    eval_single_size, eval_sizes, analyze_eval_results
)

__all__ = [
    # Main testing functions
    'test_qwen3', 'test_qwen3_single', 'compare_gemini_qwen3',
    'test_qwen3_with_vllm', 'test_qwen3_single_vllm', 'compare_gemini_qwen3_vllm',
    
    # Utility functions
    'print_available_models', 'get_available_qwen3_models', 'download_model',
    
    # Serving functions
    'serve_qwen3', 'serve_qwen3_0_6b', 'serve_qwen3_1_7b', 
    'serve_qwen3_4b', 'serve_qwen3_8b', 'serve_qwen3_14b', 'serve_qwen3_32b',
    
    # Evaluation functions
    'eval_single_size', 'eval_sizes', 'analyze_eval_results',
    
    # Modal app
    'app'
] 