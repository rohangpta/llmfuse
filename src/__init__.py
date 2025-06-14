"""
llmfuse: LLM training for stateful filesystem reasoning.

This package provides tools for generating synthetic filesystem operation
data and training LLMs to predict filesystem state changes.
"""

__version__ = "0.1.0"

# Standard library imports
import os
import sys
from pathlib import Path

# Add the src directory to Python path for clean imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and expose main classes
from .llmfuse import LLMFS, main as mount_fuse
from .eval import evaluate_dataset
from .generate_data import generate_data
from .fs_state import FSState, FileEntry
from .model import get_model_response

__all__ = [
    'LLMFS',
    'mount_fuse', 
    'evaluate_dataset',
    'generate_data',
    'FSState',
    'FileEntry',
    'get_model_response'
] 