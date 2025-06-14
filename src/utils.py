"""
Shared utilities and constants for the llmfuse project.
"""
import re
from typing import Optional

# File system constants
DEFAULT_FILE_MODE = 0o644
DEFAULT_DIR_MODE = 0o755
TEMP_DIR_MODE = 0o700

# Tree formatting constants
TREE_PADDING = 32
TREE_CONNECTOR_LAST = "└── "
TREE_CONNECTOR_MIDDLE = "├── "
TREE_VERTICAL_LINE = "│   "
TREE_INDENT = "    "

# Size formatting constants
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

def extract_result_from_llm_output(output: str) -> str:
    """
    Extracts the result string from within the <result> tags.
    
    Args:
        output: Raw LLM output string that may contain <result> tags
        
    Returns:
        The content within <result> tags, or the original output if no tags found
    """
    match = re.search(r'<result>(.*?)</result>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output.strip()  # Fallback 