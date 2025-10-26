"""
Synthetic filesystem data generation for LLM training using FUSE operations.

This module generates synthetic filesystem operation examples by running
a reference FUSE filesystem, performing operations through shell commands,
and capturing the actual FUSE operation calls for training data.

NOTE: This script requires FUSE support and should be run in Docker on Linux.
On macOS, use: docker-compose run --rm --privileged datagen-fuse python -m train.generate_data
"""
# Standard library imports
import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Note: datasets library not needed - we use custom content generation

# Local imports
from llmfuse.fs_state import FSState, FileEntry
from llmfuse.utils import DEFAULT_FILE_MODE, DEFAULT_DIR_MODE

# Generation constants
MIN_SETUP_OPERATIONS = 2
MAX_SETUP_OPERATIONS = 8
MIN_FILE_SIZE = 1
MAX_FILE_SIZE = 500
MIN_NESTED_DEPTH = 2  # Actual nesting starts at 2 (e.g., /a/b)
MAX_NESTED_DEPTH = 5  # Test deeper paths
DEFAULT_NUM_EXAMPLES = 100
DEFAULT_TARGETED_PROB = 0.25  # Fraction of examples that are targeted edge cases
FUSE_MOUNT_TIMEOUT = 30  # seconds to wait for FUSE mount
BATCH_SIZE = 50  # Process in batches for better memory management
CONTENT_CACHE_SIZE = 500  # Number of content samples to cache from Pile

# Sanitization helpers
ALLOWED_CONTROL_CODES = {9, 10, 13}


def sanitize_text_block(text: str | None) -> str:
    """Remove NULs and disallowed control characters while preserving layout."""
    if not text:
        return ""

    sanitized_chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0:
            continue
        if code < 32 and code not in ALLOWED_CONTROL_CODES:
            continue
        sanitized_chars.append(ch)
    return ''.join(sanitized_chars)


def sanitize_tree_string(tree: str | None) -> str:
    """Sanitize a filesystem tree representation."""
    if not tree:
        return ""
    return sanitize_text_block(tree).strip()


def sanitize_read_response(text: str | None) -> str:
    """Sanitize file content responses while preserving leading/trailing whitespace."""
    if text is None:
        return ""
    return sanitize_text_block(text)


def normalize_readdir_entries(entries: list[str]) -> list[str]:
    """Ensure directory listings are deduped, sorted, and include dot entries."""
    seen = set()
    normalized: list[str] = []

    # Always ensure '.' and '..' appear first exactly once
    for special in ('.', '..'):
        if special not in seen:
            normalized.append(special)
            seen.add(special)

    others = []
    for entry in entries:
        if entry in ('.', '..'):
            continue
        if entry in seen:
            continue
        seen.add(entry)
        others.append(entry)

    normalized.extend(sorted(others))
    return normalized


class OperationSelectionError(Exception):
    """Raised when an operation cannot be generated for the current filesystem state."""


class DeterministicContentGenerator:
    """Generate deterministic content for files based on filepath hash."""
    
    def __init__(self):
        # Content generation vocabulary (same as before)
        self.nouns = ['server', 'client', 'database', 'user', 'session', 'config', 'cache', 'token', 'request', 'response', 'data', 'file', 'process', 'thread', 'connection', 'query', 'result', 'error', 'warning', 'message']
        self.adjectives = ['fast', 'slow', 'new', 'old', 'active', 'inactive', 'valid', 'invalid', 'critical', 'important', 'temporary', 'permanent', 'secure', 'unsafe', 'optimized', 'legacy']
        self.verbs = ['start', 'stop', 'create', 'delete', 'update', 'fetch', 'send', 'receive', 'process', 'validate', 'initialize', 'terminate', 'connect', 'disconnect', 'load', 'save']
        self.log_levels = ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'TRACE']
        self.func_names = ['main', 'init', 'process', 'handle', 'get', 'set', 'update', 'delete', 'validate', 'parse', 'format', 'convert', 'transform', 'calculate', 'compute']
        self.var_names = ['data', 'result', 'value', 'count', 'index', 'temp', 'buffer', 'cache', 'config', 'settings', 'params', 'args', 'status', 'state', 'info']
    
    def _get_seeded_random(self, filepath: str) -> random.Random:
        """Get a seeded Random instance based on filepath hash."""
        hash_val = hashlib.sha256(filepath.encode()).hexdigest()
        seed = int(hash_val[:16], 16)  # Use first 16 hex chars for seed
        return random.Random(seed)
    
    def _generate_code_content(self, rng: random.Random) -> str:
        """Generate deterministic code content using seeded RNG."""
        lang = rng.choice(['python', 'javascript', 'bash', 'go', 'rust'])
        
        if lang == 'python':
            func = rng.choice(self.func_names)
            var1 = rng.choice(self.var_names)
            var2 = rng.choice(self.var_names)
            return f'''import os\nimport sys\nimport {rng.choice(['json', 'time', 'logging', 'requests', 'argparse'])}\n\ndef {func}({var1}):\n    """{rng.choice(self.verbs).capitalize()} the {var1}."""\n    {var2} = {rng.randint(0, 100)}\n    if {var1} is None:\n        raise ValueError("{var1} cannot be None")\n    return {var2}\n\nif __name__ == "__main__":\n    result = {func}({rng.randint(1, 50)})\n    print(f"Result: {{result}}")'''
        
        elif lang == 'javascript':
            func = rng.choice(self.func_names)
            var1 = rng.choice(self.var_names)
            return f'''const {var1} = {rng.randint(0, 100)};\n\nfunction {func}(input) {{\n  if (!input) {{\n    throw new Error('Input required');\n  }}\n  return input * {rng.randint(2, 10)};\n}}\n\nmodule.exports = {{ {func} }};'''
        
        elif lang == 'bash':
            var = rng.choice(self.var_names).upper()
            return f'''#!/bin/bash\nset -euo pipefail\n\n{var}="{rng.choice(self.nouns)}"\necho "Starting ${{SCRIPT_NAME}}"\n\nif [ -z "${{1:-}}" ]; then\n  echo "Usage: $0 <arg>"\n  exit 1\nfi\n\necho "Processing: $1"\nexit 0'''
        
        elif lang == 'go':
            func = rng.choice(self.func_names).capitalize()
            return f'''package main\n\nimport (\n\t"fmt"\n\t"log"\n)\n\nfunc {func}(val int) (int, error) {{\n\tif val < 0 {{\n\t\treturn 0, fmt.Errorf("invalid value: %d", val)\n\t}}\n\treturn val * {rng.randint(2, 10)}, nil\n}}\n\nfunc main() {{\n\tresult, err := {func}({rng.randint(1, 50)})\n\tif err != nil {{\n\t\tlog.Fatal(err)\n\t}}\n\tfmt.Println(result)\n}}'''
        
        else:  # rust
            func = rng.choice(self.func_names)
            return f'''fn {func}(x: i32) -> Result<i32, String> {{\n    if x < 0 {{\n        return Err(format!("Invalid input: {{}}", x));\n    }}\n    Ok(x * {rng.randint(2, 10)})\n}}\n\nfn main() {{\n    match {func}({rng.randint(1, 50)}) {{\n        Ok(result) => println!("Result: {{}}", result),\n        Err(e) => eprintln!("Error: {{}}", e),\n    }}\n}}'''
    
    def _generate_text_content(self, rng: random.Random) -> str:
        """Generate deterministic text content using seeded RNG."""
        content_type = rng.choice(['article', 'log', 'note', 'documentation'])
        
        if content_type == 'article':
            topic = rng.choice(self.nouns)
            return f'''# {topic.capitalize()}\n\nThe {topic} system provides {rng.choice(self.adjectives)} functionality for handling {rng.choice(['operations', 'requests', 'data', 'processes'])}.\n\n## Overview\n\nThis document describes the {topic} implementation and its key features.\n\n## Key Features\n\n- {rng.choice(self.verbs).capitalize()} operations\n- {rng.choice(self.adjectives).capitalize()} performance\n- Support for {rng.choice(['async', 'sync', 'batch', 'streaming'])} processing\n\n## Usage\n\nTo use the {topic}, follow these steps:\n\n1. Initialize the {topic}\n2. Configure parameters\n3. Execute operations\n4. Handle results'''
        
        elif content_type == 'log':
            lines = []
            for i in range(rng.randint(5, 15)):
                level = rng.choice(self.log_levels)
                timestamp = f"2025-01-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
                action = rng.choice(self.verbs)
                noun = rng.choice(self.nouns)
                lines.append(f"[{level}] {timestamp} - {action.capitalize()} {noun} (id={rng.randint(1000, 9999)})")
            return '\n'.join(lines)
        
        elif content_type == 'note':
            return f'''TODO: {rng.choice(self.verbs).capitalize()} the {rng.choice(self.nouns)}\n\nNotes:\n- {rng.choice(self.adjectives).capitalize()} approach needed\n- Consider {rng.choice(['performance', 'security', 'scalability', 'maintainability'])}\n- Status: {rng.choice(['in progress', 'pending review', 'completed', 'blocked'])}\n- Priority: {rng.choice(['high', 'medium', 'low'])}'''
        
        else:  # documentation
            component = rng.choice(self.nouns)
            return f'''## {component.capitalize()} API\n\n### Methods\n\n#### {rng.choice(self.func_names)}()\n\n{rng.choice(self.verbs).capitalize()}s the {component}.\n\n**Parameters:**\n- `{rng.choice(self.var_names)}`: Input {rng.choice(self.nouns)}\n- `{rng.choice(['timeout', 'retries', 'verbose'])}`: Optional {rng.choice(['int', 'bool', 'str'])}\n\n**Returns:** {rng.choice(['bool', 'str', 'dict', 'list'])}\n\n**Example:**\n```\n{rng.choice(self.func_names)}({rng.choice(self.var_names)}={rng.randint(1, 100)})\n```'''
    
    def _generate_structured_content(self, rng: random.Random) -> str:
        """Generate deterministic structured content using seeded RNG."""
        format_type = rng.choice(['json', 'yaml', 'config', 'env'])
        
        if format_type == 'json':
            key1 = rng.choice(self.var_names)
            key2 = rng.choice(self.nouns)
            return f'''{{\n  "{key1}": "{rng.choice(self.adjectives)}-{rng.choice(self.nouns)}",\n  "{key2}": {rng.randint(1, 1000)},\n  "enabled": {rng.choice(['true', 'false'])},\n  "options": [{rng.randint(1, 10)}, {rng.randint(10, 100)}],\n  "{rng.choice(['timestamp', 'id', 'version'])}": "{rng.randint(1000000, 9999999)}"\n}}'''
        
        elif format_type == 'yaml':
            key1 = rng.choice(self.var_names)
            key2 = rng.choice(self.nouns)
            return f'''{key1}: {rng.choice(self.adjectives)}-{rng.choice(self.nouns)}\n{key2}:\n  {rng.choice(self.var_names)}: {rng.randint(1, 1000)}\n  enabled: {rng.choice(['true', 'false'])}\n  {rng.choice(['timeout', 'retries', 'workers'])}: {rng.randint(1, 100)}'''
        
        elif format_type == 'config':
            return f'''# Configuration\n{rng.choice(self.nouns)}_{rng.choice(self.var_names)}={rng.randint(1000, 9999)}\n{rng.choice(['debug', 'verbose', 'strict'])}={rng.choice(['true', 'false'])}\n{rng.choice(['host', 'port', 'path'])}={rng.choice(['localhost', '0.0.0.0', '/tmp/data'])}\n{rng.choice(['timeout', 'retries', 'workers'])}={rng.randint(1, 100)}'''
        
        else:  # env
            var1 = rng.choice(self.nouns).upper()
            var2 = rng.choice(self.var_names).upper()
            return f'''{var1}_HOST={rng.choice(['localhost', '127.0.0.1', '0.0.0.0'])}\n{var1}_PORT={rng.randint(3000, 9000)}\n{var2}_KEY={rng.choice(self.adjectives)}{rng.randint(1000, 9999)}\n{var2}_SECRET={rng.randint(10000000, 99999999)}\nENVIRONMENT={rng.choice(['development', 'staging', 'production'])}'''
    
    def get_content_for_file(self, filepath: str, max_size: int = MAX_FILE_SIZE) -> str:
        """Get deterministic content for a file based on filepath and extension."""
        # Get seeded RNG based on filepath
        rng = self._get_seeded_random(filepath)
        
        ext = os.path.splitext(filepath)[1].lower() if filepath else ''
        
        # Choose content type based on extension
        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.sh']:
            content = self._generate_code_content(rng)
        elif ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.conf', '.cfg', '.config']:
            content = self._generate_structured_content(rng)
        else:
            content = self._generate_text_content(rng)
        
        # Truncate to max_size if needed
        if len(content) > max_size:
            truncated = content[:max_size]
            last_newline = truncated.rfind('\n')
            if last_newline > max_size * 0.7:
                content = truncated[:last_newline]
            else:
                content = truncated
        
        return content


# Global generator instance
_content_generator: Optional[DeterministicContentGenerator] = None


def get_content_generator() -> DeterministicContentGenerator:
    """Get or create the global content generator."""
    global _content_generator
    if _content_generator is None:
        _content_generator = DeterministicContentGenerator()
    return _content_generator

# CORE FUSE operations we want to train on (filtering out noise)
# Includes both state-changing and query operations for complete filesystem functionality
USEFUL_FUSE_OPERATIONS = {
    'getattr',    # Get file/directory attributes (stat-like) - QUERY
    'readdir',    # List directory contents - QUERY  
    'read',       # Read file contents - QUERY
    'mkdir',      # Create directory - STATE CHANGE
    'rmdir',      # Remove directory - STATE CHANGE
    'create',     # Create file - STATE CHANGE
    'write',      # Write to file - STATE CHANGE
    'unlink',     # Remove file - STATE CHANGE
    'chmod',      # Change permissions - STATE CHANGE
    'chown',      # Change ownership - STATE CHANGE
    'truncate',   # Change file size - STATE CHANGE
    'rename',     # Rename/move file - STATE CHANGE
    'symlink',    # Create symbolic link - STATE CHANGE
    'link',       # Create hard link - STATE CHANGE
}

# Operations to EXCLUDE (too low-level or not useful for training)
EXCLUDED_FUSE_OPERATIONS = {
    'release',    # File handle cleanup
    'flush',      # Flush file buffers
    'fsync',      # Sync file to disk
    'utimens',    # Update timestamps
    'access',     # Check file access permissions
    'open',       # Open file handle
    'opendir',    # Open directory handle
    'releasedir', # Close directory handle
    'statfs',     # Get filesystem statistics
    'getxattr',   # Get extended attributes
    'setxattr',   # Set extended attributes
    'listxattr',  # List extended attributes
    'removexattr', # Remove extended attributes
}

# Balanced operation weights for random selection
# Now includes read/write operations for complete filesystem functionality
OPERATION_WEIGHTS = {
    'mkdir': 0.12,      # Directory creation
    'touch': 0.12,      # File creation (create)
    'write': 0.12,      # Write to file
    'read': 0.12,       # Read from file
    'rm': 0.10,         # File removal (unlink)
    'rmdir': 0.10,      # Directory removal
    'chmod': 0.08,      # Permission changes
    'chown': 0.08,      # Ownership changes
    'ls': 0.08,         # Directory listing (readdir)
    'stat': 0.08,       # File information (getattr)
    'truncate': 0.06,   # Truncate file
    'rename': 0.06,     # Rename/move file
    'symlink': 0.06,    # Create symbolic link
}

# Content templates are now generated dynamically based on file extensions

def check_fuse_support() -> None:
    """
    Check if FUSE is supported on this platform.
    
    Raises:
        SystemExit: If FUSE is not supported or not available
    """
    system = platform.system()
    
    if system == "Darwin":
        print("ERROR: This script requires FUSE support and cannot run natively on macOS.")
        print("Please use Docker instead:")
        print("  docker-compose run --rm --privileged datagen-fuse python -m train.generate_data [options]")
        sys.exit(1)
    
    if system != "Linux":
        print(f"ERROR: Unsupported platform: {system}")
        print("This script requires Linux with FUSE support.")
        print("Please use Docker:")
        print("  docker-compose run --rm --privileged datagen-fuse python -m train.generate_data [options]")
        sys.exit(1)
    
    # Check if FUSE is available
    if not shutil.which('fusermount'):
        print("ERROR: FUSE is not installed or fusermount is not available.")
        print("Please install FUSE or run in Docker:")
        print("  docker-compose run --rm --privileged datagen-fuse python -m train.generate_data [options]")
        sys.exit(1)

def generate_random_filename() -> str:
    """Generate more realistic filenames."""
    prefixes = ['config', 'data', 'log', 'temp', 'backup', 'script', 'doc', 'test']
    suffixes = ['txt', 'log', 'json', 'py', 'sh', 'cfg', 'conf', 'dat', 'tmp']
    
    # Sometimes add descriptive middle part
    if random.random() < 0.3:
        middle_parts = ['server', 'client', 'main', 'utils', 'helper', 'core']
        middle = f"_{random.choice(middle_parts)}"
    else:
        middle = ""
    
    return f"{random.choice(prefixes)}{middle}{random.randint(1, 999)}.{random.choice(suffixes)}"

def generate_random_dirname() -> str:
    """Generate more realistic directory names."""
    names = [
        'src', 'lib', 'bin', 'etc', 'var', 'tmp', 'opt', 'usr',
        'docs', 'tests', 'scripts', 'config', 'data', 'logs',
        'backup', 'cache', 'build', 'dist', 'assets', 'resources'
    ]
    
    # Sometimes add version or descriptive suffix
    if random.random() < 0.2:
        suffix = f"_{random.choice(['v1', 'v2', 'old', 'new', 'temp'])}"
    else:
        suffix = ""
    
    return f"{random.choice(names)}{suffix}{random.randint(1, 99) if random.random() < 0.5 else ''}"

def generate_deterministic_content(filepath: str) -> str:
    """Generate deterministic content based on filepath hash."""
    generator = get_content_generator()
    return generator.get_content_for_file(filepath, max_size=MAX_FILE_SIZE)

def get_random_existing_path(mount_dir: str) -> Optional[str]:
    """Get a random existing file or directory path."""
    all_paths = []
    try:
        for root, dirs, files in os.walk(mount_dir):
            # Skip special directories
            dirs[:] = [d for d in dirs if d not in {'dev', 'proc', 'sys'}]
            
            for d in dirs:
                full_path = os.path.join(root, d)
                rel_path = os.path.relpath(full_path, mount_dir)
                if rel_path != '.':
                    all_paths.append(rel_path)
                    
            for f in files:
                # Skip special files
                if f in {'llm', '.fuse_hidden'}:
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, mount_dir)
                all_paths.append(rel_path)
    except (OSError, PermissionError):
        pass
    
    return random.choice(all_paths) if all_paths else None


def get_random_existing_file(mount_dir: str) -> Optional[str]:
    """Get a random existing file path."""
    files: list[str] = []
    try:
        for root, _, filenames in os.walk(mount_dir):
            rel_root = os.path.relpath(root, mount_dir)
            for name in filenames:
                if name in {'llm', '.fuse_hidden'}:
                    continue
                rel_path = os.path.normpath(os.path.join(rel_root, name))
                if rel_path == '.':
                    continue
                files.append(rel_path)
    except (OSError, PermissionError):
        pass
    return random.choice(files) if files else None


def get_random_directory(mount_dir: str, require_children: bool = False) -> Optional[str]:
    """Select a random directory, optionally requiring at least one child entry."""
    candidates: list[str] = []
    try:
        for root, dirs, files in os.walk(mount_dir):
            # Skip special directories at the first level
            dirs[:] = [d for d in dirs if d not in {'dev', 'proc', 'sys'}]

            rel_root = os.path.relpath(root, mount_dir)
            has_children = bool(dirs or files)
            if require_children and not has_children:
                continue
            # Always allow root directory ('.')
            candidates.append(rel_root)
    except (OSError, PermissionError):
        pass

    return random.choice(candidates) if candidates else None


def ensure_minimum_regular_files(mount_dir: str, min_files: int = 1) -> None:
    """Ensure the filesystem contains at least `min_files` regular files with deterministic content."""
    current_files = 0
    try:
        for _, _, filenames in os.walk(mount_dir):
            current_files += len([f for f in filenames if f not in {'llm', '.fuse_hidden'}])
            if current_files >= min_files:
                return
    except (OSError, PermissionError):
        pass

    files_needed = max(0, min_files - current_files)
    for _ in range(files_needed):
        for _ in range(10):
            filename = generate_random_filename()
            abs_path = os.path.join(mount_dir, filename)
            if os.path.exists(abs_path):
                continue
            content = generate_deterministic_content(filename)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            break


def ensure_directory_coverage(mount_dir: str, min_children: int = 2) -> None:
    """
    Ensure there is at least one empty directory and one populated directory.
    This avoids generating readdir examples that list entries absent from the prompt.
    """
    empty_directory_present = False
    populated_directory_present = False
    try:
        for root, dirs, files in os.walk(mount_dir):
            dirs[:] = [d for d in dirs if d not in {'dev', 'proc', 'sys'}]
            rel_root = os.path.relpath(root, mount_dir)
            if rel_root == '.':
                rel_root = ''

            has_children = bool(dirs or files)
            if rel_root:
                if has_children:
                    populated_directory_present = True
                else:
                    empty_directory_present = True

            if empty_directory_present and populated_directory_present:
                return
    except (OSError, PermissionError):
        pass

    if not empty_directory_present:
        for _ in range(10):
            dirname = generate_random_dirname()
            abs_dir = os.path.join(mount_dir, dirname)
            if os.path.exists(abs_dir):
                continue
            os.makedirs(abs_dir, exist_ok=True)
            empty_directory_present = True
            break

    if not populated_directory_present:
        for _ in range(10):
            parent = generate_random_dirname()
            abs_dir = os.path.join(mount_dir, parent)
            if os.path.exists(abs_dir):
                continue
            os.makedirs(abs_dir, exist_ok=True)
            child_count = max(1, min_children)
            for _ in range(child_count):
                child_name = generate_random_filename()
                rel_child = os.path.join(parent, child_name)
                child_path = os.path.join(mount_dir, rel_child)
                content = generate_deterministic_content(rel_child)
                with open(child_path, 'w', encoding='utf-8') as fh:
                    fh.write(content)
            populated_directory_present = True
            break


def reset_mount_directory(mount_dir: str) -> None:
    """Remove all entries inside the mounted filesystem (except special dirs)."""
    try:
        for entry in os.scandir(mount_dir):
            if entry.name in {'dev', 'proc', 'sys'}:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    os.unlink(entry.path)
            except Exception:
                # Ignore cleanup errors; subsequent operations can still proceed
                pass
    except (FileNotFoundError, PermissionError):
        pass

def generate_random_operation(mount_dir: str) -> Tuple[str, str]:
    """
    Generate a random filesystem operation command.
    
    Args:
        mount_dir: Root mount directory path
        
    Returns:
        Tuple of (operation_type, command_string)
    """
    operation = random.choices(
        list(OPERATION_WEIGHTS.keys()), 
        weights=list(OPERATION_WEIGHTS.values())
    )[0]
    
    match operation:
        case 'mkdir':
            dirname = generate_random_dirname()
            # Sometimes create nested directories
            if random.random() < 0.3:
                depth = random.randint(MIN_NESTED_DEPTH, MAX_NESTED_DEPTH)
                nested_parts = [generate_random_dirname() for _ in range(depth)]
                dirname = os.path.join(*nested_parts)
            return 'mkdir', f'mkdir -p {shlex.quote(dirname)}'
            
        case 'touch':
            filename = generate_random_filename()
            # Sometimes create in subdirectory
            if random.random() < 0.3:
                existing_path = get_random_existing_path(mount_dir)
                if existing_path and os.path.isdir(os.path.join(mount_dir, existing_path)):
                    filename = os.path.join(existing_path, filename)
            return 'touch', f'touch {shlex.quote(filename)}'
            
        case 'rm':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path and os.path.isfile(os.path.join(mount_dir, existing_path)):
                return 'rm', f'rm {shlex.quote(existing_path)}'
            # Fallback to creating and removing a file
            filename = generate_random_filename()
            return 'rm', f'touch {shlex.quote(filename)} && rm {shlex.quote(filename)}'
            
        case 'rmdir':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path and os.path.isdir(os.path.join(mount_dir, existing_path)):
                # Check if directory is empty
                full_path = os.path.join(mount_dir, existing_path)
                try:
                    if not os.listdir(full_path):
                        return 'rmdir', f'rmdir {shlex.quote(existing_path)}'
                except (OSError, PermissionError):
                    pass
            # Fallback to creating and removing an empty directory
            dirname = generate_random_dirname()
            return 'rmdir', f'mkdir {shlex.quote(dirname)} && rmdir {shlex.quote(dirname)}'
            
        case 'chmod':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                # Use common permission modes
                modes = ['644', '755', '600', '700', '664', '775']
                mode = random.choice(modes)
                return 'chmod', f'chmod {mode} {shlex.quote(existing_path)}'
            # Fallback
            filename = generate_random_filename()
            return 'chmod', f'touch {shlex.quote(filename)} && chmod 644 {shlex.quote(filename)}'
            
        case 'chown':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                # Get current user for realistic chown
                import getpass
                current_user = getpass.getuser()
                return 'chown', f'chown {current_user}:{current_user} {shlex.quote(existing_path)}'
            # Fallback
            filename = generate_random_filename()
            import getpass
            current_user = getpass.getuser()
            return 'chown', f'touch {shlex.quote(filename)} && chown {current_user}:{current_user} {shlex.quote(filename)}'
            

        case 'truncate':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path and os.path.isfile(os.path.join(mount_dir, existing_path)):
                # Truncate to random size (0 to 100 bytes)
                size = random.randint(0, 100)
                return 'truncate', f'truncate -s {size} {shlex.quote(existing_path)}'
            # Fallback: ONLY truncate existing files
            # If no files exist, try mkdir instead
            return 'mkdir', f'mkdir -p {shlex.quote(generate_random_dirname())}'
            
        case 'rename':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                # Generate new name based on file type
                if os.path.isfile(os.path.join(mount_dir, existing_path)):
                    new_name = generate_random_filename()
                else:
                    new_name = generate_random_dirname()
                return 'rename', f'mv {shlex.quote(existing_path)} {shlex.quote(new_name)}'
            # Fallback: ONLY rename existing files
            # If no files exist, try touch instead
            filename = generate_random_filename()
            return 'touch', f'touch {shlex.quote(filename)}'
            
        case 'symlink':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                link_name = f"link_to_{generate_random_filename()}"
                return 'symlink', f'ln -s {shlex.quote(existing_path)} {shlex.quote(link_name)}'
            # Fallback: ONLY symlink to existing files
            # If no files exist, try mkdir instead
            dirname = generate_random_dirname()
            return 'mkdir', f'mkdir -p {shlex.quote(dirname)}'
            
        case 'ls':
            # Prefer directories that already contain entries to reinforce correct listings
            existing_dir = get_random_directory(mount_dir, require_children=True)
            if existing_dir is None and random.random() < 0.5:
                existing_dir = get_random_directory(mount_dir, require_children=False)

            if existing_dir and existing_dir != '.':
                return 'ls', f'ls -la {shlex.quote(existing_dir)}'
            return 'ls', 'ls -la .'
            
        case 'stat':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                return 'stat', f'stat {shlex.quote(existing_path)}'
            # Fallback to stat root
            return 'stat', 'stat .'
        
        case 'write':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path and os.path.isfile(os.path.join(mount_dir, existing_path)):
                # Write to existing file with deterministic content
                content = generate_deterministic_content(existing_path)
                return 'write', f'echo {shlex.quote(content)} > {shlex.quote(existing_path)}'
            else:
                # Create new file and write to it with deterministic content
                filename = generate_random_filename()
                content = generate_deterministic_content(filename)
                return 'write', f'echo {shlex.quote(content)} > {shlex.quote(filename)}'
        
        case 'read':
            existing_file = get_random_existing_file(mount_dir)
            if existing_file and os.path.isfile(os.path.join(mount_dir, existing_file)):
                return 'read', f'cat {shlex.quote(existing_file)}'
            raise OperationSelectionError("No readable file available in current state")
            
        case _:
            # Default fallback
            return 'touch', f'touch {generate_random_filename()}'


def generate_targeted_operation(mount_dir: str) -> Tuple[str, str]:
    """
    Generate a targeted operation to reinforce known failure modes.

    Scenarios:
      - empty root readdir → expects [".", ".."]
      - empty subdir readdir → expects [".", ".."]
      - populated root/subdir readdir → ensure entries are preserved
      - nested mkdir chain → deep path creation
      - rename/symlink simple text file → no binary/NUL content
      - write deterministic text log (avoid NULs)
      - read immediately after deterministic write

    Returns:
      Tuple (operation_label, shell_command)
    """
    import random, shlex

    scenario = random.choice([
        'nested_mkdir',
        'rename_text',
        'symlink_text',
        'write_text_log',
    ])

    match scenario:
        case 'nested_mkdir':
            # Deeply nested directory creation
            parts = [generate_random_dirname(), generate_random_dirname(), generate_random_dirname()]
            path = "/".join(parts)
            return 'mkdir', f"mkdir -p {shlex.quote(path)}"

        case 'rename_text':
            # Rename/symlink operations should ONLY operate on existing files
            # Fall back to the regular random operation generator which handles this correctly
            return generate_random_operation(mount_dir)

        case 'symlink_text':
            # Rename/symlink operations should ONLY operate on existing files
            # Fall back to the regular random operation generator which handles this correctly
            return generate_random_operation(mount_dir)

        case 'write_text_log':
            fname = generate_random_filename()
            # Deterministic, printable log content (no NULs)
            text = "[INFO] init\n[DEBUG] step1\n[INFO] done"
            cmd = f"printf %s {shlex.quote(text)} > {shlex.quote(fname)}"
            return 'write', cmd

        case _:
            return generate_random_operation(mount_dir)

def execute_command(command: str, mount_dir: str) -> Tuple[bool, str]:
    """
    Execute a shell command in the given directory.
    
    Args:
        command: Shell command to execute
        mount_dir: Directory to execute the command in
        
    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=mount_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def generate_example_in_session(mount_dir: str, log_file: str, targeted_prob: float) -> Dict[str, Any]:
    """Generate a single training example using an existing FUSE session."""
    reset_mount_directory(mount_dir)

    with open(log_file, 'w'):
        pass

    num_setup_ops = random.choices([0, 1, 2, 3], weights=[20, 30, 30, 20])[0]
    for _ in range(num_setup_ops):
        try:
            _, setup_command = generate_random_operation(mount_dir)
        except OperationSelectionError:
            continue
        execute_command(setup_command, mount_dir)

    if random.random() < 0.5:
        filename = generate_random_filename()
        content = generate_deterministic_content(filename)
        file_path = os.path.join(mount_dir, filename)
        try:
            with open(file_path, 'w') as f:
                f.write(content)
        except Exception:
            pass

    ensure_minimum_regular_files(mount_dir, min_files=1)
    ensure_directory_coverage(mount_dir, min_children=2)

    fs_state = FSState(mount_dir)
    fs_state.sync_from_fs()

    max_file_size = 300
    max_tokens = 5000

    estimated_tokens = fs_state.estimate_token_count(include_contents=True, max_file_size=max_file_size)
    if estimated_tokens > max_tokens * 0.6:
        max_file_size = max(100, max_file_size // 2)
        estimated_tokens = fs_state.estimate_token_count(include_contents=True, max_file_size=max_file_size)

    initial_state = sanitize_tree_string(
        fs_state.to_tree_string(include_contents=True, max_file_size=max_file_size)
    )

    with open(log_file, 'w'):
        pass

    use_targeted = random.random() < max(0.0, min(1.0, targeted_prob))
    try:
        if use_targeted:
            operation_type, shell_command = generate_targeted_operation(mount_dir)
        else:
            operation_type, shell_command = generate_random_operation(mount_dir)
    except OperationSelectionError:
        return None

    success, output = execute_command(shell_command, mount_dir)
    if not success:
        return None

    time.sleep(0.01)
    logged_operations = read_operation_log(log_file)

    fuse_operation = None
    if logged_operations:
        main_ops = [op for op in logged_operations if op['operation'] not in ['getattr', 'open']]
        fuse_operation = main_ops[-1] if main_ops else logged_operations[-1]
    if fuse_operation is None:
        return None

    fs_state.sync_from_fs()

    query_operations = {'readdir', 'getattr', 'read'}

    if fuse_operation:
        actual_operation = fuse_operation['operation']
    elif operation_type in ['ls']:
        actual_operation = 'readdir'
    elif operation_type in ['stat']:
        actual_operation = 'getattr'
    else:
        actual_operation = operation_type

    operation_str = shell_command
    if fuse_operation:
        params = fuse_operation.get('parameters', {})
        if params:
            param_str = ", ".join(f"{key}={value}" for key, value in params.items())
            operation_str = f"{fuse_operation['operation']}('{fuse_operation['path']}', {param_str})"
        else:
            operation_str = f"{fuse_operation['operation']}('{fuse_operation['path']}')"

    if actual_operation in query_operations:
        result = generate_query_response(actual_operation, fuse_operation, output, success, fs_state)
        op_type = "query"
    else:
        result = sanitize_tree_string(
            fs_state.to_tree_string(include_contents=True, max_file_size=max_file_size)
        )
        op_type = "state_change"

    total_content = initial_state + operation_str + result
    final_tokens = len(total_content) // 4
    if final_tokens > max_tokens:
        return None

    return {
        'initial_state': initial_state,
        'operation': operation_str,
        'result': result,
        'operation_type': op_type,
        'shell_command': shell_command,
        'fuse_operations': logged_operations
    }


def start_fuse_filesystem(mount_dir: str, log_file: str) -> subprocess.Popen:
    """
    Start the reference FUSE filesystem in a subprocess.
    
    Args:
        mount_dir: Directory to mount the filesystem
        log_file: Path to the operation log file
        
    Returns:
        Subprocess handle for the FUSE filesystem
    """
    import sys
    python_path = sys.executable
    
    # Create a temporary root directory for the loopback filesystem
    root_dir = os.path.join(os.path.dirname(mount_dir), 'fuse_root')
    os.makedirs(root_dir, exist_ok=True)
    
    # Start the reference FUSE filesystem with loopback to root_dir
    cmd = [python_path, '-m', 'train.reference_fuse', mount_dir, root_dir, log_file]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=False
    )
    
    # Wait for filesystem to mount
    for _ in range(FUSE_MOUNT_TIMEOUT * 10):  # Check every 0.1 seconds
        if os.path.ismount(mount_dir):
            break
        time.sleep(0.1)
    else:
        # If mount failed, terminate process and raise error
        process.terminate()
        process.wait()
        raise RuntimeError(f"FUSE filesystem failed to mount at {mount_dir}")
    
    return process

def stop_fuse_filesystem(mount_dir: str, process: subprocess.Popen) -> None:
    """
    Stop the FUSE filesystem and unmount.
    
    Args:
        mount_dir: Directory where filesystem is mounted
        process: Subprocess handle for the FUSE filesystem
    """
    try:
        # Try to unmount gracefully
        subprocess.run(['fusermount', '-u', mount_dir], 
                      capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            # Fallback to umount
            subprocess.run(['umount', mount_dir], 
                          capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Terminate the FUSE process
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def generate_query_response(operation: str, fuse_operation: Optional[Dict[str, Any]], 
                           shell_output: str, success: bool, fs_state: FSState) -> str:
    """
    Generate appropriate response for query operations.
    
    This teaches the LLM to return specific query responses rather than complete filesystem trees.
    
    Args:
        operation: Operation type ('readdir', 'getattr', 'read')
        fuse_operation: FUSE operation details from log  
        shell_output: Shell command output
        success: Whether the operation succeeded
        fs_state: Current filesystem state
        
    Returns:
        Formatted query response appropriate for the operation type
    """
    if not success:
        return f"ERROR: {shell_output}"
    
    path = fuse_operation.get('path', '/') if fuse_operation else '/'
    
    if operation == 'readdir':
        # Return directory contents as a list
        try:
            # Extract directory entries from filesystem state
            entries = ['.', '..']  # Always include parent references
            
            # Get relative path for state lookup
            clean_path = path.strip('/')
            if not clean_path:
                # Root directory - get top-level entries
                for file_path, entry in fs_state.state.items():
                    if '/' not in file_path and file_path:
                        entries.append(entry.name)
            else:
                # Subdirectory - get entries under this path
                prefix = clean_path + '/'
                for file_path, entry in fs_state.state.items():
                    if file_path.startswith(prefix):
                        relative = file_path[len(prefix):]
                        if '/' not in relative:  # Direct child only
                            entries.append(entry.name)
            
            # Return as JSON list for consistency
            entries = normalize_readdir_entries(entries)
            return json.dumps(entries)
            
        except Exception as e:
            return f"ERROR: Failed to read directory {path}: {e}"
    
    elif operation == 'getattr':
        # Return file/directory attributes
        try:
            clean_path = path.strip('/')
            if clean_path in fs_state.state:
                entry = fs_state.state[clean_path]
                attrs = {
                    'mode': entry.mode,
                    'size': entry.size,
                    'type': 'directory' if entry.is_dir else 'file',
                    'mtime': entry.mtime,
                    'owner': entry.owner,
                    'group': entry.group
                }
                return json.dumps(attrs)
            else:
                return "ERROR: No such file or directory"
                
        except Exception as e:
            return f"ERROR: Failed to get attributes for {path}: {e}"
    
    elif operation == 'read':
        # Return file contents
        try:
            clean_path = path.strip('/')
            if clean_path in fs_state.state:
                entry = fs_state.state[clean_path]
                if entry.is_dir:
                    return "ERROR: Is a directory"
                
                # Read actual file content from filesystem
                if fs_state.root_path:
                    full_path = os.path.join(fs_state.root_path, clean_path) if clean_path else fs_state.root_path
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        return sanitize_read_response(content)
                    except (UnicodeDecodeError, IOError) as e:
                        return f"ERROR: Cannot read file {path}: {e}"
                else:
                    return "ERROR: No filesystem root available"
            else:
                return "ERROR: No such file or directory"
                
        except Exception as e:
            return f"ERROR: Failed to read file {path}: {e}"
    
    else:
        # Fallback for unknown query operations
        return sanitize_read_response(shell_output)


def filter_fuse_operations(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter FUSE operations to only include useful ones for training.
    
    Args:
        operations: Raw FUSE operations from log
        
    Returns:
        Filtered list of useful operations
    """
    filtered = []
    for op in operations:
        op_name = op.get('operation', '').lower()
        
        # Skip excluded operations
        if op_name in EXCLUDED_FUSE_OPERATIONS:
            continue
            
        # Only include useful operations
        if op_name in USEFUL_FUSE_OPERATIONS:
            filtered.append(op)
            
    return filtered

def read_operation_log(log_file: str) -> List[Dict[str, Any]]:
    """
    Read and parse the FUSE operation log, filtering to useful operations.
    
    Args:
        log_file: Path to the operation log file
        
    Returns:
        List of filtered operation log entries
    """
    operations = []
    
    # Early return for missing log file
    if not os.path.exists(log_file):
        return filter_fuse_operations([])
    
    # Parse log file
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                operations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Filter to only useful operations
    return filter_fuse_operations(operations)

def generate_one(_worker_id: Optional[int] = None, targeted_prob: float = DEFAULT_TARGETED_PROB) -> Dict[str, Any]:
    """
    Backward-compatible wrapper that generates a single example with its own FUSE mount.
    """
    with tempfile.TemporaryDirectory() as temp_base:
        mount_dir = os.path.join(temp_base, 'mount')
        log_file = os.path.join(temp_base, 'operations.log')

        os.makedirs(mount_dir, exist_ok=True)
        fuse_process = start_fuse_filesystem(mount_dir, log_file)

        try:
            example = None
            attempts = 0
            while example is None and attempts < 10:
                example = generate_example_in_session(mount_dir, log_file, targeted_prob)
                attempts += 1
            if example is None:
                raise RuntimeError("Failed to generate example after multiple retries")
            return example
        finally:
            stop_fuse_filesystem(mount_dir, fuse_process)

def generate_data(num_examples: int = DEFAULT_NUM_EXAMPLES, output_dir: Optional[str] = None, targeted_prob: float = DEFAULT_TARGETED_PROB) -> List[Dict[str, Any]]:
    """
    Generate training data using FUSE operations with operation filtering.
    
    Always saves in HuggingFace JSONL format with deterministic naming.
    
    Args:
        num_examples: Number of training examples to generate
        output_dir: Directory to save output file (if None, returns data without saving)
        
    Returns:
        List of generated training examples (always in structured format)
    """
    print(f"🚀 Generating {num_examples} examples using FUSE operations...")
    print(f"📋 Filtering to useful operations: {', '.join(sorted(USEFUL_FUSE_OPERATIONS))}")
    print(f"🚫 Excluding noisy operations: {', '.join(sorted(EXCLUDED_FUSE_OPERATIONS))}")
    
    examples = []
    operation_counts = {}

    print(f"📝 Generating {num_examples} examples within a shared FUSE session...")

    with tempfile.TemporaryDirectory() as temp_base:
        mount_dir = os.path.join(temp_base, 'mount')
        log_file = os.path.join(temp_base, 'operations.log')

        os.makedirs(mount_dir, exist_ok=True)
        fuse_process = start_fuse_filesystem(mount_dir, log_file)

        try:
            while len(examples) < num_examples:
                try:
                    example = None
                    attempts = 0
                    while example is None and attempts < 10:
                        example = generate_example_in_session(mount_dir, log_file, targeted_prob)
                        attempts += 1
                    if example is None:
                        raise RuntimeError("Failed to generate example after multiple retries")

                    examples.append(example)

                    operation_str = example.get('operation', '')
                    if '(' in operation_str:
                        op_name = operation_str.split('(')[0]
                        operation_counts[op_name] = operation_counts.get(op_name, 0) + 1
                    else:
                        operation_counts['shell_command'] = operation_counts.get('shell_command', 0) + 1

                    print(f"  ✅ Generated example {len(examples)}/{num_examples}")
                except Exception as e:
                    print(f"❌ Failed to generate example {len(examples)+1}: {e}")
        finally:
            stop_fuse_filesystem(mount_dir, fuse_process)
    
    print(f"\n📊 Operation Distribution:")
    for op, count in sorted(operation_counts.items()):
        status = "✅" if op in USEFUL_FUSE_OPERATIONS else "🚫" if op in EXCLUDED_FUSE_OPERATIONS else "❓"
        print(f"  {status} {op}: {count}")
    
    if output_dir:
        # Create output directory and generate deterministic filename
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate deterministic filename: fuse_{num_samples}_{unixtime}.jsonl
        unix_time = int(time.time())
        filename = f"fuse_{num_examples}_{unix_time}.jsonl"
        output_file = output_path / filename
        
        print(f"\n💾 Writing {len(examples)} examples to {output_file}")
        print("📄 Using HuggingFace JSONL format")
        save_hf_format(examples, str(output_file))
        print("✅ Data generation complete!")
        print(f"📄 Output saved to: {output_file}")
    
    return examples

def convert_to_hf_format(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert structured training examples to HuggingFace format.
    
    Implements SMART DUAL TRAINING format:
    - STATE-CHANGING: (State T, operation) → State T+1
    - QUERY: (State T, query) → Specific response
    
    Args:
        examples: List of structured training examples
        
    Returns:
        List of HuggingFace format examples with 'prompt' and 'completion' keys
    """
    hf_examples = []
    
    for example in examples:
        current_state = example['initial_state']
        operation = example['operation']
        op_type = example['operation_type']
        
        if op_type == 'state_change':
            # STATE MONAD: State -> Operation -> State
            prompt = f"<W>\n{operation}\n---\n{current_state}"
        
        else:  # op_type == 'query'
            # STATE QUERY: State -> Query -> Result
            prompt = f"<R>\n{operation}\n---\n{current_state}"
        
        completion = example['result']
        
        hf_examples.append({
            'prompt': prompt,
            'completion': completion
        })
    
    return hf_examples

def save_hf_format(examples: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save examples in HuggingFace JSONL format.
    
    Args:
        examples: List of training examples  
        output_file: Path to output JSONL file
    """
    hf_examples = convert_to_hf_format(examples)
    
    with open(output_file, 'w') as f:
        for example in hf_examples:
            f.write(json.dumps(example) + '\n')

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic FUSE filesystem training data with deterministic output naming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m train.generate_data --num_examples 100 --output_dir /app/data/train
  python -m train.generate_data -n 1000 --output_dir ./data/train
  
Output files are automatically named: fuse_{num_samples}_{unixtime}.jsonl
        """
    )
    
    parser.add_argument(
        '-n', '--num_examples',
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help=f'Number of training examples to generate (default: {DEFAULT_NUM_EXAMPLES})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/app/data/train',
        help='Output directory for generated data (default: /app/data/train)'
    )
    parser.add_argument(
        '--targeted_prob',
        type=float,
        default=DEFAULT_TARGETED_PROB,
        help=f'Probability to generate a targeted edge-case example (default: {DEFAULT_TARGETED_PROB})'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting FUSE data generation...")
    print(f"📊 Samples: {args.num_examples}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📋 Format: HuggingFace JSONL (deterministic naming)")
    print()
    
    # Generate the data
    generate_data(num_examples=args.num_examples, output_dir=args.output_dir, targeted_prob=args.targeted_prob)

if __name__ == "__main__":
    check_fuse_support()
    main() 
