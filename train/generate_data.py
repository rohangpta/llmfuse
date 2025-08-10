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
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from llmfuse.fs_state import FSState, FileEntry
from llmfuse.utils import DEFAULT_FILE_MODE, DEFAULT_DIR_MODE

# Generation constants
MIN_SETUP_OPERATIONS = 2
MAX_SETUP_OPERATIONS = 8
MIN_FILE_SIZE = 1
MAX_FILE_SIZE = 500
MIN_NESTED_DEPTH = 1
MAX_NESTED_DEPTH = 4
DEFAULT_NUM_EXAMPLES = 100
FUSE_MOUNT_TIMEOUT = 30  # seconds to wait for FUSE mount
BATCH_SIZE = 50  # Process in batches for better memory management

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

def generate_random_content(filename: str = None) -> str:
    """Generate content appropriate for the file extension."""
    
    # If no filename provided, generate generic content
    if not filename:
        return random.choice([
            "# Configuration file\nserver_port=8080\ndebug=true\nlog_level=info",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "user:password:1000:1000:Test User:/home/user:/bin/bash",
            "127.0.0.1 localhost\n192.168.1.1 gateway",
        ])
    
    # Generate content based on file extension
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.json':
        # JSON files get JSON content
        return """{
  "name": "example",
  "version": "1.0.0",
  "description": "Test configuration file"
}"""
    
    elif ext == '.py':
        # Python files get Python content
        return """import os
import sys

def main():
    print('Python script executed')
    return 0

if __name__ == '__main__':
    main()"""
    
    elif ext == '.sh':
        # Shell scripts get shell content
        return """#!/bin/bash
echo 'Shell script executed'
exit 0"""
    
    elif ext in ['.conf', '.cfg', '.config']:
        # Config files get config content
        return """# Configuration file
server_port=8080
debug=true
log_level=info
max_connections=100"""
    
    elif ext in ['.log']:
        # Log files get log content
        return """[INFO] 2025-01-01 10:00:00 - Application started
[DEBUG] 2025-01-01 10:00:01 - Loading configuration
[INFO] 2025-01-01 10:00:02 - Server ready on port 8080"""
    
    elif ext in ['.md']:
        # Markdown files get markdown content
        return """# README

This is a test file for filesystem operations.

## Usage

Run the commands as needed."""
    
    elif ext in ['.yaml', '.yml']:
        # YAML files get YAML content
        return """name: example
version: 1.0.0
description: Test configuration
debug: true"""
    
    else:
        # Text files and others get plain text
        return """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
This is sample text content for testing filesystem operations.
Created for LLMFuse training data generation."""

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
            # Fallback to creating and truncating a file
            filename = generate_random_filename()
            content = generate_random_content(filename)
            size = random.randint(0, 50)
            return 'truncate', f'echo {shlex.quote(content)} > {shlex.quote(filename)} && truncate -s {size} {shlex.quote(filename)}'
            
        case 'rename':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                # Generate new name based on file type
                if os.path.isfile(os.path.join(mount_dir, existing_path)):
                    new_name = generate_random_filename()
                else:
                    new_name = generate_random_dirname()
                return 'rename', f'mv {shlex.quote(existing_path)} {shlex.quote(new_name)}'
            # Fallback to creating and renaming a file
            old_name = generate_random_filename()
            new_name = generate_random_filename()
            return 'rename', f'touch {shlex.quote(old_name)} && mv {shlex.quote(old_name)} {shlex.quote(new_name)}'
            
        case 'symlink':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path:
                link_name = f"link_to_{generate_random_filename()}"
                return 'symlink', f'ln -s {shlex.quote(existing_path)} {shlex.quote(link_name)}'
            # Fallback to creating a file and linking to it
            target = generate_random_filename()
            link_name = f"link_to_{target}"
            return 'symlink', f'touch {shlex.quote(target)} && ln -s {shlex.quote(target)} {shlex.quote(link_name)}'
            
        case 'ls':
            # Sometimes ls a specific directory, sometimes root
            if random.random() < 0.5:
                existing_path = get_random_existing_path(mount_dir)
                if existing_path and os.path.isdir(os.path.join(mount_dir, existing_path)):
                    return 'ls', f'ls -la {shlex.quote(existing_path)}'
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
                # Write to existing file
                content = generate_random_content(existing_path)
                return 'write', f'echo {shlex.quote(content)} > {shlex.quote(existing_path)}'
            else:
                # Create new file and write to it
                filename = generate_random_filename()
                content = generate_random_content(filename)
                return 'write', f'echo {shlex.quote(content)} > {shlex.quote(filename)}'
        
        case 'read':
            existing_path = get_random_existing_path(mount_dir)
            if existing_path and os.path.isfile(os.path.join(mount_dir, existing_path)):
                # Read from existing file
                return 'read', f'cat {shlex.quote(existing_path)}'
            else:
                # Create a file and read from it
                filename = generate_random_filename()
                content = generate_random_content(filename)
                return 'read', f'echo {shlex.quote(content)} > {shlex.quote(filename)} && cat {shlex.quote(filename)}'
            
        case _:
            # Default fallback
            return 'touch', f'touch {generate_random_filename()}'

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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
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
                        return content
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
        return shell_output


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

def generate_one(_worker_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a single training example using FUSE operations.
    
    Implements SMART DUAL TRAINING methodology:
    - STATE-CHANGING operations: (State T, operation) → (State T+1) - Complete filesystem state
    - QUERY operations: (State T, query) → Specific response - Directory contents, file attributes, etc.
    
    This teaches the LLM both state transitions AND proper query responses,
    enabling a fully functional mountable filesystem.
    
    Creates a temporary mount point, starts a reference FUSE filesystem,
    performs random operations, and captures the FUSE operation calls
    for training data.
    
    Args:
        _worker_id: Worker ID for multiprocessing (ignored, for compatibility)
        
    Returns:
        Dictionary containing the training example with keys:
        - 'initial_state': Tree representation of initial filesystem state (State T)
        - 'operation': The FUSE operation that was called
        - 'result': For state-changing: Complete filesystem state (State T+1)
                   For queries: Specific response (directory contents, file attrs, file content)
        - 'operation_type': 'state_change' or 'query'
    """
    with tempfile.TemporaryDirectory() as temp_base:
        mount_dir = os.path.join(temp_base, 'mount')
        log_file = os.path.join(temp_base, 'operations.log')
        
        os.makedirs(mount_dir)
        
        # Start FUSE filesystem
        fuse_process = start_fuse_filesystem(mount_dir, log_file)
        
        try:
            # Clear the log file
            with open(log_file, 'w') as f:
                pass
            
            # Perform minimal setup operations (reduced for speed)
            num_setup_ops = random.randint(1, 3)  # Reduced from 2-8 to 1-3
            for _ in range(num_setup_ops):
                _, setup_command = generate_random_operation(mount_dir)
                execute_command(setup_command, mount_dir)
            
            # Create minimal files with content (reduced for speed)
            if random.random() < 0.5:  # Only 50% chance to create extra files
                filename = generate_random_filename()
                content = generate_random_content(filename)
                file_path = os.path.join(mount_dir, filename)
                try:
                    with open(file_path, 'w') as f:
                        f.write(content)
                except Exception:
                    pass
            
            # Capture initial state with content and size limits
            fs_state = FSState(mount_dir)
            fs_state.sync_from_fs()
            
            # Start with conservative limits and adjust based on size
            max_file_size = 300
            max_tokens = 5000
            
            # Check initial size and adjust if needed
            estimated_tokens = fs_state.estimate_token_count(include_contents=True, max_file_size=max_file_size)
            if estimated_tokens > max_tokens * 0.6:  # If > 60% of limit
                max_file_size = max(100, max_file_size // 2)
                estimated_tokens = fs_state.estimate_token_count(include_contents=True, max_file_size=max_file_size)
            
            initial_state = fs_state.to_tree_string(include_contents=True, max_file_size=max_file_size)
            
            # Clear log before the operation we want to capture
            with open(log_file, 'w') as f:
                pass
            
            # Generate and execute the operation we want to learn
            operation_type, shell_command = generate_random_operation(mount_dir)
            success, output = execute_command(shell_command, mount_dir)
            
            # Read the FUSE operations that were logged
            # Small delay to ensure log is written (reduced from 0.1s)
            time.sleep(0.01)  # 10ms should be sufficient
            logged_operations = read_operation_log(log_file)
            
            # Find the most relevant FUSE operation
            fuse_operation = None
            if logged_operations:
                # Filter out getattr calls and focus on the main operation
                main_ops = [op for op in logged_operations 
                           if op['operation'] not in ['getattr', 'open']]
                if main_ops:
                    fuse_operation = main_ops[-1]  # Take the last main operation
                else:
                    fuse_operation = logged_operations[-1]  # Fallback to last operation
            
            # SMART DUAL TRAINING: Different operations need different responses
            # State-changing ops → Complete filesystem state (State T → State T+1)  
            # Query ops → Specific query responses (readdir contents, getattr data)
            
            fs_state.sync_from_fs()
            
            # Determine operation type based on what the operation actually does
            query_operations = {'readdir', 'getattr', 'read'}
            state_changing_operations = {'mkdir', 'create', 'unlink', 'rmdir', 'chmod', 'chown', 'truncate', 'rename', 'symlink', 'write'}
            
            # Check FUSE operation type if available, otherwise use shell command
            actual_operation = None
            if fuse_operation:
                actual_operation = fuse_operation['operation']
            elif operation_type in ['ls']:
                actual_operation = 'readdir'
            elif operation_type in ['stat']:
                actual_operation = 'getattr'
            else:
                actual_operation = operation_type
            
            if actual_operation in query_operations:
                # QUERY OPERATIONS: Return specific query response
                result = generate_query_response(actual_operation, fuse_operation, output, success, fs_state)
                op_type = "query"
            else:
                # STATE-CHANGING OPERATIONS: Return complete filesystem state with contents
                result = fs_state.to_tree_string(include_contents=True, max_file_size=max_file_size)
                op_type = "state_change"
            
            # Final size check - skip if too large
            total_content = initial_state + operation_str + result
            final_tokens = len(total_content) // 4
            if final_tokens > max_tokens:
                print(f"Skipping example: total size {final_tokens} tokens exceeds limit {max_tokens}")
                # Try generating a new example with different operations
                # For now, return a minimal example
                return {
                    'initial_state': '/ dir 755 user:group Jan 01 00:00',
                    'operation': 'touch(\'empty.txt\')',
                    'result': '/ dir 755 user:group Jan 01 00:00\n└── empty.txt file 644 user:group Jan 01 00:00',
                    'operation_type': 'state_change',
                    'shell_command': 'touch empty.txt',
                    'fuse_operations': [],
                    'skipped_due_to_size': True
                }
            
            # Use FUSE operation if available, otherwise fall back to shell command
            if fuse_operation:
                params = fuse_operation.get('parameters', {})
                param_str = ""
                if params:
                    param_parts = []
                    for key, value in params.items():
                        param_parts.append(f"{key}={value}")
                    param_str = ", " + ", ".join(param_parts)
                
                # Use consistent format without "FUSE" prefix
                operation_str = f"{fuse_operation['operation']}('{fuse_operation['path']}'{param_str})"
            else:
                operation_str = shell_command
            
            return {
                'initial_state': initial_state,
                'operation': operation_str,
                'result': result,
                'operation_type': op_type,
                'shell_command': shell_command,  # Keep for debugging
                'fuse_operations': logged_operations  # Keep for debugging
            }
            
        finally:
            # Always clean up the FUSE filesystem
            stop_fuse_filesystem(mount_dir, fuse_process)

def generate_batch(batch_size: int) -> List[Dict[str, Any]]:
    """
    Generate a batch of examples using a single FUSE mount (performance optimization).
    
    Args:
        batch_size: Number of examples to generate in this batch
        
    Returns:
        List of generated training examples
    """
    examples = []
    
    with tempfile.TemporaryDirectory() as temp_base:
        mount_dir = os.path.join(temp_base, 'mount')
        log_file = os.path.join(temp_base, 'operations.log')
        
        os.makedirs(mount_dir)
        
        # Start FUSE filesystem once for the entire batch
        fuse_process = start_fuse_filesystem(mount_dir, log_file)
        
        try:
            for i in range(batch_size):
                # Clear the log file
                with open(log_file, 'w') as f:
                    pass
                
                # Perform minimal setup operations (reduced for speed)
                num_setup_ops = random.randint(1, 3)  # Reduced from 2-8 to 1-3
                for _ in range(num_setup_ops):
                    _, setup_command = generate_random_operation(mount_dir)
                    execute_command(setup_command, mount_dir)
                
                # Create minimal files with content (reduced for speed)
                if random.random() < 0.5:  # Only 50% chance to create extra files
                    filename = generate_random_filename()
                    content = generate_random_content(filename)
                    file_path = os.path.join(mount_dir, filename)
                    try:
                        with open(file_path, 'w') as f:
                            f.write(content)
                    except Exception:
                        pass
                
                # Capture initial state with content and size limits
                fs_state = FSState(mount_dir)
                fs_state.sync_from_fs()
                
                # Start with conservative limits and adjust based on size
                max_file_size = 300
                max_tokens = 5000
                
                # Check initial size and adjust if needed
                estimated_tokens = fs_state.estimate_token_count(include_contents=True, max_file_size=max_file_size)
                if estimated_tokens > max_tokens * 0.6:  # If > 60% of limit
                    max_file_size = max(100, max_file_size // 2)
                    estimated_tokens = fs_state.estimate_token_count(include_contents=True, max_file_size=max_file_size)
                
                initial_state = fs_state.to_tree_string(include_contents=True, max_file_size=max_file_size)
                
                # Clear log before the operation we want to capture
                with open(log_file, 'w') as f:
                    pass
                
                # Generate and execute the operation we want to learn
                operation_type, shell_command = generate_random_operation(mount_dir)
                success, output = execute_command(shell_command, mount_dir)
                
                # Read the FUSE operations that were logged
                # Small delay to ensure log is written (reduced from 0.1s)
                time.sleep(0.01)  # 10ms should be sufficient
                logged_operations = read_operation_log(log_file)
                
                # Find the most relevant FUSE operation
                fuse_operation = None
                if logged_operations:
                    # Filter out getattr calls and focus on the main operation
                    main_ops = [op for op in logged_operations 
                               if op['operation'] not in ['getattr', 'open']]
                    if main_ops:
                        fuse_operation = main_ops[-1]  # Take the last main operation
                    else:
                        fuse_operation = logged_operations[-1]  # Fallback to last operation
                
                # SMART DUAL TRAINING: Different operations need different responses
                # State-changing ops → Complete filesystem state (State T → State T+1)  
                # Query ops → Specific query responses (readdir contents, getattr data)
                
                fs_state.sync_from_fs()
                
                # Determine operation type based on what the operation actually does
                query_operations = {'readdir', 'getattr', 'read'}
                state_changing_operations = {'mkdir', 'create', 'unlink', 'rmdir', 'chmod', 'chown', 'truncate', 'rename', 'symlink', 'write'}
                
                # Check FUSE operation type if available, otherwise use shell command
                actual_operation = None
                if fuse_operation:
                    actual_operation = fuse_operation['operation']
                elif operation_type in ['ls']:
                    actual_operation = 'readdir'
                elif operation_type in ['stat']:
                    actual_operation = 'getattr'
                else:
                    actual_operation = operation_type
                
                if actual_operation in query_operations:
                    # QUERY OPERATIONS: Return specific query response
                    result = generate_query_response(actual_operation, fuse_operation, output, success, fs_state)
                    op_type = "query"
                else:
                    # STATE-CHANGING OPERATIONS: Return complete filesystem state with contents
                    result = fs_state.to_tree_string(include_contents=True, max_file_size=max_file_size)
                    op_type = "state_change"
                
                # Size check for batch generation 
                total_content = initial_state + (operation_str if 'operation_str' in locals() else shell_command) + result
                final_tokens = len(total_content) // 4
                if final_tokens > max_tokens:
                    print(f"Skipping batch example {i}: size {final_tokens} tokens exceeds limit")
                    continue
                
                # Use FUSE operation if available, otherwise fall back to shell command
                if fuse_operation:
                    params = fuse_operation.get('parameters', {})
                    param_str = ""
                    if params:
                        param_parts = []
                        for key, value in params.items():
                            param_parts.append(f"{key}={value}")
                        param_str = ", " + ", ".join(param_parts)
                    
                    # Use consistent format without "FUSE" prefix
                    operation_str = f"{fuse_operation['operation']}('{fuse_operation['path']}'{param_str})"
                else:
                    operation_str = shell_command
                
                examples.append({
                    'initial_state': initial_state,
                    'operation': operation_str,
                    'result': result,
                    'operation_type': op_type,
                    'shell_command': shell_command,  # Keep for debugging
                    'fuse_operations': logged_operations  # Keep for debugging
                })
                
        finally:
            # Always clean up the FUSE filesystem
            stop_fuse_filesystem(mount_dir, fuse_process)
    
    return examples

def generate_data(num_examples: int = DEFAULT_NUM_EXAMPLES, output_file: Optional[str] = None, hf_format: bool = False) -> List[Dict[str, Any]]:
    """
    Generate training data using FUSE operations with operation filtering.
    
    Args:
        num_examples: Number of training examples to generate
        output_file: Path to output file (if None, returns data without saving)
        hf_format: If True, save in HuggingFace JSONL format; if False, save structured JSON
        
    Returns:
        List of generated training examples (always in structured format)
    """
    print(f"🚀 Generating {num_examples} examples using FUSE operations...")
    print(f"📋 Filtering to useful operations: {', '.join(sorted(USEFUL_FUSE_OPERATIONS))}")
    print(f"🚫 Excluding noisy operations: {', '.join(sorted(EXCLUDED_FUSE_OPERATIONS))}")
    
    # Use batched generation to reuse FUSE mounts (major performance improvement)
    BATCH_SIZE = 20  # Generate 20 examples per FUSE mount
    examples = []
    operation_counts = {}
    
    for batch_start in range(0, num_examples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_examples)
        batch_size = batch_end - batch_start
        
        print(f"📝 Generating batch {batch_start//BATCH_SIZE + 1}/{(num_examples + BATCH_SIZE - 1)//BATCH_SIZE} ({batch_start}-{batch_end-1})...")
        
        try:
            batch_examples = generate_batch(batch_size)
            examples.extend(batch_examples)
            
            # Track operation distribution from the actual operation string
            for example in batch_examples:
                operation_str = example.get('operation', '')
                if '(' in operation_str:
                    op_name = operation_str.split('(')[0]
                    operation_counts[op_name] = operation_counts.get(op_name, 0) + 1
                else:
                    operation_counts['shell_command'] = operation_counts.get('shell_command', 0) + 1
                
        except Exception as e:
            print(f"❌ Failed to generate batch {batch_start//BATCH_SIZE + 1}: {e}")
            # Don't continue if FUSE operations are failing
            raise
    
    print(f"\n📊 Operation Distribution:")
    for op, count in sorted(operation_counts.items()):
        status = "✅" if op in USEFUL_FUSE_OPERATIONS else "🚫" if op in EXCLUDED_FUSE_OPERATIONS else "❓"
        print(f"  {status} {op}: {count}")
    
    if output_file:
        print(f"\n💾 Writing {len(examples)} examples to {output_file}")
        if hf_format:
            print("📄 Using HuggingFace JSONL format")
            save_hf_format(examples, output_file)
        else:
            print("📄 Using structured JSON format")
            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)
        print("✅ Data generation complete!")
    
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
            # STATE-CHANGING OPERATIONS: Return complete filesystem state
            prompt = f"""You are implementing a virtual filesystem backend service. You store filesystem state in context and serve filesystem operations on this virtual data. Given the current virtual filesystem state and an operation, return EXACTLY the new virtual filesystem state.

Current virtual filesystem state:
{current_state}

Operation: {operation}

CRITICAL REQUIREMENTS:
- Return the COMPLETE virtual filesystem tree in EXACT same format as input
- Use root:root for ownership (not user:user)
- Preserve exact timestamp format: "Jun 14 17:57"
- Include file sizes (e.g., "36 B", "0 B")
- Use exact tree symbols: ├── └── │
- If operation fails (file doesn't exist, etc.), return UNCHANGED state
- NO extra text, just the virtual filesystem tree

New virtual filesystem state:"""
        
        else:  # op_type == 'query'
            # QUERY OPERATIONS: Return specific query response
            prompt = f"""You are implementing a virtual filesystem backend service. You store filesystem state in context and serve filesystem operations on this virtual data. Given the current virtual filesystem state and a query operation, return the specific information requested.

Current virtual filesystem state:
{current_state}

Query: {operation}

CRITICAL REQUIREMENTS:
- For readdir: Return JSON list of virtual directory entries: [".", "..", "file1.txt", "file2.txt"]
- For getattr: Return JSON object with virtual file attributes: {{"mode": 644, "size": 1024, "type": "file", ...}}
- For read: Return the virtual file contents as plain text
- If operation fails, return "ERROR: <description>"
- NO extra text, just the requested data

Response:"""
        
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
        description="Generate synthetic FUSE filesystem training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m train.generate_data --num_examples 10 --output test_data.json
  python -m train.generate_data -n 50 -o training_data.json
        """
    )
    
    parser.add_argument(
        '-n', '--num_examples',
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help=f'Number of training examples to generate (default: {DEFAULT_NUM_EXAMPLES})'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='training_data.json',
        help='Output file path (default: training_data.json)'
    )
    
    parser.add_argument(
        '--hf-format',
        action='store_true',
        help='Save in HuggingFace JSONL format instead of structured JSON'
    )
    
    args = parser.parse_args()
    
    # Generate the data
    generate_data(num_examples=args.num_examples, output_file=args.output, hf_format=args.hf_format)

if __name__ == "__main__":
    check_fuse_support()
    main() 