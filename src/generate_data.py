"""
Synthetic filesystem data generation for LLM training using FUSE operations.

This module generates synthetic filesystem operation examples by running
a reference FUSE filesystem, performing operations through shell commands,
and capturing the actual FUSE operation calls for training data.

NOTE: This script requires FUSE support and should be run in Docker on Linux.
On macOS, use: docker-compose run --rm --privileged datagen-fuse python -m src.generate_data
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
from .fs_state import FSState, FileEntry
from .utils import DEFAULT_FILE_MODE, DEFAULT_DIR_MODE

# Generation constants
MIN_SETUP_OPERATIONS = 1
MAX_SETUP_OPERATIONS = 10
MIN_FILE_SIZE = 1
MAX_FILE_SIZE = 1000
MIN_NESTED_DEPTH = 1
MAX_NESTED_DEPTH = 5
DEFAULT_NUM_EXAMPLES = 100
FUSE_MOUNT_TIMEOUT = 10  # seconds to wait for FUSE mount

# Operation weights for random selection
OPERATION_WEIGHTS = {
    'mkdir': 0.15,
    'touch': 0.15,
    'rm': 0.15,
    'rmdir': 0.15,
    'chmod': 0.10,
    'chown': 0.10,
    'ls': 0.10,
    'stat': 0.10,
}

# File content templates
FILE_CONTENT_TEMPLATES = [
    "Hello, world!",
    "This is a test file.",
    "Lorem ipsum dolor sit amet.",
    "#!/bin/bash\necho 'Hello from script'",
    "# Configuration file\nkey=value\n",
    "import os\nprint('Python script')",
]

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
        print("  docker-compose run --rm --privileged datagen-fuse python -m src.generate_data [options]")
        sys.exit(1)
    
    if system != "Linux":
        print(f"ERROR: Unsupported platform: {system}")
        print("This script requires Linux with FUSE support.")
        print("Please use Docker:")
        print("  docker-compose run --rm --privileged datagen-fuse python -m src.generate_data [options]")
        sys.exit(1)
    
    # Check if FUSE is available
    if not shutil.which('fusermount'):
        print("ERROR: FUSE is not installed or fusermount is not available.")
        print("Please install FUSE or run in Docker:")
        print("  docker-compose run --rm --privileged datagen-fuse python -m src.generate_data [options]")
        sys.exit(1)

def generate_random_filename() -> str:
    """
    Generate a random filename.
    
    Returns:
        A random filename string
    """
    prefixes = ['file', 'doc', 'data', 'test', 'temp', 'config']
    suffixes = ['txt', 'log', 'dat', 'cfg', 'tmp', 'py', 'sh']
    return f"{random.choice(prefixes)}{random.randint(1, 999)}.{random.choice(suffixes)}"

def generate_random_dirname() -> str:
    """
    Generate a random directory name.
    
    Returns:
        A random directory name string
    """
    names = ['docs', 'data', 'config', 'temp', 'backup', 'logs', 'scripts', 'tests']
    return f"{random.choice(names)}{random.randint(1, 99)}"

def generate_random_content() -> str:
    """
    Generate random file content.
    
    Returns:
        Random content string for a file
    """
    base_content = random.choice(FILE_CONTENT_TEMPLATES)
    # Sometimes add extra lines
    if random.random() < 0.3:
        extra_lines = [f"Line {i}" for i in range(random.randint(1, 5))]
        return base_content + "\n" + "\n".join(extra_lines)
    return base_content

def get_random_existing_path(mount_dir: str) -> Optional[str]:
    """
    Get a random existing file or directory path from the mount directory.
    
    Args:
        mount_dir: Root mount directory path
        
    Returns:
        Random existing path relative to mount_dir, or None if no paths exist
    """
    all_paths = []
    try:
        for root, dirs, files in os.walk(mount_dir):
            for d in dirs:
                if d == 'dev':  # Skip /dev directory
                    continue
                full_path = os.path.join(root, d)
                rel_path = os.path.relpath(full_path, mount_dir)
                if rel_path != '.':
                    all_paths.append(rel_path)
            for f in files:
                if f == 'llm':  # Skip /dev/llm device
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
    # Import here to avoid circular imports
    import sys
    python_path = sys.executable
    
    # Create a temporary root directory for the loopback filesystem
    root_dir = os.path.join(os.path.dirname(mount_dir), 'fuse_root')
    os.makedirs(root_dir, exist_ok=True)
    
    # Start the reference FUSE filesystem with loopback to root_dir
    cmd = [python_path, '-m', 'src.reference_fuse', mount_dir, root_dir, log_file]
    
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

def read_operation_log(log_file: str) -> List[Dict[str, Any]]:
    """
    Read and parse the FUSE operation log.
    
    Args:
        log_file: Path to the operation log file
        
    Returns:
        List of operation log entries
    """
    operations = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        operations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    
    return operations

def generate_one(_worker_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a single training example using FUSE operations.
    
    Creates a temporary mount point, starts a reference FUSE filesystem,
    performs random operations, and captures the FUSE operation calls
    for training data.
    
    Args:
        _worker_id: Worker ID for multiprocessing (ignored, for compatibility)
        
    Returns:
        Dictionary containing the training example with keys:
        - 'initial_state': Tree representation of initial filesystem state
        - 'operation': The FUSE operation that was called
        - 'result': Either new state tree or operation result
        - 'operation_type': Type of operation (state-changing vs query)
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
            
            # Perform random setup operations
            num_setup_ops = random.randint(MIN_SETUP_OPERATIONS, MAX_SETUP_OPERATIONS)
            for _ in range(num_setup_ops):
                _, setup_command = generate_random_operation(mount_dir)
                execute_command(setup_command, mount_dir)
            
            # Create some files with content
            for _ in range(random.randint(0, 3)):
                filename = generate_random_filename()
                content = generate_random_content()
                file_path = os.path.join(mount_dir, filename)
                try:
                    with open(file_path, 'w') as f:
                        f.write(content)
                except Exception:
                    pass
            
            # Capture initial state
            fs_state = FSState(mount_dir)
            fs_state.sync_from_fs()
            initial_state = fs_state.to_tree_string()
            
            # Clear log before the operation we want to capture
            with open(log_file, 'w') as f:
                pass
            
            # Generate and execute the operation we want to learn
            operation_type, shell_command = generate_random_operation(mount_dir)
            success, output = execute_command(shell_command, mount_dir)
            
            # Read the FUSE operations that were logged
            time.sleep(0.1)  # Give time for log to be written
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
            
            # Determine result based on operation type
            if operation_type in ['ls', 'stat']:
                # Query operations: return the command output
                result = output if success else f"Error: {output}"
                op_type = "query"
                
                # For query operations, use the shell command as the operation
                if fuse_operation:
                    operation_str = f"FUSE {fuse_operation['operation']}('{fuse_operation['path']}')"
                else:
                    operation_str = shell_command
            else:
                # State-changing operations: return the new filesystem state
                fs_state.sync_from_fs()
                result = fs_state.to_tree_string()
                op_type = "state_change"
                
                # For state-changing operations, use the FUSE operation
                if fuse_operation:
                    params = fuse_operation.get('parameters', {})
                    param_str = ""
                    if params:
                        param_parts = []
                        for key, value in params.items():
                            param_parts.append(f"{key}={value}")
                        param_str = ", " + ", ".join(param_parts)
                    
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

def generate_data(num_examples: int = DEFAULT_NUM_EXAMPLES, output_file: Optional[str] = None, hf_format: bool = False) -> List[Dict[str, Any]]:
    """
    Generate training data using FUSE operations.
    
    Args:
        num_examples: Number of training examples to generate
        output_file: Path to output file (if None, returns data without saving)
        hf_format: If True, save in HuggingFace JSONL format; if False, save structured JSON
        
    Returns:
        List of generated training examples (always in structured format)
    """
    print(f"Generating {num_examples} examples using FUSE operations...")
    
    # For now, generate sequentially to avoid FUSE mount conflicts
    # In production, you might want to use a pool of pre-mounted FUSE filesystems
    examples = []
    for i in range(num_examples):
        if i % 10 == 0:
            print(f"Generated {i}/{num_examples} examples...")
        try:
            example = generate_one()
            examples.append(example)
        except Exception as e:
            print(f"Failed to generate example {i}: {e}")
            # Don't continue if FUSE operations are failing
            raise
    
    if output_file:
        print(f"Writing {len(examples)} examples to {output_file}")
        if hf_format:
            print("Using HuggingFace JSONL format")
            save_hf_format(examples, output_file)
        else:
            print("Using structured JSON format")
            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)
        print("Data generation complete!")
    
    return examples

def convert_to_hf_format(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert structured training examples to HuggingFace format.
    
    Args:
        examples: List of structured training examples
        
    Returns:
        List of HuggingFace format examples with 'prompt' and 'completion' keys
    """
    hf_examples = []
    
    for example in examples:
        # Construct the prompt (same logic as in LLMFS)
        current_state = example['initial_state']
        operation = example['operation']
        
        # Extract operation and parameters for better formatting
        if '(' in operation and ')' in operation:
            op_name = operation.split('(')[0]
            params_part = operation[operation.find('(')+1:operation.rfind(')')]
            
            prompt = f"""You are a filesystem. Given the current state and an operation, return EXACTLY the new filesystem state.

Current filesystem state:
{current_state}

Operation: {operation}

CRITICAL REQUIREMENTS:
- Return the COMPLETE filesystem tree in EXACT same format as input
- Use root:root for ownership (not user:user)
- Preserve exact timestamp format: "Jun 14 17:57"
- Include file sizes (e.g., "36 B", "0 B")
- Use exact tree symbols: ├── └── │
- If operation fails (file doesn't exist, etc.), return UNCHANGED state
- NO extra text, just the filesystem tree

New filesystem state:"""
        else:
            # Fallback for malformed operations
            prompt = f"""You are a filesystem. Given the current state and an operation, return the new state.

Current filesystem state:
{current_state}

Operation: {operation}

New filesystem state:"""
        
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
  python -m src.generate_data --num_examples 10 --output test_data.json
  python -m src.generate_data -n 50 -o training_data.json
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