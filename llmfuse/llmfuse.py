#!/usr/bin/env python3
"""
LLM-driven FUSE filesystem implementation.

This filesystem uses an LLM to handle all filesystem operations by maintaining
the filesystem state as a text representation and querying the LLM for each operation.
"""

import json
import os
import sys
from errno import ENOENT, EEXIST, ENOTDIR, EISDIR, ENOTEMPTY
from stat import S_IFDIR, S_IFREG, S_IFLNK
from time import time
from typing import Dict, List, Any, Optional

from fuse import FUSE, FuseOSError, Operations, LoggingMixIn

from .fs_state import FSState, FileEntry
from src.model import get_model_response

# Check if FUSE is available
try:
    from fuse import FUSE
    FUSE_AVAILABLE = True
except ImportError:
    FUSE_AVAILABLE = False

class LLMFS(LoggingMixIn, Operations):
    """
    A FUSE filesystem that uses an LLM to handle all operations.
    
    The filesystem state is maintained as a text representation and
    the LLM is queried for each operation to determine the new state.
    """

    def __init__(self, initial_state: Optional[str] = None):
        self.fs_state = FSState(".")
        self.fd = 0
        
        # Initialize with empty filesystem if no initial state provided
        if initial_state:
            self._parse_state_from_string(initial_state)
        else:
            # Start with just root directory
            self.fs_state.state = {
                '': FileEntry(
                    name='/',
                    is_dir=True,
                    mode=0o755,
                    owner="user",
                    group="user",
                    mtime=time(),
                    size=0
                )
            }
        
        # Special handling for /dev/llm
        self.llm_device_content = ""

    def _get_state_string(self) -> str:
        """Get current filesystem state as a tree string."""
        return self.fs_state.to_tree_string()

    def _parse_state_from_string(self, state_str: str) -> None:
        """Parse filesystem state from tree format string."""
        self.fs_state.state = {}
        
        if not state_str or not state_str.strip():
            return
        
        lines = state_str.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            # Skip the root directory line (starts with /)
            if line.strip().startswith('/') and 'dir' in line:
                continue
                
            # Extract filename from tree prefixes more robustly
            cleaned = line
            tree_prefixes = ['├── ', '└── ', '│   ', '├──', '└──', '│']
            for prefix in tree_prefixes:
                cleaned = cleaned.replace(prefix, '')
            
            # Split the line to extract filename and metadata
            parts = cleaned.strip().split()
            if not parts:
                continue
                
            filename = parts[0]
            if not filename or filename in ['.', '..'] or filename.startswith('/'):
                continue
            
            # Determine if it's a directory or file
            is_dir = len(parts) > 1 and parts[1] == 'dir'
            
            # Extract mode if available
            mode = 0o755 if is_dir else 0o644
            if len(parts) > 2:
                try:
                    # Handle both octal strings and plain numbers
                    mode_str = parts[2]
                    if mode_str.startswith('0o'):
                        mode = int(mode_str, 8)
                    else:
                        mode = int(mode_str, 8)
                except (ValueError, IndexError):
                    pass
            
            # Extract size if available
            size = 0
            for i, part in enumerate(parts):
                if part.endswith('B') and i > 0:
                    try:
                        size_str = part[:-1]  # Remove 'B'
                        size = int(size_str) if size_str.isdigit() else 0
                    except (ValueError, IndexError):
                        pass
                    break
            
            # Create FileEntry
            entry = FileEntry(
                name=filename,
                is_dir=is_dir,
                mode=mode,
                owner="root",  # Use root instead of user for consistency
                group="root", 
                mtime=time(),
                size=size
            )
            
            # Store with filename as key (simplified - assumes flat structure)
            self.fs_state.state[filename] = entry

    def _query_llm_for_operation(self, operation: str, path: str, **kwargs) -> str:
        """Query the LLM for a filesystem operation."""
        print(f"DEBUG: _query_llm_for_operation called: {operation}({path})")
        current_state = self._get_state_string()
        
        # Format the operation with parameters
        params_str = ""
        if kwargs:
            params_list = []
            for key, value in kwargs.items():
                if isinstance(value, str):
                    params_list.append(f"{key}='{value}'")
                else:
                    params_list.append(f"{key}={value}")
            params_str = ", " + ", ".join(params_list)

        prompt = f"""You are implementing a virtual filesystem backend service. You store filesystem state in context and serve filesystem operations on this virtual data. Given the current virtual filesystem state and an operation, return EXACTLY the new virtual filesystem state.

Current virtual filesystem state:
{current_state}

Operation: {operation}('{path}'{params_str})

CRITICAL REQUIREMENTS:
- Return the COMPLETE virtual filesystem tree in EXACT same format as input
- Use root:root for ownership (not user:user)
- Preserve exact timestamp format: "Jun 14 17:57"
- Include file sizes (e.g., "36 B", "0 B")
- Use exact tree symbols: ├── └── │
- If operation fails (file doesn't exist, etc.), return UNCHANGED state
- NO extra text, NO reasoning, NO <think> tags - ONLY the virtual filesystem tree
- Start your response immediately with "/" (the root directory)

New virtual filesystem state:"""

        try:
            print(f"DEBUG: About to call get_model_response...")
            import time
            start = time.time()
            response = get_model_response(prompt, temperature=0.0)
            elapsed = (time.time() - start) * 1000
            print(f"DEBUG: LLM call completed in {elapsed:.1f}ms")
            return response.strip()
        except Exception as e:
            print(f"DEBUG: LLM call failed with exception: {e}")
            return f"ERROR: LLM query failed: {str(e)}"

    def _handle_llm_response(self, response: str) -> bool:
        """Handle LLM response and update filesystem state."""
        print(f"DEBUG: _handle_llm_response called with response starting: {response[:50]}...")
        if response.startswith("ERROR:"):
            print(f"LLM operation failed: {response}")
            return False
        
        try:
            self._parse_state_from_string(response)
            print("DEBUG: Successfully parsed LLM response")
            return True
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return False

    def getattr(self, path, fh=None):
        """Get file attributes."""
        print(f"DEBUG: getattr called for path: {path}")
        
        # Handle special /dev/llm device
        if path == '/dev/llm':
            now = time()
            return dict(
                st_mode=(S_IFREG | 0o666),
                st_nlink=1,
                st_size=len(self.llm_device_content),
                st_ctime=now,
                st_mtime=now,
                st_atime=now,
                st_uid=os.getuid(),
                st_gid=os.getgid()
            )
        
        # Handle root directory
        if path == '/':
            now = time()
            return dict(
                st_mode=(S_IFDIR | 0o755),
                st_nlink=2,
                st_size=0,
                st_ctime=now,
                st_mtime=now,
                st_atime=now,
                st_uid=os.getuid(),
                st_gid=os.getgid()
            )
        
        # Handle /dev directory
        if path == '/dev':
            now = time()
            return dict(
                st_mode=(S_IFDIR | 0o755),
                st_nlink=2,
                st_size=0,
                st_ctime=now,
                st_mtime=now,
                st_atime=now,
                st_uid=os.getuid(),
                st_gid=os.getgid()
            )
        
        # Query LLM for other paths
        response = self._query_llm_for_operation('getattr', path)
        
        # For getattr, we need to extract attributes from the response
        # This is simplified - in practice, you'd want the LLM to return structured data
        rel_path = path.lstrip('/')
        if rel_path in self.fs_state.state:
            entry = self.fs_state.state[rel_path]
            return dict(
                st_mode=(S_IFDIR if entry.is_dir else S_IFREG) | entry.mode,
                st_nlink=2 if entry.is_dir else 1,
                st_size=entry.size,
                st_ctime=entry.mtime,
                st_mtime=entry.mtime,
                st_atime=entry.mtime,
                st_uid=os.getuid(),
                st_gid=os.getgid()
            )
        
        raise FuseOSError(ENOENT)

    def readdir(self, path, fh):
        """Read directory contents."""
        print(f"DEBUG: readdir called for path: {path}")
        
        # Handle special directories
        if path == '/':
            entries = ['.', '..']
            # Add top-level entries from filesystem state
            for file_path, entry in self.fs_state.state.items():
                if '/' not in file_path and file_path != '':
                    entries.append(entry.name)
            # Always include /dev
            if 'dev' not in entries:
                entries.append('dev')
            return entries
        
        if path == '/dev':
            return ['.', '..', 'llm']
        
        # Query LLM for directory listing
        response = self._query_llm_for_operation('readdir', path)
        
        # Parse directory entries from response
        # This is simplified - the LLM should return a structured list
        entries = ['.', '..']
        
        # Extract entries from current state
        prefix = path.lstrip('/').rstrip('/')
        if prefix:
            prefix += '/'
        
        for file_path, entry in self.fs_state.state.items():
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if '/' not in relative:  # Direct child only
                    entries.append(entry.name)
        
        return entries

    def mkdir(self, path, mode):
        """Create a directory."""
        print(f"DEBUG: mkdir called for path: {path}")
        response = self._query_llm_for_operation('mkdir', path, mode=oct(mode))
        if not self._handle_llm_response(response):
            print(f"DEBUG: mkdir failed, raising EEXIST for {path}")
            raise FuseOSError(EEXIST)
        print(f"DEBUG: mkdir succeeded for {path}")

    def create(self, path, mode):
        """Create a file."""
        print(f"DEBUG: create called for path: {path}")
        response = self._query_llm_for_operation('create', path, mode=oct(mode))
        if not self._handle_llm_response(response):
            raise FuseOSError(EEXIST)
        
        self.fd += 1
        return self.fd

    def unlink(self, path):
        """Remove a file."""
        response = self._query_llm_for_operation('unlink', path)
        if not self._handle_llm_response(response):
            raise FuseOSError(ENOENT)

    def rmdir(self, path):
        """Remove a directory."""
        response = self._query_llm_for_operation('rmdir', path)
        if not self._handle_llm_response(response):
            raise FuseOSError(ENOENT)

    def chmod(self, path, mode):
        """Change file permissions."""
        response = self._query_llm_for_operation('chmod', path, mode=oct(mode))
        if not self._handle_llm_response(response):
            raise FuseOSError(ENOENT)
        return 0

    def chown(self, path, uid, gid):
        """Change file ownership."""
        response = self._query_llm_for_operation('chown', path, uid=uid, gid=gid)
        if not self._handle_llm_response(response):
            raise FuseOSError(ENOENT)

    def open(self, path, flags):
        """Open a file."""
        print(f"DEBUG: open called for path: {path}")
        self.fd += 1
        return self.fd

    def read(self, path, size, offset, fh):
        """Read from a file."""
        if path == '/dev/llm':
            # Return the current LLM device content
            content = self.llm_device_content.encode('utf-8')
            return content[offset:offset + size]
        
        # For other files, this is simplified - in practice, you'd maintain file contents
        return b''

    def write(self, path, data, offset, fh):
        """Write to a file."""
        print(f"DEBUG: write called for path: {path}, data length: {len(data)}")
        if path == '/dev/llm':
            # Handle writes to the LLM device
            question = data.decode('utf-8').strip()
            
            # Query LLM with the current filesystem state
            current_state = self._get_state_string()
            prompt = f"""You are a filesystem assistant. A user has asked a question about the current filesystem state.

Current filesystem state:
{current_state}

User question: {question}

Provide a helpful answer based on the current filesystem state:"""
            
            try:
                answer = get_model_response(prompt, temperature=0.1)
                self.llm_device_content = answer
            except Exception as e:
                self.llm_device_content = f"Error: {str(e)}"
            
            return len(data)
        
        # For other files, this would update file contents
        # This is simplified for now
        return len(data)

    def truncate(self, path, length, fh=None):
        """Truncate a file."""
        # Simplified implementation
        return 0

    def rename(self, old, new):
        """Rename a file or directory."""
        response = self._query_llm_for_operation('rename', old, new_path=new)
        if not self._handle_llm_response(response):
            raise FuseOSError(ENOENT)

    def symlink(self, target, source):
        """Create a symbolic link."""
        response = self._query_llm_for_operation('symlink', source, target=target)
        if not self._handle_llm_response(response):
            raise FuseOSError(EEXIST)
        return 0

    def readlink(self, path):
        """Read the target of a symbolic link."""
        # Simplified implementation
        return ""

    def statfs(self, path):
        """Get filesystem statistics."""
        return dict(f_bsize=512, f_blocks=4096, f_bavail=2048)

def main(mountpoint: str):
    """Run the LLM-driven FUSE filesystem."""
    if not FUSE_AVAILABLE:
        print("Error: FUSE is not available. Please install libfuse.")
        print("On Ubuntu/Debian: sudo apt-get install fuse3 libfuse3-dev")
        sys.exit(1)
    
    fs = LLMFS()
    FUSE(fs, mountpoint, nothreads=True, foreground=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <mountpoint>' % sys.argv[0])
        sys.exit(1)
    
    mountpoint = sys.argv[1]
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    main(mountpoint)
