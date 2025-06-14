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

# Conditional FUSE import - only needed for actual mounting
try:
    from fuse import FUSE, FuseOSError, Operations, LoggingMixIn
    FUSE_AVAILABLE = True
except ImportError:
    # Create dummy classes for evaluation mode
    class Operations:
        pass
    class LoggingMixIn:
        pass
    class FuseOSError(Exception):
        def __init__(self, errno):
            self.errno = errno
            super().__init__(f"FUSE error: {errno}")
    FUSE_AVAILABLE = False

from .fs_state import FSState, FileEntry
from .model import get_model_response

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
        
        lines = state_str.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            # Skip the root directory line (starts with /)
            if line.strip().startswith('/') and 'dir' in line:
                continue
                
            # Extract filename from tree prefixes
            cleaned = line
            for prefix in ['├── ', '└── ', '│   ']:
                cleaned = cleaned.replace(prefix, '')
            
            # Split the line to extract filename and metadata
            parts = cleaned.strip().split()
            if not parts:
                continue
                
            filename = parts[0]
            if not filename or filename in ['.', '..']:
                continue
            
            # Determine if it's a directory or file
            is_dir = len(parts) > 1 and parts[1] == 'dir'
            
            # Extract mode if available
            mode = 0o755 if is_dir else 0o644
            if len(parts) > 2:
                try:
                    mode = int(parts[2], 8)
                except (ValueError, IndexError):
                    pass
            
            # Create FileEntry
            entry = FileEntry(
                name=filename,
                is_dir=is_dir,
                mode=mode,
                owner="user",
                group="user", 
                mtime=time(),
                size=0
            )
            
            # Store with filename as key (simplified - assumes flat structure)
            self.fs_state.state[filename] = entry

    def _query_llm_for_operation(self, operation: str, path: str, **kwargs) -> str:
        """Query the LLM for a filesystem operation."""
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
        
        prompt = f"""You are a filesystem. Given the current state and an operation, return the new filesystem state.

Current filesystem state:
{current_state}

Operation: {operation}('{path}'{params_str})

Return ONLY the new filesystem state as a tree structure. Use the same format as the input state.
If the operation fails (e.g., file doesn't exist, permission denied), return the UNCHANGED filesystem state.

New filesystem state:"""

        try:
            response = get_model_response(prompt, temperature=0.0)
            return response.strip()
        except Exception as e:
            return f"ERROR: LLM query failed: {str(e)}"

    def _handle_llm_response(self, response: str) -> bool:
        """Handle LLM response and update filesystem state."""
        if response.startswith("ERROR:"):
            print(f"LLM operation failed: {response}")
            return False
        
        try:
            self._parse_state_from_string(response)
            return True
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return False

    def getattr(self, path, fh=None):
        """Get file attributes."""
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
        response = self._query_llm_for_operation('mkdir', path, mode=oct(mode))
        if not self._handle_llm_response(response):
            raise FuseOSError(EEXIST)

    def create(self, path, mode):
        """Create a file."""
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
