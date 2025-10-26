#!/usr/bin/env python3
"""
LLM-driven FUSE filesystem implementation.

This filesystem uses an LLM to handle all filesystem operations by maintaining
the filesystem state as a text representation and querying the LLM for each operation.
"""

import json
import os
import re
import sys
from datetime import datetime
from errno import EEXIST, ENOENT
from stat import S_IFDIR, S_IFLNK, S_IFREG
from time import time
from typing import Any, Dict, List, Optional, Tuple

from fuse import FUSE, FuseOSError, LoggingMixIn, Operations

from llmfuse.fs_state import FSState, FileEntry
from llmfuse.utils import extract_result_from_llm_output


def get_model_response(*args, **kwargs) -> str:
    """
    Placeholder hook for future LLM integration.

    The production system will wire this up to a real model endpoint. For now,
    we raise to make it explicit that filesystem queries are not implemented.
    """
    raise NotImplementedError("LLM-backed filesystem operations are not yet implemented.")

# Check if FUSE is available
try:
    from fuse import FUSE
    FUSE_AVAILABLE = True
except ImportError:
    FUSE_AVAILABLE = False

class LLMFuse(LoggingMixIn, Operations):
    """
    A FUSE filesystem that uses an LLM to handle all operations.
    
    The filesystem state is maintained as a text representation and
    the LLM is queried for each operation to determine the new state.
    """

    def __init__(self, initial_state: Optional[str] = None, root_owner: str = "root", root_group: str = "root", root_mode: int = 0o755):
        self.fs_state = FSState(".")
        self.fd = 0

        # Initialize with provided state or empty filesystem
        if initial_state:
            self._parse_state_from_string(initial_state)
        else:
            now = time()
            self.fs_state.state = {
                "": FileEntry(
                    name="/",
                    is_dir=True,
                    mode=root_mode,
                    owner=root_owner,
                    group=root_group,
                    mtime=now,
                    size=0,
                )
            }

        # Special handling for /dev/llm
        self.llm_device_content = ""

    def _get_state_string(self) -> str:
        """Get current filesystem state as a tree string."""
        return self.fs_state.to_tree_string()

    def _parse_state_from_string(self, state_str: str) -> None:
        """Parse filesystem state from tree format string into FSState."""
        self.fs_state.state = {}

        if not state_str:
            return

        lines = [line.rstrip("\n") for line in state_str.splitlines() if line.strip()]
        if not lines:
            return

        root_line = lines[0]
        self._process_entry_line(root_line, "")

        stack: List[Tuple[int, str]] = []

        for line in lines[1:]:
            stripped = line.rstrip()
            if not stripped:
                continue

            depth = 0
            idx = 0
            while idx < len(stripped):
                if stripped.startswith("│   ", idx) or stripped.startswith("    ", idx):
                    depth += 1
                    idx += 4
                else:
                    break

            if stripped[idx:].startswith("├── "):
                idx += 4
            elif stripped[idx:].startswith("└── "):
                idx += 4

            entry_line = stripped[idx:]
            parent_path = ""
            while stack and stack[-1][0] >= depth:
                stack.pop()
            if stack:
                parent_path = stack[-1][1]

            rel_path = self._process_entry_line(entry_line, parent_path)
            if rel_path is not None:
                entry = self.fs_state.state.get(rel_path)
                if entry and entry.is_dir:
                    stack.append((depth, rel_path))

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
    """
    Main entry point for the LLM-driven FUSE filesystem.
    
    Args:
        mountpoint: Path where the filesystem should be mounted
    """
    if not FUSE_AVAILABLE:
        print("Error: FUSE is not available. Please install libfuse.")
        print("On Ubuntu/Debian: sudo apt-get install fuse3 libfuse3-dev")
        sys.exit(1)
    
    # Create mountpoint if it doesn't exist
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    print(f"Mounting LLM-driven FUSE filesystem at {mountpoint}")
    print("Press Ctrl+C to unmount")
    
    fs = LLMFuse()
    FUSE(fs, mountpoint, nothreads=True, foreground=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <mountpoint>' % sys.argv[0])
        sys.exit(1)
    
    mountpoint = sys.argv[1]
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    main(mountpoint)
