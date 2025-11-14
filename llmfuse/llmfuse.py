#!/usr/bin/env python3
"""
LLM-driven FUSE filesystem implementation.

This filesystem uses an LLM to handle all filesystem operations by maintaining
the filesystem state as a text representation and querying the LLM for each operation.
"""

import json
import os
import sys
from datetime import datetime
from errno import EEXIST, ENOENT, EIO
from stat import S_IFDIR, S_IFLNK, S_IFREG
from time import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fuse import FUSE, FuseOSError, LoggingMixIn, Operations

from eval.runner import build_prompt_with_contract
from llmfuse.fs_state import FSState, FileEntry
from llmfuse.utils import extract_result_from_llm_output


REMOTE_ENDPOINT_ENV = "LLMFUSE_REMOTE_ENDPOINT"
REMOTE_TOKEN_ENV = "LLMFUSE_REMOTE_TOKEN"
REMOTE_TIMEOUT_ENV = "LLMFUSE_REMOTE_TIMEOUT"
REMOTE_RETRY_ENV = "LLMFUSE_REMOTE_RETRIES"
DEFAULT_REMOTE_TIMEOUT = 30
DEFAULT_REMOTE_RETRIES = 1

def _load_local_model_fn():
    try:
        from common.model import get_model_response as local_get_model_response  # type: ignore
        return local_get_model_response
    except Exception:
        return None


_LOCAL_MODEL_FN = _load_local_model_fn()


def _call_remote_model(prompt: str, temperature: float = 0.0) -> str:
    raw_endpoint = os.environ.get(REMOTE_ENDPOINT_ENV)
    if not raw_endpoint:
        raise RuntimeError("LLMFUSE remote endpoint is not configured.")

    endpoint = raw_endpoint.rstrip("/")
    if not endpoint.endswith("/generate"):
        endpoint = f"{endpoint}/generate"

    timeout = float(os.environ.get(REMOTE_TIMEOUT_ENV, DEFAULT_REMOTE_TIMEOUT))
    max_retries = int(os.environ.get(REMOTE_RETRY_ENV, DEFAULT_REMOTE_RETRIES))
    token = os.environ.get(REMOTE_TOKEN_ENV)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-LLMFuse-Token"] = token

            payload = {"prompt": prompt, "temperature": temperature}
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("text", "").strip()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Remote LLM request failed: {last_error}")


def get_model_response(prompt: str, temperature: float = 0.0) -> str:
    """
    Resolve a response using either the remote Modal endpoint or a local model.
    """
    if os.environ.get(REMOTE_ENDPOINT_ENV):
        return _call_remote_model(prompt, temperature=temperature)
    if _LOCAL_MODEL_FN is not None:
        return _LOCAL_MODEL_FN(prompt, temperature=temperature)
    raise RuntimeError(
        "No model backend configured. Set LLMFUSE_REMOTE_ENDPOINT or install local weights."
    )

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
        self.fs_state = FSState(root_path=None)
        self.fd = 0

        # Initialize with provided state or empty filesystem
        if initial_state:
            self._load_state_from_xml(initial_state)
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
        return self.fs_state.to_xml_string(include_contents=True, max_file_size=10_000)

    def _load_state_from_xml(self, xml_state: str) -> None:
        """Parse filesystem state from canonical XML into FSState."""
        self.fs_state.from_xml_string(xml_state)

    def _handle_llm_response(self, response: str) -> bool:
        """Handle LLM response and update filesystem state."""
        print(f"DEBUG: _handle_llm_response called with response starting: {response[:50]}...")
        if response.startswith("ERROR:"):
            print(f"LLM operation failed: {response}")
            return False

        try:
            cleaned = extract_result_from_llm_output(response).strip()
            if "<filesystem" not in cleaned:
                raise ValueError("LLM response did not contain <filesystem>")
            self._load_state_from_xml(cleaned)
            print("DEBUG: Successfully parsed LLM response")
            return True
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return False

    def _format_operation_call(self, operation: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        def format_value(key: Optional[str], value: Any) -> str:
            raw_keys = {"mode", "uid", "gid", "size", "offset", "length", "data_length", "fh", "flags", "datasync"}
            if value is None:
                return "None"
            if isinstance(value, str):
                if key == "data":
                    return value
                if key in raw_keys:
                    return value
                return repr(value)
            return str(value)

        rendered_args = [format_value(None, arg) for arg in args]
        rendered_kwargs = [f"{key}={format_value(key, value)}" for key, value in kwargs.items()]
        params = ", ".join([p for p in (*rendered_args, *rendered_kwargs) if p])
        return f"{operation}({params})"

    def _build_llm_prompt(self, op_tag: str, operation_line: str) -> str:
        current_state = self._get_state_string()
        base_prompt = f"{op_tag}\n{operation_line}\n---\n{current_state}"
        return build_prompt_with_contract(base_prompt)

    def _query_llm_for_operation(
        self,
        operation: str,
        *op_args: Any,
        response_mode: str = "tree",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Query the LLM for a filesystem operation."""
        op_line = self._format_operation_call(operation, op_args, kwargs)
        op_tag = "<R>" if response_mode != "tree" else "<W>"
        prompt = self._build_llm_prompt(op_tag, op_line)

        try:
            response = get_model_response(prompt, temperature=temperature)
            return response.strip()
        except Exception as e:
            return f"ERROR: LLM query failed: {str(e)}"

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
        
        rel_path = path.lstrip('/').rstrip('/')
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

        if path == '/dev':
            return ['.', '..', 'llm']

        try:
            response = self._query_llm_for_operation('readdir', path, fh=fh, response_mode="json")
            cleaned = extract_result_from_llm_output(response)
            entries = json.loads(cleaned)
            if not isinstance(entries, list):
                raise ValueError("readdir response was not a list")
        except Exception as exc:
            print(f"DEBUG: readdir fallback due to {exc}")
            entries = self._fallback_readdir_entries(path)

        normalized = []
        seen = set()
        for entry in entries:
            name = str(entry)
            if name not in seen:
                normalized.append(name)
                seen.add(name)

        if '.' not in seen:
            normalized.insert(0, '.')
            seen.add('.')
        if '..' not in seen:
            normalized.insert(1 if normalized else 0, '..')
            seen.add('..')

        if path == '/' and 'dev' not in seen:
            normalized.append('dev')

        return normalized

    def _fallback_readdir_entries(self, path: str) -> List[str]:
        entries = ['.', '..']
        if path == '/':
            for file_path, entry in self.fs_state.state.items():
                if file_path and '/' not in file_path:
                    entries.append(entry.name)
            if 'dev' not in entries:
                entries.append('dev')
            return entries

        prefix = path.lstrip('/').rstrip('/')
        if prefix:
            prefix += '/'

        for file_path, entry in self.fs_state.state.items():
            if not file_path.startswith(prefix):
                continue
            relative = file_path[len(prefix):]
            if '/' not in relative and relative:
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

        response = self._query_llm_for_operation('read', path, size=size, offset=offset, fh=fh, response_mode="text")
        if response.startswith("ERROR:"):
            raise FuseOSError(EIO)
        cleaned = extract_result_from_llm_output(response)
        data = cleaned.encode('utf-8')
        start = min(offset, len(data))
        end = min(start + size, len(data))
        return data[start:end]

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

        data_str = data.decode('utf-8', errors='replace')
        response = self._query_llm_for_operation(
            'write',
            path,
            data=data_str,
            data_length=len(data),
            offset=offset,
            fh=fh,
        )
        if not self._handle_llm_response(response):
            raise FuseOSError(EIO)
        return len(data)

    def truncate(self, path, length, fh=None):
        """Truncate a file."""
        response = self._query_llm_for_operation('truncate', path, length=length, fh=fh)
        if not self._handle_llm_response(response):
            raise FuseOSError(EIO)
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
        response = self._query_llm_for_operation('readlink', path, response_mode="text")
        if response.startswith("ERROR:"):
            raise FuseOSError(ENOENT)
        return extract_result_from_llm_output(response)

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
