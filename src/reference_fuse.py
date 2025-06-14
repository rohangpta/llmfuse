#!/usr/bin/env python3
"""
Reference FUSE filesystem implementation that logs all operations.

This serves as the ground truth for training data generation by logging
every FUSE operation call with its parameters and results.
"""

import json
import os
import sys
import threading
from errno import ENOENT, ENOTEMPTY, EEXIST, EISDIR, ENOTDIR
from stat import S_IFDIR, S_IFREG, S_IFLNK
from time import time
from typing import Dict, List, Any, Optional

from fuse import FUSE, FuseOSError, Operations, LoggingMixIn

from .fs_state import FSState, FileEntry

class LoggingFS(LoggingMixIn, Operations):
    """
    A FUSE filesystem that logs all operations for training data generation.
    
    This implementation maintains both an in-memory filesystem state and
    logs every operation call for use in LLM training.
    """

    def __init__(self, log_file: str = "fuse_operations.log"):
        self.files = {}
        self.data = {}
        self.symlinks = {}
        self.fd = 0
        self.log_file = log_file
        self.operation_log = []
        self.lock = threading.Lock()
        
        # Initialize root directory
        now = time()
        self.files['/'] = dict(
            st_mode=(S_IFDIR | 0o755),
            st_ctime=now,
            st_mtime=now,
            st_atime=now,
            st_nlink=2,
            st_uid=os.getuid(),
            st_gid=os.getgid(),
            st_size=0
        )

    def _log_operation(self, operation: str, path: str, **kwargs) -> None:
        """Log a FUSE operation with its parameters."""
        with self.lock:
            log_entry = {
                'operation': operation,
                'path': path,
                'timestamp': time(),
                'parameters': kwargs
            }
            self.operation_log.append(log_entry)
            
            # Write to log file immediately for real-time access
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    def _get_current_state(self) -> str:
        """Get current filesystem state as a tree string."""
        # Create a temporary FSState from our current files
        temp_state = FSState(".")
        temp_state.state = {}  # Fixed: use 'state' not 'files'
        
        for path, attrs in self.files.items():
            if path == '/':
                # Handle root directory specially
                temp_state.state[''] = FileEntry(
                    name='/',
                    is_dir=True,
                    mode=attrs['st_mode'] & 0o777,
                    owner='user',
                    group='user',
                    mtime=attrs.get('st_mtime', 0),
                    size=attrs.get('st_size', 0)
                )
                continue
                
            # Convert absolute path to relative path for FSState
            rel_path = path.lstrip('/')
            if not rel_path:
                continue
                
            temp_state.state[rel_path] = FileEntry(
                name=os.path.basename(rel_path) if rel_path else '',
                is_dir=(attrs['st_mode'] & S_IFDIR) != 0,
                mode=attrs['st_mode'] & 0o777,
                owner='user',
                group='user',
                mtime=attrs.get('st_mtime', 0),
                size=attrs.get('st_size', 0)
            )
        
        return temp_state.to_tree_string()

    def getattr(self, path, fh=None):
        """Get file attributes."""
        self._log_operation('getattr', path, fh=fh)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        return self.files[path]

    def readdir(self, path, fh):
        """Read directory contents."""
        self._log_operation('readdir', path, fh=fh)
        
        if path == '/':
            # Return all top-level entries
            entries = ['.', '..']
            for file_path in self.files:
                if file_path != '/' and '/' not in file_path.strip('/'):
                    entries.append(file_path.strip('/'))
            return entries
        else:
            # Return entries in subdirectory
            entries = ['.', '..']
            prefix = path.rstrip('/') + '/'
            for file_path in self.files:
                if file_path.startswith(prefix):
                    relative = file_path[len(prefix):]
                    if '/' not in relative:  # Direct child only
                        entries.append(relative)
            return entries

    def mkdir(self, path, mode):
        """Create a directory."""
        self._log_operation('mkdir', path, mode=oct(mode))
        
        # Check if path already exists
        if path in self.files:
            raise FuseOSError(EEXIST)
        
        now = time()
        self.files[path] = dict(
            st_mode=(S_IFDIR | mode),
            st_nlink=2,
            st_size=0,
            st_ctime=now,
            st_mtime=now,
            st_atime=now,
            st_uid=os.getuid(),
            st_gid=os.getgid()
        )

    def create(self, path, mode):
        """Create a file."""
        self._log_operation('create', path, mode=oct(mode))
        
        # Check if path already exists
        if path in self.files:
            raise FuseOSError(EEXIST)
        
        now = time()
        self.files[path] = dict(
            st_mode=(S_IFREG | mode),
            st_nlink=1,
            st_size=0,
            st_ctime=now,
            st_mtime=now,
            st_atime=now,
            st_uid=os.getuid(),
            st_gid=os.getgid()
        )
        self.data[path] = b''
        self.fd += 1
        return self.fd

    def unlink(self, path):
        """Remove a file."""
        self._log_operation('unlink', path)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        if self.files[path]['st_mode'] & S_IFDIR:
            raise FuseOSError(EISDIR)
        
        if path in self.data:
            self.data.pop(path)
        self.files.pop(path)

    def rmdir(self, path):
        """Remove a directory."""
        self._log_operation('rmdir', path)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        if not (self.files[path]['st_mode'] & S_IFDIR):
            raise FuseOSError(ENOTDIR)
        
        # Check if directory is empty
        prefix = path.rstrip('/') + '/'
        for file_path in self.files:
            if file_path.startswith(prefix):
                raise FuseOSError(ENOTEMPTY)  # Fixed: use ENOTEMPTY not ENOENT
        
        self.files.pop(path)

    def chmod(self, path, mode):
        """Change file permissions."""
        self._log_operation('chmod', path, mode=oct(mode))
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        # Preserve file type bits, update permission bits
        file_type = self.files[path]['st_mode'] & 0o170000
        self.files[path]['st_mode'] = file_type | mode
        return 0

    def chown(self, path, uid, gid):
        """Change file ownership."""
        self._log_operation('chown', path, uid=uid, gid=gid)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        self.files[path]['st_uid'] = uid
        self.files[path]['st_gid'] = gid

    def open(self, path, flags):
        """Open a file."""
        self._log_operation('open', path, flags=flags)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        self.fd += 1
        return self.fd

    def read(self, path, size, offset, fh):
        """Read from a file."""
        self._log_operation('read', path, size=size, offset=offset, fh=fh)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        return self.data.get(path, b'')[offset:offset + size]

    def write(self, path, data, offset, fh):
        """Write to a file."""
        self._log_operation('write', path, data_length=len(data), offset=offset, fh=fh)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        if path not in self.data:
            self.data[path] = b''
        
        # Extend file if necessary
        if offset > len(self.data[path]):
            self.data[path] += b'\x00' * (offset - len(self.data[path]))
        
        # Write data
        self.data[path] = (
            self.data[path][:offset] + 
            data + 
            self.data[path][offset + len(data):]
        )
        
        # Update file size and mtime
        if path in self.files:
            self.files[path]['st_size'] = len(self.data[path])
            self.files[path]['st_mtime'] = time()
        
        return len(data)

    def truncate(self, path, length, fh=None):
        """Truncate a file."""
        self._log_operation('truncate', path, length=length, fh=fh)
        
        if path not in self.files:
            raise FuseOSError(ENOENT)
        
        if path in self.data:
            self.data[path] = self.data[path][:length]
        if path in self.files:
            self.files[path]['st_size'] = length
            self.files[path]['st_mtime'] = time()

    def rename(self, old, new):
        """Rename a file or directory."""
        self._log_operation('rename', old, new_path=new)
        
        if old not in self.files:
            raise FuseOSError(ENOENT)
        
        if new in self.files:
            raise FuseOSError(EEXIST)
        
        self.files[new] = self.files.pop(old)
        if old in self.data:
            self.data[new] = self.data.pop(old)
        if old in self.symlinks:
            self.symlinks[new] = self.symlinks.pop(old)

    def symlink(self, target, source):
        """Create a symbolic link."""
        self._log_operation('symlink', source, target=target)
        
        if source in self.files:
            raise FuseOSError(EEXIST)
        
        now = time()
        self.symlinks[source] = target
        self.files[source] = dict(
            st_mode=(S_IFLNK | 0o777),
            st_nlink=1,
            st_size=len(target),
            st_ctime=now,
            st_mtime=now,
            st_atime=now,
            st_uid=os.getuid(),
            st_gid=os.getgid()
        )
        return 0

    def readlink(self, path):
        """Read the target of a symbolic link."""
        self._log_operation('readlink', path)
        
        if path not in self.symlinks:
            raise FuseOSError(ENOENT)
        
        return self.symlinks[path]

    def statfs(self, path):
        """Get filesystem statistics."""
        self._log_operation('statfs', path)
        return dict(f_bsize=512, f_blocks=4096, f_bavail=2048)

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the current operation log."""
        with self.lock:
            return self.operation_log.copy()

    def clear_operation_log(self) -> None:
        """Clear the operation log."""
        with self.lock:
            self.operation_log.clear()

def main(mountpoint: str, log_file: str = "fuse_operations.log"):
    """Run the reference FUSE filesystem."""
    fs = LoggingFS(log_file)
    FUSE(fs, mountpoint, nothreads=True, foreground=True)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s <mountpoint> [log_file]' % sys.argv[0])
        sys.exit(1)
    
    mountpoint = sys.argv[1]
    log_file = sys.argv[2] if len(sys.argv) > 2 else "fuse_operations.log"
    
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    main(mountpoint, log_file) 