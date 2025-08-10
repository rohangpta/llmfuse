#!/usr/bin/env python3
"""
Reference FUSE filesystem implementation that logs all operations.

This implementation uses a loopback filesystem that delegates all operations
to a real filesystem directory, ensuring perfect semantic correctness while
logging every FUSE operation call for training data generation.
"""

import json
import os
import sys
import threading
from errno import EACCES, ENOENT, ENOTEMPTY, EEXIST, EISDIR, ENOTDIR
from os.path import realpath
from stat import S_IFDIR, S_IFREG, S_IFLNK
from time import time
from typing import Dict, List, Any, Optional

from fuse import FUSE, FuseOSError, Operations, LoggingMixIn

from llmfuse.fs_state import FSState, FileEntry



class LoggingLoopbackFS(LoggingMixIn, Operations):
    """
    A loopback FUSE filesystem that logs all operations for training data generation.
    
    This implementation delegates all filesystem operations to a real directory
    on the host filesystem, ensuring perfect semantic correctness while logging
    every operation for LLM training data.
    """

    def __init__(self, root_dir: str, log_file: str = "fuse_operations.log"):
        self.root = realpath(root_dir)
        self.log_file = log_file
        self.operation_log = []
        self.lock = threading.Lock()
        self.rwlock = threading.Lock()
        
        # Ensure root directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

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

    def __call__(self, op, path, *args):
        """Route all operations through the root directory."""
        return super(LoggingLoopbackFS, self).__call__(op, self.root + path, *args)

    def access(self, path, mode):
        """Check file access permissions."""
        self._log_operation('access', path.replace(self.root, ''), mode=mode)
        if not os.access(path, mode):
            raise FuseOSError(EACCES)

    def chmod(self, path, mode):
        """Change file permissions."""
        self._log_operation('chmod', path.replace(self.root, ''), mode=oct(mode))
        return os.chmod(path, mode)

    def chown(self, path, uid, gid):
        """Change file ownership."""
        self._log_operation('chown', path.replace(self.root, ''), uid=uid, gid=gid)
        return os.chown(path, uid, gid)

    def create(self, path, mode):
        """Create a file."""
        self._log_operation('create', path.replace(self.root, ''), mode=oct(mode))
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)

    def flush(self, path, fh):
        """Flush file data."""
        # Skip logging flush operations as they're not needed for training
        return os.fsync(fh)

    def fsync(self, path, datasync, fh):
        """Synchronize file data."""
        self._log_operation('fsync', path.replace(self.root, ''), datasync=datasync, fh=fh)
        if datasync != 0:
            return os.fdatasync(fh)
        else:
            return os.fsync(fh)

    def getattr(self, path, fh=None):
        """Get file attributes."""
        self._log_operation('getattr', path.replace(self.root, ''), fh=fh)
        st = os.lstat(path)
        return dict((key, getattr(st, key)) for key in (
            'st_atime', 'st_ctime', 'st_gid', 'st_mode', 'st_mtime',
            'st_nlink', 'st_size', 'st_uid'))

    def link(self, target, source):
        """Create a hard link."""
        self._log_operation('link', source.replace(self.root, ''), target=target.replace(self.root, ''))
        return os.link(self.root + source, target)

    def mkdir(self, path, mode):
        """Create a directory."""
        self._log_operation('mkdir', path.replace(self.root, ''), mode=oct(mode))
        return os.mkdir(path, mode)

    def mknod(self, path, mode, dev):
        """Create a special file."""
        self._log_operation('mknod', path.replace(self.root, ''), mode=oct(mode), dev=dev)
        return os.mknod(path, mode, dev)

    def open(self, path, flags):
        """Open a file."""
        self._log_operation('open', path.replace(self.root, ''), flags=flags)
        return os.open(path, flags)

    def read(self, path, size, offset, fh):
        """Read from a file."""
        self._log_operation('read', path.replace(self.root, ''), size=size, offset=offset, fh=fh)
        with self.rwlock:
            os.lseek(fh, offset, 0)
            return os.read(fh, size)

    def readdir(self, path, fh):
        """Read directory contents."""
        self._log_operation('readdir', path.replace(self.root, ''), fh=fh)
        return ['.', '..'] + os.listdir(path)

    def readlink(self, path):
        """Read the target of a symbolic link."""
        self._log_operation('readlink', path.replace(self.root, ''))
        return os.readlink(path)

    def release(self, path, fh):
        """Release a file handle."""
        self._log_operation('release', path.replace(self.root, ''), fh=fh)
        return os.close(fh)

    def rename(self, old, new):
        """Rename a file or directory."""
        self._log_operation('rename', old.replace(self.root, ''), new_path=new.replace(self.root, ''))
        return os.rename(old, self.root + new)

    def rmdir(self, path):
        """Remove a directory."""
        self._log_operation('rmdir', path.replace(self.root, ''))
        return os.rmdir(path)

    def statfs(self, path):
        """Get filesystem statistics."""
        self._log_operation('statfs', path.replace(self.root, ''))
        stv = os.statvfs(path)
        return dict((key, getattr(stv, key)) for key in (
            'f_bavail', 'f_bfree', 'f_blocks', 'f_bsize', 'f_favail',
            'f_ffree', 'f_files', 'f_flag', 'f_frsize', 'f_namemax'))

    def symlink(self, target, source):
        """Create a symbolic link."""
        self._log_operation('symlink', source.replace(self.root, ''), target=target)
        return os.symlink(source, target)

    def truncate(self, path, length, fh=None):
        """Truncate a file."""
        self._log_operation('truncate', path.replace(self.root, ''), length=length, fh=fh)
        with open(path, 'r+') as f:
            f.truncate(length)

    def unlink(self, path):
        """Remove a file."""
        self._log_operation('unlink', path.replace(self.root, ''))
        return os.unlink(path)

    def utimens(self, path, times=None):
        """Set file timestamps."""
        self._log_operation('utimens', path.replace(self.root, ''), times=times)
        return os.utime(path, times)

    def write(self, path, data, offset, fh):
        """Write to a file."""
        self._log_operation('write', path.replace(self.root, ''), data_length=len(data), offset=offset, fh=fh)
        with self.rwlock:
            os.lseek(fh, offset, 0)
            return os.write(fh, data)

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the current operation log."""
        with self.lock:
            return self.operation_log.copy()

    def clear_operation_log(self) -> None:
        """Clear the operation log."""
        with self.lock:
            self.operation_log.clear()

    def _get_current_state(self) -> str:
        """Get current filesystem state as a tree string."""
        # Create FSState from the actual filesystem
        fs_state = FSState(self.root)
        fs_state.sync_from_fs()
        return fs_state.to_tree_string()

def main(mountpoint: str, root_dir: str, log_file: str = "fuse_operations.log"):
    """Run the reference loopback FUSE filesystem."""
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    fs = LoggingLoopbackFS(root_dir, log_file)
    FUSE(fs, mountpoint, nothreads=True, foreground=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: %s <mountpoint> <root_directory> [log_file]' % sys.argv[0])
        print('  mountpoint: Directory where the FUSE filesystem will be mounted')
        print('  root_directory: Real directory to use as the filesystem backend')
        print('  log_file: Optional log file path (default: fuse_operations.log)')
        sys.exit(1)
    
    mountpoint = sys.argv[1]
    root_dir = sys.argv[2]
    log_file = sys.argv[3] if len(sys.argv) > 3 else "fuse_operations.log"
    
    main(mountpoint, root_dir, log_file) 