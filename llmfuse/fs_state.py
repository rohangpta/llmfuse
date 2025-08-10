"""
Filesystem state representation and management.

This module provides classes for representing and manipulating filesystem state,
including conversion to tree-like string representations.
"""


# Standard library imports
import os
import stat
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

# Third-party imports
import pwd
import grp

# Local imports
from .utils import (
    TREE_PADDING, TREE_CONNECTOR_LAST, TREE_CONNECTOR_MIDDLE, 
    TREE_VERTICAL_LINE, TREE_INDENT, BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB
)

@dataclass
class FileEntry:
    """
    Represents a single filesystem entry (file or directory).
    
    Attributes:
        name: The basename of the file/directory
        is_dir: True if this is a directory, False for files
        mode: Unix file permissions (e.g., 0o755)
        owner: Username of the file owner
        group: Group name of the file group
        mtime: Last modification time as Unix timestamp
        size: File size in bytes (0 for directories)
    """
    name: str
    is_dir: bool
    mode: int
    owner: str
    group: str
    mtime: float
    size: int = 0

class FSState:
    """
    Represents the complete state of a filesystem tree.
    
    This class can sync from an actual filesystem directory and provide
    various representations of the filesystem state, including tree-like
    string output suitable for LLM training.
    """
    
    def __init__(self, root_path: Optional[str] = None) -> None:
        """
        Initialize filesystem state.
        
        Args:
            root_path: Path to the root directory to represent. If None,
                      creates an empty state that can be populated manually.
        """
        self.root_path = root_path
        self.state: Dict[str, FileEntry] = {}

    def _get_file_info(self, path: str) -> Optional[FileEntry]:
        """
        Get FileEntry for a given path relative to the root.
        
        Args:
            path: Relative path from root (empty string for root itself)
            
        Returns:
            FileEntry object if successful, None if file cannot be accessed
        """
        full_path = os.path.join(self.root_path, path)
        try:
            st = os.stat(full_path)
            owner = pwd.getpwuid(st.st_uid).pw_name
            group = grp.getgrgid(st.st_gid).gr_name
            return FileEntry(
                name=os.path.basename(path) if path else "/",
                is_dir=stat.S_ISDIR(st.st_mode),
                mode=stat.S_IMODE(st.st_mode),
                owner=owner,
                group=group,
                size=st.st_size,
                mtime=st.st_mtime,
            )
        except (FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not stat file {full_path}: {e}")
            return None

    def sync_from_fs(self) -> None:
        """
        Sync the internal state from the actual filesystem.
        
        Walks the filesystem tree starting from root_path and populates
        the internal state dictionary with FileEntry objects.
        """
        if not self.root_path or not os.path.exists(self.root_path):
            self.state = {}
            return

        new_state: Dict[str, FileEntry] = {}
        for root, dirs, files in os.walk(self.root_path):
            all_entries = dirs + files
            for name in all_entries:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, self.root_path)
                
                # For the root directory itself
                if rel_path == '.':
                    rel_path = ''

                entry = self._get_file_info(rel_path)
                if entry:
                    new_state[rel_path] = entry

        # Add the root directory itself
        root_entry = self._get_file_info('')
        if root_entry:
            new_state[''] = root_entry

        self.state = new_state

    def get_state(self, path: str) -> Optional[FileEntry]:
        """
        Get the internal state for a given path.
        
        Args:
            path: Relative path to look up
            
        Returns:
            FileEntry if path exists, None otherwise
        """
        return self.state.get(path)

    def remove_state(self, path: str) -> None:
        """
        Remove an entry from the internal state.
        
        Args:
            path: Relative path to remove
        """
        if path in self.state:
            del self.state[path]

    def update_state(self, path: str, entry: FileEntry) -> None:
        """
        Update the internal state for a given path.
        
        Args:
            path: Relative path to update
            entry: New FileEntry to store at this path
        """
        self.state[path] = entry

    def to_tree_string(self, include_contents: bool = True, max_file_size: int = 500) -> str:
        """
        Return a string representation of the filesystem state as a tree.
        
        Generates a tree-like representation similar to the Unix `tree` command,
        with proper Unicode box-drawing characters and aligned columns for
        file metadata. Optionally includes file contents.
        
        Args:
            include_contents: If True, include file contents for small files
            max_file_size: Maximum file size (in chars) to include contents for
        
        Returns:
            Multi-line string representation of the filesystem tree
        """
        if not self.state:
            return ""

        # Build a tree structure from the flat path dictionary
        tree: Dict[str, Dict] = {}
        for path, entry in sorted(self.state.items()):
            if path == '': 
                continue
            parts = path.split(os.sep)
            node = tree
            for part in parts:
                node = node.setdefault(part, {})
        
        # Recursively build the output string
        s = io.StringIO()
        
        # Handle the root entry separately
        root_entry = self.state.get('')
        if root_entry:
            time_str = datetime.fromtimestamp(root_entry.mtime).strftime('%b %d %H:%M')
            s.write(f"{'/':<{TREE_PADDING // 2}} dir  {root_entry.mode:o} {root_entry.owner}:{root_entry.group} {time_str}\n")

        def build_lines_recursive_with_path(subtree: Dict[str, Dict], parent_path: str = "", prefix: str = "") -> None:
            """
            Recursively build tree lines with proper path tracking.
            
            Args:
                subtree: Dictionary representing the tree structure
                parent_path: Current parent path for building full paths
                prefix: Current line prefix for tree formatting
            """
            children = sorted(subtree.keys())
            for i, child_name in enumerate(children):
                is_last = (i == len(children) - 1)
                current_path = os.path.join(parent_path, child_name)
                entry = self.state[current_path]

                connector = TREE_CONNECTOR_LAST if is_last else TREE_CONNECTOR_MIDDLE
                
                time_str = datetime.fromtimestamp(entry.mtime).strftime('%b %d %H:%M')
                type_str = "dir" if entry.is_dir else "file"
                size_str = self._format_size(entry.size) if not entry.is_dir else ""
                
                name_part = f"{prefix}{connector}{entry.name}"
                
                line = f"{name_part:<{TREE_PADDING}} {type_str:<4} {entry.mode:o} {entry.owner}:{entry.group} {time_str} {size_str}".rstrip()
                s.write(line + "\n")

                # Include file contents if requested and file is small enough
                if include_contents and not entry.is_dir and entry.size <= max_file_size and self.root_path:
                    try:
                        full_path = os.path.join(self.root_path, current_path) if current_path else self.root_path
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Add file contents with proper indentation
                        content_prefix = prefix + (TREE_INDENT if is_last else TREE_VERTICAL_LINE) + "    "
                        content_lines = content.split('\n')
                        
                        # Limit content lines to prevent excessive output
                        max_content_lines = 20
                        if len(content_lines) > max_content_lines:
                            content_lines = content_lines[:max_content_lines] + [f"... ({len(content_lines) - max_content_lines} more lines)"]
                        
                        for content_line in content_lines:
                            if content_line or len(content_lines) == 1:  # Show empty lines unless it's the only line
                                s.write(f"{content_prefix}│ {content_line}\n")
                    
                    except (UnicodeDecodeError, IOError):
                        # Skip binary files or files that can't be read
                        content_prefix = prefix + (TREE_INDENT if is_last else TREE_VERTICAL_LINE) + "    "
                        s.write(f"{content_prefix}│ <binary or unreadable file>\n")

                if entry.is_dir and subtree[child_name]:
                    new_prefix = prefix + (TREE_INDENT if is_last else TREE_VERTICAL_LINE)
                    build_lines_recursive_with_path(subtree[child_name], current_path, new_prefix)

        build_lines_recursive_with_path(tree)
        return s.getvalue().strip()

    def estimate_token_count(self, include_contents: bool = True, max_file_size: int = 500) -> int:
        """
        Estimate the token count of the tree string representation.
        
        Args:
            include_contents: If True, include file contents in estimation
            max_file_size: Maximum file size to include contents for
            
        Returns:
            Estimated number of tokens (rough approximation: 1 token ≈ 4 chars)
        """
        tree_str = self.to_tree_string(include_contents=include_contents, max_file_size=max_file_size)
        # Rough approximation: 1 token ≈ 4 characters
        return len(tree_str) // 4
    
    def get_total_content_size(self) -> int:
        """
        Get the total size of all file contents in the filesystem.
        
        Returns:
            Total content size in characters
        """
        total_size = 0
        if not self.root_path:
            return 0
            
        for path, entry in self.state.items():
            if not entry.is_dir and path:  # Skip root directory
                try:
                    full_path = os.path.join(self.root_path, path)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    total_size += len(content)
                except (UnicodeDecodeError, IOError):
                    # Skip binary or unreadable files
                    pass
        return total_size

    def _format_size(self, size: Optional[int]) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size: Size in bytes, or None
            
        Returns:
            Formatted size string (e.g., "1.50 KB", "2.34 MB")
        """
        if size is None:
            return ""
        if size == 0:
            return "0 B"
        if size < BYTES_PER_KB:
            return f"{size} B"
        if size < BYTES_PER_MB:
            return f"{size / BYTES_PER_KB:.2f} KB"
        if size < BYTES_PER_GB:
            return f"{size / BYTES_PER_MB:.2f} MB"
        return f"{size / BYTES_PER_GB:.2f} GB"