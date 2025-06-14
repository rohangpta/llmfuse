"""
Unit tests for the filesystem state representation module.

This module tests the FSState class and FileEntry dataclass to ensure
proper filesystem state management and tree representation.
"""
# Standard library imports
import os
import stat
import tempfile
from datetime import datetime
from typing import Generator

# Third-party imports
import pytest

# Local imports
from .fs_state import FSState, FileEntry
from .utils import DEFAULT_FILE_MODE, DEFAULT_DIR_MODE, TREE_PADDING

@pytest.fixture
def temp_fs(tmp_path) -> str:
    """
    Create a temporary filesystem for testing.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        String path to the temporary filesystem root
    """
    (tmp_path / "home" / "user").mkdir(parents=True)
    (tmp_path / "home" / "user" / "doc.txt").touch()
    (tmp_path / "tmp").mkdir()
    return str(tmp_path)

def test_fs_state_initialization(temp_fs: str) -> None:
    """
    Test FSState initialization and basic state loading.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Check if root directory exists in state
    assert "" in fs.state
    root_entry = fs.state[""]
    assert root_entry.is_dir
    assert root_entry.name == "/"

def test_file_info_retrieval(temp_fs: str) -> None:
    """
    Test retrieval of file information.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Test directory info
    home_dir = fs.get_state("home")
    assert home_dir is not None
    assert home_dir.is_dir
    assert home_dir.name == "home"
    
    # Test file info
    doc_file = fs.get_state("home/user/doc.txt")
    assert doc_file is not None
    assert not doc_file.is_dir
    assert doc_file.name == "doc.txt"
    assert doc_file.size == 0

def test_tree_string_format(temp_fs: str) -> None:
    """
    Test the tree string representation format.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    tree_str = fs.to_tree_string()
    
    # Basic format checks
    assert "dir" in tree_str
    assert "file" in tree_str
    assert "home" in tree_str
    assert "doc.txt" in tree_str
    
    # Check for proper indentation
    lines = tree_str.split("\n")
    assert any("  " in line for line in lines)  # Should have indentation
    
    # Check for proper mode representation
    assert "755" in tree_str or "750" in tree_str or "777" in tree_str

def test_state_updates(temp_fs: str) -> None:
    """
    Test updating the internal state.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Create a new file entry
    new_file = FileEntry(
        name="new_file.txt",
        is_dir=False,
        mode=DEFAULT_FILE_MODE,
        owner="user",
        group="user",
        size=1024,
        mtime=datetime.now().timestamp()
    )
    
    # Update state
    fs.update_state("home/user/new_file.txt", new_file)
    
    # Verify update
    updated_entry = fs.get_state("home/user/new_file.txt")
    assert updated_entry is not None
    assert updated_entry.name == "new_file.txt"
    assert updated_entry.size == 1024

def test_state_removal(temp_fs: str) -> None:
    """
    Test removing entries from state.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Remove an existing entry
    fs.remove_state("home/user/doc.txt")
    
    # Verify removal
    assert fs.get_state("home/user/doc.txt") is None
    
    # Verify other entries still exist
    assert fs.get_state("home/user") is not None

def test_mode_formatting(temp_fs: str) -> None:
    """
    Test mode string formatting.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Check if mode is properly formatted in tree string
    tree_str = fs.to_tree_string()
    assert (f"{DEFAULT_DIR_MODE:o}" in tree_str or
            f"{0o750:o}" in tree_str or
            f"{0o777:o}" in tree_str)

def test_size_formatting(temp_fs: str) -> None:
    """
    Test file size formatting.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Create a file with known size
    test_file = FileEntry(
        name="size_test.txt",
        is_dir=False,
        mode=DEFAULT_FILE_MODE,
        owner="user",
        group="user",
        size=1024,
        mtime=datetime.now().timestamp()
    )
    
    fs.update_state("home/user/size_test.txt", test_file)
    tree_str = fs.to_tree_string()
    
    # Check if size is properly formatted
    assert "1.00 KB" in tree_str

def test_invalid_path_handling(temp_fs: str) -> None:
    """
    Test handling of invalid paths.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Test getting state for non-existent path
    assert fs.get_state("nonexistent/path") is None
    
    # Test removing non-existent path
    fs.remove_state("nonexistent/path")  # Should not raise an error

def test_directory_traversal(temp_fs: str) -> None:
    """
    Test proper directory traversal and state building.
    
    Args:
        temp_fs: Temporary filesystem path from fixture
    """
    fs = FSState(temp_fs)
    fs.sync_from_fs()
    
    # Verify all expected paths are in state
    expected_paths = {
        "",
        "home",
        "home/user",
        "home/user/doc.txt",
        "tmp"
    }
    
    assert set(fs.state.keys()) == expected_paths

def test_to_tree_string_exact_format() -> None:
    """
    Test the exact output format of to_tree_string.
    
    Creates a controlled filesystem state to verify exact tree formatting
    with proper connectors, spacing, and metadata display.
    """
    # Create an FSState instance without relying on a real filesystem.
    fs = FSState()
    
    # Use a fixed timestamp for deterministic tests
    fixed_time = datetime(2023, 10, 27, 10, 0, 0).timestamp()
    time_str = datetime.fromtimestamp(fixed_time).strftime('%b %d %H:%M')

    fs.state = {
        '': FileEntry(name='/', is_dir=True, mode=DEFAULT_DIR_MODE, owner='root', group='root', mtime=fixed_time),
        'home': FileEntry(name='home', is_dir=True, mode=DEFAULT_DIR_MODE, owner='root', group='root', mtime=fixed_time),
        'home/user': FileEntry(name='user', is_dir=True, mode=0o750, owner='user', group='user', mtime=fixed_time),
        'home/user/doc.txt': FileEntry(name='doc.txt', is_dir=False, mode=DEFAULT_FILE_MODE, owner='user', group='user', size=1024, mtime=fixed_time),
        'tmp': FileEntry(name='tmp', is_dir=True, mode=0o777, owner='root', group='root', mtime=fixed_time),
    }

    expected_output = f"""
/                dir  {DEFAULT_DIR_MODE:o} root:root {time_str}
├── home                         dir  {DEFAULT_DIR_MODE:o} root:root {time_str}
│   └── user                     dir  750 user:user {time_str}
│       └── doc.txt              file {DEFAULT_FILE_MODE:o} user:user {time_str} 1.00 KB
└── tmp                          dir  777 root:root {time_str}
""".strip()

    actual_output = fs.to_tree_string().strip()
    assert actual_output == expected_output

def test_empty_directory_tree_string() -> None:
    """
    Test that an empty filesystem state produces the correct string.
    
    Verifies that a directory with only the root entry produces
    a minimal but correct tree representation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        fs = FSState(temp_dir)
        fs.sync_from_fs()
        # The only entry should be the root.
        actual_output = fs.to_tree_string().strip()
        
        entry = fs.state['']
        time_str = datetime.fromtimestamp(entry.mtime).strftime('%b %d %H:%M')
        expected_output = f"/                dir  {entry.mode:o} {entry.owner}:{entry.group} {time_str}"
        
        assert actual_output == expected_output

def test_deeply_nested_directory() -> None:
    """
    Test tree representation with deeply nested directory structures.
    
    Verifies that the tree formatting correctly handles multiple levels
    of nesting with proper indentation and connectors.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a deeply nested structure
        nested_path = os.path.join(temp_dir, "a", "b", "c", "d", "e")
        os.makedirs(nested_path)
        
        # Create a file at the deepest level
        file_path = os.path.join(nested_path, "deep_file.txt")
        with open(file_path, 'w') as f:
            f.write("deep content")
        
        fs = FSState(temp_dir)
        fs.sync_from_fs()
        tree_str = fs.to_tree_string()
        
        # Verify the structure is represented correctly
        assert "deep_file.txt" in tree_str
        assert "└──" in tree_str  # Should have proper connectors
        
        # Count the levels of indentation
        lines = tree_str.split('\n')
        max_indent = 0
        for line in lines:
            if "deep_file.txt" in line:
                # Count leading spaces/connectors to verify depth
                stripped = line.lstrip()
                indent_chars = len(line) - len(stripped)
                max_indent = max(max_indent, indent_chars)
        
        # Should have significant indentation for deep nesting
        assert max_indent >= 20

def test_large_number_of_files() -> None:
    """
    Test tree representation with a large number of files.
    
    Verifies that the system can handle directories with many files
    and produces properly formatted output.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create many files in a single directory
        num_files = 50
        for i in range(num_files):
            file_path = os.path.join(temp_dir, f"file_{i:03d}.txt")
            with open(file_path, 'w') as f:
                f.write(f"content {i}")
        
        fs = FSState(temp_dir)
        fs.sync_from_fs()
        tree_str = fs.to_tree_string()
        
        # Verify all files are represented
        for i in range(num_files):
            assert f"file_{i:03d}.txt" in tree_str
        
        # Verify proper formatting is maintained
        lines = tree_str.split('\n')
        assert len(lines) >= num_files  # At least one line per file
        
        # Check that files are properly sorted
        file_lines = [line for line in lines if "file_" in line]
        file_names = []
        for line in file_lines:
            # Extract filename from the line
            parts = line.split()
            for part in parts:
                if part.startswith("file_"):
                    file_names.append(part)
                    break
        
        # Verify alphabetical sorting
        assert file_names == sorted(file_names) 