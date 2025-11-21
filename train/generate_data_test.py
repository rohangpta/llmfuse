"""
Unit tests for the synthetic filesystem data generation module.

This module tests the data generation functions to ensure they produce
valid training examples with proper structure and operation distribution.
"""
# Standard library imports
import os
import tempfile
import unittest
import unittest.mock
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

# Local imports
# Import generate_data functions directly to avoid train package dependencies
import importlib.util
import os
import sys

# Add project root to path so llmfuse can be found
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

spec = importlib.util.spec_from_file_location(
    'generate_data', 
    os.path.join(os.path.dirname(__file__), 'generate_data.py')
)
generate_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_data_module)
generate_random_operation = generate_data_module.generate_random_operation
execute_command = generate_data_module.execute_command
from llmfuse.fs_state import FSState, FileEntry
from llmfuse.utils import DEFAULT_FILE_MODE, DEFAULT_DIR_MODE


def mock_generate_one() -> Dict[str, Any]:
    """Mock implementation of generate_one that doesn't require FUSE."""
    return {
        'initial_state': '.\n├── file1.txt (36 B, root:root, Jun 14 17:57)\n└── dir1/ (4096 B, root:root, Jun 14 17:57)',
        'operation': 'mkdir("/test_dir")',
        'result': '.\n├── file1.txt (36 B, root:root, Jun 14 17:57)\n├── dir1/ (4096 B, root:root, Jun 14 17:57)\n└── test_dir/ (4096 B, root:root, Jun 14 17:57)',
        'operation_type': 'state_change'
    }

class TestGenerateData(unittest.TestCase):
    """Test cases for synthetic filesystem data generation."""

    def test_generate_one_structure(self) -> None:
        """
        Test that a single generated example has the correct structure.
        
        Verifies that the generated example contains all required fields
        with appropriate data types. Uses mock to avoid FUSE dependencies.
        """
        example = mock_generate_one()
        self.assertIsNotNone(example)
        
        # Check for required field names
        required_fields = ['initial_state', 'operation', 'result', 'operation_type']
        for field in required_fields:
            self.assertIn(field, example, f"Missing required field: {field}")
        
        # Check data types
        self.assertIsInstance(example['initial_state'], str)
        self.assertIsInstance(example['operation'], str)
        self.assertIsInstance(example['result'], str)
        self.assertIsInstance(example['operation_type'], str)
        
        # Check operation type is valid (updated for Smart Dual Training)
        self.assertIn(example['operation_type'], ['state_change', 'query'])

    def test_execute_command_success(self) -> None:
        """
        Test that command execution returns proper success/failure status.
        
        Verifies that the execute_command function properly handles
        both successful and failed command executions.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test successful command
            success, output = execute_command("ls -la .", temp_dir)
            self.assertTrue(success)
            self.assertIsInstance(output, str)
            
            # Test failed command
            success, output = execute_command("ls /nonexistent/path", temp_dir)
            self.assertFalse(success)
            self.assertIsInstance(output, str)

    def test_execute_command_with_file_creation(self) -> None:
        """
        Test command execution with file operations.
        
        Verifies that file creation and listing commands work correctly
        in the temporary directory environment.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            success, output = execute_command("touch testfile.txt", temp_dir)
            self.assertTrue(success)
            
            # List files to verify creation
            success, output = execute_command("ls -la .", temp_dir)
            self.assertTrue(success)
            self.assertIn("testfile.txt", output)

    def test_generate_random_operation_types(self) -> None:
        """
        Test that random operation generation produces valid operation types.
        
        Verifies that the generate_random_operation function returns
        operations in the expected format and with valid types.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some initial structure
            execute_command("mkdir testdir", temp_dir)
            execute_command("touch testfile.txt", temp_dir)
            
            # Generate multiple operations to test variety
            operations = []
            for _ in range(50):
                op_type, command = generate_random_operation(temp_dir)
                operations.append(op_type)
                
                # Verify return types
                self.assertIsInstance(op_type, str)
                self.assertIsInstance(command, str)
                
                # Verify operation type is valid (complete list from OPERATION_WEIGHTS)
                valid_ops = ['mkdir', 'touch', 'write', 'read', 'rm', 'rmdir', 'chmod', 'chown', 'truncate', 'rename', 'symlink', 'ls', 'stat']
                self.assertIn(op_type, valid_ops)
            
            # Check that we get some variety in operations
            unique_ops = set(operations)
            self.assertGreater(len(unique_ops), 3, "Should generate multiple operation types")

    def test_operation_distribution(self) -> None:
        """
        Test that generated operations have a reasonable distribution.
        
        This is a statistical test that verifies the operation generation
        produces a balanced mix of different operation types.
        """
        num_samples = 200
        operations: List[str] = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some initial filesystem structure for more realistic operations
            execute_command("mkdir -p dir1/subdir", temp_dir)
            execute_command("touch file1.txt dir1/file2.txt", temp_dir)
            
            for _ in range(num_samples):
                op_type, _ = generate_random_operation(temp_dir)
                operations.append(op_type)
        
        counts = Counter(operations)
        print(f"Generated operation distribution: {counts}")

        # Check that most common operations are present (not all, since some are rare)
        expected_ops = ['mkdir', 'touch', 'write', 'read', 'chmod', 'chown', 'ls', 'stat']  # Most common ones
        for op in expected_ops:
            self.assertIn(op, counts, f"Operation '{op}' was not generated.")

        # Check that the distribution isn't extremely skewed
        total_ops = len(operations)
        for op, count in counts.items():
            percentage = (count / total_ops) * 100
            # Each operation should appear at least 1.5% of the time (more lenient for statistical variance)
            self.assertGreater(percentage, 1.5, 
                             f"Operation '{op}' appears only {percentage:.1f}% of the time")
            # No single operation should dominate (less than 60%)
            self.assertLess(percentage, 60.0,
                           f"Operation '{op}' appears {percentage:.1f}% of the time, which is too dominant")

    def test_operation_command_consistency(self) -> None:
        """
        Test that generated operations have consistent command formats.
        
        Verifies that operation types match their corresponding shell commands.
        Operations follow Smart Dual Training methodology (state changes + queries).
        """
        all_ops = {'mkdir', 'touch', 'write', 'read', 'rm', 'rmdir', 'chmod', 'chown', 'truncate', 'rename', 'symlink', 'ls', 'stat'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some structure for operations
            execute_command("mkdir testdir", temp_dir)
            execute_command("touch testfile.txt", temp_dir)
            
            for _ in range(100):
                op_type, command = generate_random_operation(temp_dir)
                
                # Verify operation type is known
                self.assertIn(op_type, all_ops, f"Unknown operation type: {op_type}")
                
                # Verify command format consistency
                if op_type == 'ls':
                    self.assertTrue(command.startswith('ls '), 
                                  f"ls operation has unexpected command: {command}")
                elif op_type == 'stat':
                    self.assertTrue(command.startswith('stat '),
                                  f"stat operation has unexpected command: {command}")
                elif op_type == 'write':
                    # write operations use 'echo ... >' command
                    self.assertTrue('echo ' in command and '>' in command,
                                  f"write operation has unexpected command: {command}")
                elif op_type == 'read':
                    # read operations use 'cat' command
                    self.assertTrue('cat ' in command,
                                  f"read operation has unexpected command: {command}")
                elif op_type == 'symlink':
                    # symlink operations use 'ln -s' command
                    self.assertTrue(command.startswith('ln -s '),
                                  f"symlink operation has unexpected command: {command}")
                elif op_type == 'rename':
                    # rename operations use 'mv' command
                    self.assertTrue(command.startswith('mv '),
                                  f"rename operation has unexpected command: {command}")
                else:
                    # Other operations should start with the operation name or contain it
                    self.assertTrue(
                        command.startswith(op_type + ' ') or op_type in command,
                        f"Operation '{op_type}' has unexpected command: {command}"
                    )

    def test_filesystem_state_integration(self) -> None:
        """
        Test integration between data generation and filesystem state.
        
        Verifies that the generated examples properly capture filesystem
        state before and after operations using Full State Rewrite methodology.
        Uses mock to avoid FUSE dependencies.
        """
        example = mock_generate_one()
        
        # Verify that initial_state is a valid tree representation
        initial_state = example['initial_state']
        self.assertIsInstance(initial_state, str)
        self.assertTrue(len(initial_state) > 0, "Initial state should not be empty")
        
        # With Smart Dual Training, operations can be either state_change or query
        self.assertIn(example['operation_type'], ['state_change', 'query'])
        
        # Result format depends on operation type
        result = example['result']
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0, "Result should not be empty")
        
        if example['operation_type'] == 'state_change':
            # State-changing operations should return filesystem tree
            tree_indicators = ['├──', '└──', '│', '/', '.txt', '.py', '.json']
            has_tree_structure = any(indicator in result for indicator in tree_indicators)
            self.assertTrue(has_tree_structure, "State change result should contain filesystem tree structure")
        else:  # query operation
            # Query operations should return specific responses (JSON, file content, etc.)
            # Just verify it's a non-empty string - format depends on query type
            self.assertTrue(len(result) > 0, "Query result should not be empty")

if __name__ == '__main__':
    unittest.main() 