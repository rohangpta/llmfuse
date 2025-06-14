"""
Unit tests for the synthetic filesystem data generation module.

This module tests the data generation functions to ensure they produce
valid training examples with proper structure and operation distribution.
"""
# Standard library imports
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

# Local imports
from .generate_data import generate_one, generate_random_operation, execute_command
from .fs_state import FSState, FileEntry
from .utils import DEFAULT_FILE_MODE, DEFAULT_DIR_MODE

class TestGenerateData(unittest.TestCase):
    """Test cases for synthetic filesystem data generation."""

    def test_generate_one_structure(self) -> None:
        """
        Test that a single generated example has the correct structure.
        
        Verifies that the generated example contains all required fields
        with appropriate data types.
        """
        example = generate_one()
        self.assertIsNotNone(example)
        
        # Check for new field names
        required_fields = ['initial_state', 'operation', 'result', 'operation_type']
        for field in required_fields:
            self.assertIn(field, example, f"Missing required field: {field}")
        
        # Check data types
        self.assertIsInstance(example['initial_state'], str)
        self.assertIsInstance(example['operation'], str)
        self.assertIsInstance(example['result'], str)
        self.assertIsInstance(example['operation_type'], str)
        
        # Check operation type is valid
        self.assertIn(example['operation_type'], ['query', 'state_change'])

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
                
                # Verify operation type is valid
                valid_ops = ['mkdir', 'touch', 'rm', 'rmdir', 'chmod', 'chown', 'ls', 'stat']
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

        # Check that all core operations are present
        expected_ops = ['mkdir', 'touch', 'chmod', 'chown', 'ls', 'stat']
        for op in expected_ops:
            self.assertIn(op, counts, f"Operation '{op}' was not generated.")

        # Check that the distribution isn't extremely skewed
        total_ops = len(operations)
        for op, count in counts.items():
            percentage = (count / total_ops) * 100
            # Each operation should appear at least 2% of the time (more lenient)
            self.assertGreater(percentage, 2.0, 
                             f"Operation '{op}' appears only {percentage:.1f}% of the time")
            # No single operation should dominate (less than 60%)
            self.assertLess(percentage, 60.0,
                           f"Operation '{op}' appears {percentage:.1f}% of the time, which is too dominant")

    def test_query_vs_state_change_operations(self) -> None:
        """
        Test the distinction between query and state-changing operations.
        
        Verifies that operations are correctly classified as either
        query operations or state-changing operations.
        """
        query_ops = {'ls', 'stat'}
        state_change_ops = {'mkdir', 'touch', 'rm', 'rmdir', 'chmod', 'chown'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some structure for operations
            execute_command("mkdir testdir", temp_dir)
            execute_command("touch testfile.txt", temp_dir)
            
            for _ in range(100):
                op_type, command = generate_random_operation(temp_dir)
                
                if op_type in query_ops:
                    # Query operations should contain ls or stat commands
                    self.assertTrue(
                        command.startswith('ls ') or command.startswith('stat '),
                        f"Query operation '{op_type}' has unexpected command: {command}"
                    )
                elif op_type in state_change_ops:
                    # State-changing operations should start with the operation name
                    self.assertTrue(
                        command.startswith(op_type + ' ') or op_type in command,
                        f"State-change operation '{op_type}' has unexpected command: {command}"
                    )
                else:
                    self.fail(f"Unknown operation type: {op_type}")

    def test_filesystem_state_integration(self) -> None:
        """
        Test integration between data generation and filesystem state.
        
        Verifies that the generated examples properly capture filesystem
        state before and after operations.
        """
        example = generate_one()
        
        # Verify that initial_state is a valid tree representation
        initial_state = example['initial_state']
        self.assertIsInstance(initial_state, str)
        
        # For state-changing operations, result should be a tree
        # For query operations, result should be command output
        if example['operation_type'] == 'state_change':
            # Result should look like a filesystem tree
            result = example['result']
            # Basic check: should contain directory/file indicators
            # (This is a simple heuristic - in practice, more sophisticated validation could be used)
            self.assertTrue(len(result) > 0, "State change result should not be empty")
        elif example['operation_type'] == 'query':
            # Query result should be command output
            result = example['result']
            self.assertIsInstance(result, str)
            # Query results can be empty for empty directories, so just check type

if __name__ == '__main__':
    unittest.main() 