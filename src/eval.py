"""
LLM evaluation pipeline for FUSE filesystem operations.

This module evaluates LLM performance on filesystem operations by testing
the LLM-driven FUSE filesystem against ground truth data.
"""
# Standard library imports
import json
import os
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from .fs_state import FSState
from .llmfuse import LLMFS
from .model import get_model_response
from .utils import extract_result_from_llm_output

# Evaluation constants
DEFAULT_BATCH_SIZE = 10
SIMILARITY_THRESHOLD = 0.8
MAX_EXAMPLES_DEFAULT = 50
FUSE_OPERATION_TIMEOUT = 30

def parse_fuse_operation(operation_str: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse a FUSE operation string into components.
    
    Args:
        operation_str: String like "mkdir('/path', mode='0o755')"
        
    Returns:
        Tuple of (operation_name, path, parameters_dict)
    """
    # Handle both FUSE operations and shell commands
    if operation_str.startswith('FUSE '):
        operation_str = operation_str[5:]  # Remove 'FUSE ' prefix
    
    # Simple parsing - in production, you'd want more robust parsing
    if '(' in operation_str and ')' in operation_str:
        op_name = operation_str.split('(')[0]
        params_part = operation_str[operation_str.find('(')+1:operation_str.rfind(')')]
        
        # Extract path (first parameter)
        path = ""
        params = {}
        
        if params_part:
            parts = [p.strip() for p in params_part.split(',')]
            if parts:
                # First part is usually the path
                path_part = parts[0].strip("'\"")
                path = path_part
                
                # Parse additional parameters
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        params[key] = value
        
        return op_name, path, params
    else:
        # Fallback for shell commands
        return "shell", operation_str, {}

def simulate_fuse_operation(fs: LLMFS, operation: str, path: str, **params) -> Tuple[bool, Any]:
    """
    Simulate a FUSE operation on the LLM filesystem.
    
    Args:
        fs: LLMFS instance
        operation: Operation name (e.g., 'mkdir', 'readdir')
        path: Path for the operation
        **params: Additional parameters
        
    Returns:
        Tuple of (success, result)
    """
    try:
        if operation == 'mkdir':
            mode = int(params.get('mode', '0o755'), 8)
            fs.mkdir(path, mode)
            return True, None
            
        elif operation == 'create':
            mode = int(params.get('mode', '0o644'), 8)
            fs.create(path, mode)
            return True, None
            
        elif operation == 'unlink':
            fs.unlink(path)
            return True, None
            
        elif operation == 'rmdir':
            fs.rmdir(path)
            return True, None
            
        elif operation == 'chmod':
            mode = int(params.get('mode', '0o644'), 8)
            fs.chmod(path, mode)
            return True, None
            
        elif operation == 'chown':
            uid = int(params.get('uid', 0))
            gid = int(params.get('gid', 0))
            fs.chown(path, uid, gid)
            return True, None
            
        elif operation == 'readdir':
            result = fs.readdir(path, None)
            return True, result
            
        elif operation == 'getattr':
            result = fs.getattr(path)
            return True, result
            
        elif operation == 'rename':
            new_path = params.get('new_path', '')
            fs.rename(path, new_path)
            return True, None
            
        else:
            return False, f"Unknown operation: {operation}"
            
    except Exception as e:
        return False, str(e)

def evaluate_single_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single training example using the LLM-driven FUSE filesystem.
    
    Args:
        example: Training example with initial_state, operation, result, operation_type
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Initialize LLM filesystem with the initial state
        fs = LLMFS(initial_state=example['initial_state'])
        
        # Parse the operation
        operation_str = example['operation']
        op_name, path, params = parse_fuse_operation(operation_str)
        
        # Handle shell commands vs FUSE operations
        if op_name == 'shell':
            # For shell commands, we need to evaluate differently
            # This is a fallback for backwards compatibility
            return evaluate_shell_command_example(example)
        
        # Execute the operation
        success, result = simulate_fuse_operation(fs, op_name, path, **params)
        
        if not success:
            return {
                'correct': False,
                'predicted': f"Error: {result}",
                'expected': example['result'],
                'operation': operation_str,
                'operation_type': example['operation_type'],
                'error': f"Operation failed: {result}"
            }
        
        # Get the result based on operation type
        if example['operation_type'] == 'query':
            # For query operations, compare the direct result
            predicted = str(result) if result is not None else ""
            expected = example['result']
            
            # Simple string similarity for query results
            correct = calculate_similarity(predicted, expected) > SIMILARITY_THRESHOLD
            
        else:
            # For state-changing operations, compare the filesystem state
            predicted_state = fs._get_state_string()
            expected_state = example['result']
            
            correct = calculate_similarity(predicted_state, expected_state) > SIMILARITY_THRESHOLD
            predicted = predicted_state
            expected = expected_state
        
        return {
            'correct': correct,
            'predicted': predicted,
            'expected': expected,
            'operation': operation_str,
            'operation_type': example['operation_type']
        }
        
    except Exception as e:
        return {
            'correct': False,
            'predicted': f"Error: {str(e)}",
            'expected': example['result'],
            'operation': example['operation'],
            'operation_type': example['operation_type'],
            'error': str(e)
        }

def evaluate_shell_command_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback evaluation for shell command examples (backwards compatibility).
    
    Args:
        example: Training example with shell command operation
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Use the model to predict the result directly
        prompt = f"""Given the filesystem state and operation, predict the result.

Initial filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Expected result format: {'filesystem state tree' if example['operation_type'] == 'state_change' else 'command output'}

Result:"""

        predicted = get_model_response(prompt, temperature=0.0)
        predicted = extract_result_from_llm_output(predicted)
        
        expected = example['result']
        correct = calculate_similarity(predicted, expected) > SIMILARITY_THRESHOLD
        
        return {
            'correct': correct,
            'predicted': predicted,
            'expected': expected,
            'operation': example['operation'],
            'operation_type': example['operation_type']
        }
        
    except Exception as e:
        return {
            'correct': False,
            'predicted': f"Error: {str(e)}",
            'expected': example['result'],
            'operation': example['operation'],
            'operation_type': example['operation_type'],
            'error': str(e)
        }

def calculate_similarity(predicted: str, expected: str) -> float:
    """
    Calculate similarity between predicted and expected results.
    
    Args:
        predicted: Predicted result string
        expected: Expected result string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not predicted and not expected:
        return 1.0
    
    if not predicted or not expected:
        return 0.0
    
    # Normalize whitespace and compare
    pred_normalized = ' '.join(predicted.split())
    exp_normalized = ' '.join(expected.split())
    
    if pred_normalized == exp_normalized:
        return 1.0
    
    # Simple character-level similarity
    pred_chars = set(pred_normalized.lower())
    exp_chars = set(exp_normalized.lower())
    
    if not pred_chars and not exp_chars:
        return 1.0
    
    intersection = len(pred_chars.intersection(exp_chars))
    union = len(pred_chars.union(exp_chars))
    
    return intersection / union if union > 0 else 0.0

def evaluate_batch(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of examples.
    
    Args:
        examples: List of training examples
        
    Returns:
        List of evaluation results
    """
    results = []
    for i, example in enumerate(examples):
        print(f"Evaluating example {i+1}/{len(examples)}")
        result = evaluate_single_example(example)
        results.append(result)
    
    return results

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate evaluation statistics from results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with statistics
    """
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Statistics by operation type
    by_type = defaultdict(list)
    for result in results:
        by_type[result['operation_type']].append(result)
    
    type_stats = {}
    for op_type, type_results in by_type.items():
        type_correct = sum(1 for r in type_results if r['correct'])
        type_stats[op_type] = {
            'total': len(type_results),
            'correct': type_correct,
            'accuracy': type_correct / len(type_results) if type_results else 0.0
        }
    
    # Statistics by operation name
    by_operation = defaultdict(list)
    for result in results:
        op_name = result['operation'].split('(')[0].strip()
        if op_name.startswith('FUSE '):
            op_name = op_name[5:]
        by_operation[op_name].append(result)
    
    operation_stats = {}
    for op_name, op_results in by_operation.items():
        op_correct = sum(1 for r in op_results if r['correct'])
        operation_stats[op_name] = {
            'total': len(op_results),
            'correct': op_correct,
            'accuracy': op_correct / len(op_results) if op_results else 0.0
        }
    
    return {
        'total_examples': total,
        'correct_predictions': correct,
        'overall_accuracy': correct / total,
        'accuracy_by_type': type_stats,
        'accuracy_by_operation': operation_stats,
        'error_examples': [r for r in results if not r['correct']][:5]  # First 5 errors
    }

def evaluate_dataset(
    dataset_file: str, 
    output_file: str, 
    max_examples: int = MAX_EXAMPLES_DEFAULT
) -> Dict[str, Any]:
    """
    Evaluate the LLM on a dataset of filesystem operations.
    
    Args:
        dataset_file: Path to JSON file with training examples
        output_file: Path to save evaluation results
        max_examples: Maximum number of examples to evaluate
        
    Returns:
        Dictionary with evaluation statistics
    """
    print(f"Loading dataset from {dataset_file}")
    
    try:
        with open(dataset_file, 'r') as f:
            examples = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing dataset file: {e}")
        return {}
    
    if not examples:
        print("No examples found in dataset")
        return {}
    
    # Limit number of examples
    if len(examples) > max_examples:
        print(f"Limiting evaluation to {max_examples} examples (out of {len(examples)})")
        examples = examples[:max_examples]
    
    print(f"Evaluating {len(examples)} examples using LLM-driven FUSE filesystem")
    
    # Evaluate in batches
    all_results = []
    batch_size = DEFAULT_BATCH_SIZE
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(examples) + batch_size - 1)//batch_size}")
        
        batch_results = evaluate_batch(batch)
        all_results.extend(batch_results)
    
    # Calculate statistics
    stats = calculate_statistics(all_results)
    
    # Save results
    output_data = {
        'statistics': stats,
        'detailed_results': all_results,
        'evaluation_info': {
            'dataset_file': dataset_file,
            'total_examples_in_dataset': len(examples),
            'examples_evaluated': len(all_results),
            'evaluation_method': 'llm_fuse_filesystem'
        }
    }
    
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Correct predictions: {stats['correct_predictions']}")
    print(f"Overall accuracy: {stats['overall_accuracy']:.2%}")
    
    print("\nAccuracy by operation type:")
    for op_type, type_stats in stats['accuracy_by_type'].items():
        print(f"  {op_type}: {type_stats['accuracy']:.2%} ({type_stats['correct']}/{type_stats['total']})")
    
    print("\nAccuracy by operation:")
    for op_name, op_stats in stats['accuracy_by_operation'].items():
        print(f"  {op_name}: {op_stats['accuracy']:.2%} ({op_stats['correct']}/{op_stats['total']})")
    
    if stats['error_examples']:
        print(f"\nFirst few error examples:")
        for i, error in enumerate(stats['error_examples'][:3]):
            print(f"  {i+1}. Operation: {error['operation']}")
            print(f"     Expected: {error['expected'][:100]}...")
            print(f"     Predicted: {error['predicted'][:100]}...")
            if 'error' in error:
                print(f"     Error: {error['error']}")
    
    return stats

if __name__ == "__main__":
    # Example usage
    dataset_file = "data/training_data_linux.json"
    output_file = "data/eval_results_fuse.json"
    
    if os.path.exists(dataset_file):
        evaluate_dataset(dataset_file, output_file, max_examples=20)
    else:
        print(f"Dataset file not found: {dataset_file}")
        print("Please generate training data first using generate_data.py") 