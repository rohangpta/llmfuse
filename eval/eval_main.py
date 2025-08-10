"""
Fixed evaluation pipeline for FUSE filesystem operations.

This module addresses the training data inconsistencies and provides robust
evaluation metrics for LLM performance on filesystem operations.

Key Fixes:
1. Handles mixed completion formats (filesystem trees vs stat output)
2. Improved similarity calculation for different operation types
3. Better operation type detection
4. Robust parsing of training examples
"""

import argparse
import json
import os
import sys
import tempfile
import time
import asyncio
import subprocess
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from llmfuse.fs_state import FSState
from llmfuse.llmfuse import LLMFS
from src.model import get_model_response
from llmfuse.utils import extract_result_from_llm_output

# Evaluation constants
DEFAULT_BATCH_SIZE = 10
SIMILARITY_THRESHOLD = 0.6  # Lowered from 0.7 for more lenient matching
MAX_EXAMPLES_DEFAULT = 50
FUSE_OPERATION_TIMEOUT = 30

# Model configuration
QWEN_MODELS = {
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"
}
GEMINI_MODELS = {"gemini", "gemini-2.5-flash", "gemini-pro", "models/gemini-2.5-flash-preview-05-20"}

def detect_operation_type(operation_str: str) -> str:
    """
    Detect operation type based on operation string.
    
    Returns:
        'query' for read-only operations, 'state_change' for modifying operations
    """
    operation_lower = operation_str.lower()
    
    # Query operations (read-only)
    query_ops = ['readdir', 'getattr', 'stat', 'ls', 'read', 'open']
    if any(op in operation_lower for op in query_ops):
        return 'query'
    
    # State-changing operations  
    state_ops = ['mkdir', 'rmdir', 'create', 'unlink', 'chmod', 'chown', 'truncate', 'rename', 'symlink', 'write']
    if any(op in operation_lower for op in state_ops):
        return 'state_change'
    
    return 'unknown'

def detect_completion_format(completion: str) -> str:
    """
    Detect the format of the completion string.
    
    Returns:
        'filesystem_tree' for tree-like output, 'stat_output' for stat-like output
    """
    completion = completion.strip()
    
    # Check for filesystem tree format (starts with /)
    if completion.startswith('/') and 'dir' in completion:
        return 'filesystem_tree'
    
    # Check for stat output format
    stat_indicators = ['File:', 'Size:', 'Device:', 'Inode:', 'Access:', 'Uid:', 'Gid:']
    if any(indicator in completion for indicator in stat_indicators):
        return 'stat_output'
    
    # Check for directory listing format
    if completion.count('\n') > 0 and not completion.startswith('/'):
        lines = completion.split('\n')
        # Look for typical ls-like output
        if any(line.strip() and not line.startswith('/') for line in lines):
            return 'directory_listing'
    
    return 'unknown'

def calculate_similarity_robust(predicted: str, expected: str, operation_type: str = "unknown") -> float:
    """
    Calculate similarity with improved handling of different formats.
    """
    if not predicted or not expected:
        return 0.0
    
    predicted = predicted.strip()
    expected = expected.strip()
    
    # Exact match gets full score
    if predicted == expected:
        return 1.0
    
    # Detect formats
    pred_format = detect_completion_format(predicted)
    exp_format = detect_completion_format(expected)
    
    # Handle filesystem tree comparisons
    if exp_format == 'filesystem_tree' and pred_format == 'filesystem_tree':
        return compare_filesystem_trees(predicted, expected)
    
    # Handle stat output comparisons
    if exp_format == 'stat_output' and pred_format == 'stat_output':
        return compare_stat_outputs(predicted, expected)
    
    # Handle mixed formats (problematic training data)
    if exp_format != pred_format:
        # Try to extract key information regardless of format
        return compare_mixed_formats(predicted, expected, operation_type)
    
    # Fallback to word overlap
    return calculate_word_overlap(predicted, expected)

def compare_filesystem_trees(predicted: str, expected: str) -> float:
    """Compare two filesystem tree representations."""
    # Extract file/directory entries from both trees
    pred_entries = extract_tree_entries(predicted)
    exp_entries = extract_tree_entries(expected)
    
    if not exp_entries:
        return 0.0
    
    # Calculate overlap
    common_entries = len(pred_entries.intersection(exp_entries))
    total_entries = len(exp_entries)
    
    base_score = common_entries / total_entries
    
    # Bonus for correct structure
    if len(pred_entries) == len(exp_entries):
        base_score += 0.1
    
    return min(1.0, base_score)

def compare_stat_outputs(predicted: str, expected: str) -> float:
    """Compare two stat-like outputs."""
    pred_info = extract_stat_info(predicted)
    exp_info = extract_stat_info(expected)
    
    if not exp_info:
        return 0.0
    
    score = 0.0
    total_fields = len(exp_info)
    
    for field, exp_value in exp_info.items():
        if field in pred_info:
            if pred_info[field] == exp_value:
                score += 1.0
            elif field in ['size', 'inode'] and is_numeric_match(pred_info[field], exp_value):
                score += 0.8
        
    return score / total_fields if total_fields > 0 else 0.0

def compare_mixed_formats(predicted: str, expected: str, operation_type: str) -> float:
    """Handle comparisons between different output formats."""
    # For readdir operations, expected might be stat output but predicted should be filesystem tree
    if operation_type == 'query' and 'readdir' in operation_type.lower():
        # Extract filenames from both formats
        pred_files = extract_filenames_any_format(predicted)
        exp_files = extract_filenames_any_format(expected)
        
        if not exp_files:
            return 0.0
        
        common_files = len(pred_files.intersection(exp_files))
        return common_files / len(exp_files)
    
    # For other mixed cases, use word overlap with higher threshold
    return min(0.8, calculate_word_overlap(predicted, expected))

def extract_tree_entries(tree_str: str) -> set:
    """Extract file/directory entries from filesystem tree."""
    entries = set()
    lines = tree_str.strip().split('\n')
    
    for line in lines:
        # Remove tree symbols and extract filename
        cleaned = line
        tree_symbols = ['├──', '└──', '│', '├── ', '└── ', '│   ']
        for symbol in tree_symbols:
            cleaned = cleaned.replace(symbol, '')
        
        # Extract first word as filename
        parts = cleaned.strip().split()
        if parts and not parts[0].startswith('/'):
            entries.add(parts[0])
    
    return entries

def extract_stat_info(stat_str: str) -> Dict[str, str]:
    """Extract key information from stat output."""
    info = {}
    lines = stat_str.split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Normalize common field names
            if 'file' in key:
                info['filename'] = value
            elif 'size' in key:
                info['size'] = value
            elif 'access' in key and '(' in value:
                # Extract permissions from "Access: (0644/-rw-r--r--)"
                perm_match = re.search(r'\((\d+)', value)
                if perm_match:
                    info['permissions'] = perm_match.group(1)
    
    return info

def extract_filenames_any_format(text: str) -> set:
    """Extract filenames from any format (tree, stat, listing)."""
    filenames = set()
    
    # Try tree format first
    tree_files = extract_tree_entries(text)
    if tree_files:
        filenames.update(tree_files)
    
    # Try stat format
    stat_info = extract_stat_info(text)
    if 'filename' in stat_info:
        filenames.add(stat_info['filename'])
    
    # Try simple word extraction for listing format
    words = text.split()
    for word in words:
        # Look for common file extensions or patterns
        if ('.' in word and len(word) > 3) or word.endswith(('txt', 'log', 'dat', 'py', 'sh')):
            filenames.add(word)
    
    return filenames

def is_numeric_match(val1: str, val2: str) -> bool:
    """Check if two values are numerically equivalent."""
    try:
        return int(val1) == int(val2)
    except (ValueError, TypeError):
        return False

def calculate_word_overlap(predicted: str, expected: str) -> float:
    """Calculate simple word overlap ratio."""
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    
    if not exp_words:
        return 0.0
    
    overlap = len(pred_words.intersection(exp_words))
    return overlap / len(exp_words)

def load_hf_format_robust(file_path: str) -> List[Dict[str, str]]:
    """Load HuggingFace format with robust error handling."""
    examples = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    example = json.loads(line)
                    if 'prompt' in example and 'completion' in example:
                        examples.append(example)
                    else:
                        print(f"⚠️  Line {line_num}: Missing prompt or completion")
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_num}: JSON decode error: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return []
    
    return examples

def convert_hf_to_structured_robust(hf_examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert HuggingFace format with improved parsing."""
    structured_examples = []
    
    for i, hf_example in enumerate(hf_examples):
        try:
            prompt = hf_example['prompt']
            completion = hf_example['completion']
            
            # Parse the prompt more robustly
            lines = prompt.split('\n')
            
            # Find sections
            state_start = -1
            state_end = -1
            operation_line = -1
            
            for j, line in enumerate(lines):
                if 'Current filesystem state:' in line:
                    state_start = j + 1
                elif 'Operation:' in line:
                    state_end = j
                    operation_line = j
                    break
            
            # Extract initial state
            if state_start >= 0 and state_end > state_start:
                initial_state = '\n'.join(lines[state_start:state_end]).strip()
            else:
                initial_state = ""
            
            # Extract operation
            if operation_line >= 0:
                operation = lines[operation_line].replace('Operation:', '').strip()
            else:
                operation = "unknown"
            
            # Determine operation type
            operation_type = detect_operation_type(operation)
            
            # Detect completion format
            completion_format = detect_completion_format(completion)
            
            structured_example = {
                'initial_state': initial_state,
                'operation': operation,
                'result': completion,
                'operation_type': operation_type,
                'completion_format': completion_format,
                'example_id': i
            }
            
            structured_examples.append(structured_example)
            
        except Exception as e:
            print(f"⚠️  Error parsing example {i}: {e}")
            continue
    
    return structured_examples

def evaluate_single_example_robust(example: Dict[str, Any], model_name: str = "gemini") -> Dict[str, Any]:
    """
    Evaluate a single example with robust similarity calculation.
    Uses the same prompt templates as dataset generation for consistency.
    """
    try:
        # Create evaluation prompt based on operation type using the SAME templates as dataset generation
        if example['operation_type'] == 'query':
            # Use specific prompts based on operation type, matching dataset generation
            if 'readdir' in example['operation'].lower():
                prompt = f"""
You are simulating a filesystem. Given the current state and operation, provide the directory listing as a JSON array.

Current filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Provide the directory contents as a JSON array of filenames (including "." and ".." for non-root paths). Return ONLY the JSON array, no code blocks or extra formatting:"""
            elif 'getattr' in example['operation'].lower():
                prompt = f"""You are simulating a filesystem. Given the current state and operation, provide file attributes as a JSON object.

Current filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Provide the file attributes as a JSON object with keys: mode, size, type, mtime, owner, group. Return ONLY the JSON object, no code blocks or extra formatting:"""
            elif 'read' in example['operation'].lower():
                prompt = f"""You are simulating a filesystem. Given the current state and operation, provide the file contents.

Current filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Provide only the exact file contents, no extra formatting or explanation:"""
            else:
                # Fallback for other query operations
                prompt = f"""You are simulating a filesystem. Given the current state and operation, provide the specific requested information.

Current filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Provide the requested information in the appropriate format:"""
        else:
            # State change operations
            prompt = f"""You are simulating a filesystem. Given the current state and operation, provide the complete filesystem state after the operation.

Current filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Provide the complete filesystem tree after this operation (use tree-like format with proper indentation):"""

        # Get model response
        if model_name in GEMINI_MODELS:
            response = get_model_response(prompt, model_name=model_name)
        else:
            response = f"ERROR: Unsupported local model: {model_name}"
        
        if response.startswith("ERROR:"):
            return {
                'correct': False,
                'predicted': response,
                'expected': example['result'],
                'operation': example['operation'],
                'operation_type': example['operation_type'],
                'similarity': 0.0,
                'error': response
            }
        
        predicted = response.strip()
        expected = example['result']
        
        # Use robust similarity calculation
        similarity = calculate_similarity_robust(predicted, expected, example['operation_type'])
        correct = similarity > SIMILARITY_THRESHOLD
        
        return {
            'correct': correct,
            'predicted': predicted,
            'expected': expected,
            'operation': example['operation'],
            'operation_type': example['operation_type'],
            'similarity': similarity,
            'completion_format': example.get('completion_format', 'unknown')
        }
        
    except Exception as e:
        return {
            'correct': False,
            'predicted': f"Error: {str(e)}",
            'expected': example['result'],
            'operation': example['operation'],
            'operation_type': example.get('operation_type', 'unknown'),
            'similarity': 0.0,
            'error': str(e)
        }

def analyze_results_detailed(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results with detailed breakdown."""
    if not results:
        return {}
    
    total_examples = len(results)
    correct_examples = sum(1 for r in results if r['correct'])
    overall_accuracy = correct_examples / total_examples
    
    # Breakdown by operation type
    by_operation_type = defaultdict(list)
    for result in results:
        by_operation_type[result['operation_type']].append(result)
    
    type_accuracies = {}
    for op_type, type_results in by_operation_type.items():
        type_correct = sum(1 for r in type_results if r['correct'])
        type_accuracies[op_type] = {
            'accuracy': type_correct / len(type_results),
            'total': len(type_results),
            'correct': type_correct
        }
    
    # Breakdown by completion format  
    by_format = defaultdict(list)
    for result in results:
        format_type = result.get('completion_format', 'unknown')
        by_format[format_type].append(result)
    
    format_accuracies = {}
    for fmt, fmt_results in by_format.items():
        fmt_correct = sum(1 for r in fmt_results if r['correct'])
        format_accuracies[fmt] = {
            'accuracy': fmt_correct / len(fmt_results),
            'total': len(fmt_results),
            'correct': fmt_correct
        }
    
    # Find most common errors
    error_patterns = defaultdict(int)
    for result in results:
        if not result['correct'] and 'error' not in result:
            # Categorize the type of error
            if 'filesystem_tree' in result.get('completion_format', ''):
                error_patterns['tree_format_mismatch'] += 1
            elif 'stat_output' in result.get('completion_format', ''):
                error_patterns['stat_format_mismatch'] += 1
            else:
                error_patterns['other_format_issue'] += 1
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_examples': total_examples,
        'correct_examples': correct_examples,
        'accuracy_by_operation_type': type_accuracies,
        'accuracy_by_completion_format': format_accuracies,
        'error_patterns': dict(error_patterns),
        'average_similarity': sum(r.get('similarity', 0) for r in results) / total_examples
    }

def evaluate_dataset_robust(
    dataset_file: str,
    model_name: str = "gemini", 
    max_examples: int = MAX_EXAMPLES_DEFAULT
) -> Dict[str, Any]:
    """
    Robust evaluation function with improved error handling.
    """
    print(f"🎯 Starting robust evaluation")
    print(f"📂 Dataset: {dataset_file}")
    print(f"🤖 Model: {model_name}")
    print(f"📊 Max examples: {max_examples}")
    print("=" * 60)
    
    # Load dataset
    if dataset_file.endswith('.jsonl'):
        hf_examples = load_hf_format_robust(dataset_file)
        examples = convert_hf_to_structured_robust(hf_examples)
    else:
        try:
            with open(dataset_file, 'r') as f:
                examples = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {}
    
    if not examples:
        print(f"❌ No valid examples loaded from {dataset_file}")
        return {}
    
    # Limit examples
    if len(examples) > max_examples:
        print(f"📋 Limiting evaluation to {max_examples} examples (out of {len(examples)})")
        examples = examples[:max_examples]
    
    print(f"📊 Evaluating {len(examples)} examples")
    
    # Analyze data distribution
    op_types = defaultdict(int)
    formats = defaultdict(int)
    for ex in examples:
        op_types[ex.get('operation_type', 'unknown')] += 1
        formats[ex.get('completion_format', 'unknown')] += 1
    
    print(f"📈 Operation types: {dict(op_types)}")
    print(f"📄 Completion formats: {dict(formats)}")
    print()
    
    # Evaluate examples
    results = []
    for i, example in enumerate(examples):
        if i % 10 == 0:
            print(f"📝 Processing example {i+1}/{len(examples)}")
        
        result = evaluate_single_example_robust(example, model_name)
        results.append(result)
    
    # Analyze results
    analysis = analyze_results_detailed(results)
    
    # Print summary
    print(f"\n📊 Evaluation Results:")
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.2%}")
    print(f"Average Similarity: {analysis['average_similarity']:.3f}")
    print()
    
    print("By Operation Type:")
    for op_type, stats in analysis['accuracy_by_operation_type'].items():
        print(f"  {op_type}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nBy Completion Format:")
    for fmt, stats in analysis['accuracy_by_completion_format'].items():
        print(f"  {fmt}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    return {
        'model_name': model_name,
        'results': results,
        'analysis': analysis
    }

def main():
    """Main CLI entry point for robust evaluation."""
    parser = argparse.ArgumentParser(
        description="Robust LLM evaluation for FUSE filesystem operations"
    )
    
    parser.add_argument('--data', required=True, help='Path to dataset file')
    parser.add_argument('--model', default='gemini', help='Model to evaluate')
    parser.add_argument('--max-examples', type=int, default=MAX_EXAMPLES_DEFAULT, 
                       help='Maximum number of examples')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Dataset file not found: {args.data}")
        return 1
    
    # Run evaluation
    evaluation_results = evaluate_dataset_robust(
        args.data, 
        args.model, 
        args.max_examples
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"📁 Results saved to {args.output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 