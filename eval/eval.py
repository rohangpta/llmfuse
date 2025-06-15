"""
Unified evaluation pipeline for FUSE filesystem operations.

This module evaluates LLM performance on filesystem operations by testing
different models (Gemini via API, Qwen via Modal) against ground truth data.

Usage:
    python -m eval.eval --data data/fs_data.jsonl --models gemini,qwen3-1.7b,qwen3-4b
    python -m eval.eval --data data/fs_data.jsonl --models gemini --max-examples 100
    python -m eval.eval --data data/fs_data.jsonl --models qwen3-8b --output eval_qwen_8b.json
"""

import argparse
import json
import os
import sys
import tempfile
import time
import asyncio
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from src.fs_state import FSState
from src.llmfuse import LLMFS
from src.model import get_model_response
from src.utils import extract_result_from_llm_output

# Evaluation constants
DEFAULT_BATCH_SIZE = 10
SIMILARITY_THRESHOLD = 0.7
MAX_EXAMPLES_DEFAULT = 50
FUSE_OPERATION_TIMEOUT = 30

# Model configuration
QWEN_MODELS = {
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"
}
GEMINI_MODELS = {"gemini", "gemini-2.5-flash", "gemini-pro", "models/gemini-2.5-flash-preview-05-20"}

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

def calculate_similarity(predicted: str, expected: str, operation_type: str = "unknown") -> float:
    """
    Calculate similarity between predicted and expected results.
    """
    if not predicted or not expected:
        return 0.0
    
    predicted = predicted.strip()
    expected = expected.strip()
    
    # Exact match
    if predicted == expected:
        return 1.0
    
    # For query operations, use more flexible matching
    if operation_type == "query":
        # Remove extra whitespace and compare
        pred_normalized = ' '.join(predicted.split())
        exp_normalized = ' '.join(expected.split())
        
        if pred_normalized == exp_normalized:
            return 1.0
        
        # Check if key information is present
        if operation_type == "getattr" or "stat" in expected.lower():
            # For getattr operations, check if size/mode info is present
            pred_lower = predicted.lower()
            exp_lower = expected.lower()
            
            score = 0.0
            total_checks = 0
            
            # Check for size information
            if "size" in exp_lower:
                total_checks += 1
                if "size" in pred_lower:
                    score += 0.3
            
            # Check for mode information
            if "mode" in exp_lower or "permissions" in exp_lower:
                total_checks += 1
                if "mode" in pred_lower or any(perm in pred_lower for perm in ["644", "755", "rwx"]):
                    score += 0.3
            
            # Check for type information
            if "type" in exp_lower:
                total_checks += 1
                if "type" in pred_lower or any(t in pred_lower for t in ["file", "directory", "dir"]):
                    score += 0.4
            
            if total_checks > 0:
                return score
    
    # Fallback to simple word overlap
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    
    if not exp_words:
        return 0.0
    
    overlap = len(pred_words.intersection(exp_words))
    return overlap / len(exp_words)

def evaluate_single_example_local(example: Dict[str, Any], model_name: str = "gemini") -> Dict[str, Any]:
    """
    Evaluate a single training example using local model (Gemini API).
    """
    try:
        # Create the evaluation prompt
        prompt = f"""You are simulating a filesystem. Given the current state and operation, predict the exact result.

Current filesystem state:
{example['initial_state']}

Operation: {example['operation']}

Provide only the result as it would appear after the operation:"""

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
                'operation_type': example.get('operation_type', 'unknown'),
                'error': response
            }
        
        predicted = response.strip()
        expected = example['result']
        
        similarity = calculate_similarity(predicted, expected, example.get('operation_type', 'unknown'))
        correct = similarity > SIMILARITY_THRESHOLD
        
        return {
            'correct': correct,
            'predicted': predicted,
            'expected': expected,
            'operation': example['operation'],
            'operation_type': example.get('operation_type', 'unknown'),
            'similarity': similarity
        }
        
    except Exception as e:
        return {
            'correct': False,
            'predicted': f"Error: {str(e)}",
            'expected': example['result'],
            'operation': example['operation'],
            'operation_type': example.get('operation_type', 'unknown'),
            'error': str(e)
        }

def run_modal_evaluation(model_names: List[str], dataset_file: str) -> Dict[str, Any]:
    """
    Run evaluation using Modal for Qwen models.
    
    Args:
        model_names: List of Qwen model names to evaluate
        dataset_file: Path to dataset file
    
    Returns:
        Dictionary with evaluation results for all models
    """
    print(f"🚀 Running Modal evaluation for models: {', '.join(model_names)}")
    
    results = {}
    
    # Run evaluation for each model using modal run
    for model_name in model_names:
        print(f"🔥 Evaluating {model_name} via Modal...")
        
        try:
            # Use modal run to evaluate single model
            cmd = ['modal', 'run', 'modal_run.py::eval_single_size', '--model-name', model_name]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                # Try to parse the output for results
                # Modal functions typically print results to stdout
                output_lines = result.stdout.strip().split('\n')
                
                # Look for accuracy information in the output
                accuracy = 0.0
                total_examples = 0
                correct_predictions = 0
                
                for line in output_lines:
                    if 'accuracy' in line.lower() and '%' in line:
                        try:
                            # Extract percentage from line like "✅ qwen3-1.7b: 75.0% accuracy"
                            import re
                            match = re.search(r'(\d+\.?\d*)%', line)
                            if match:
                                accuracy = float(match.group(1)) / 100.0
                        except:
                            pass
                    elif 'correct:' in line.lower():
                        try:
                            # Extract numbers from line like "Correct: 6/8"
                            import re
                            match = re.search(r'(\d+)/(\d+)', line)
                            if match:
                                correct_predictions = int(match.group(1))
                                total_examples = int(match.group(2))
                        except:
                            pass
                
                results[model_name] = {
                    'model_name': model_name,
                    'total_examples': total_examples,
                    'correct_predictions': correct_predictions,
                    'overall_accuracy': accuracy,
                    'accuracy_by_type': {},
                    'detailed_results': [],
                    'modal_output': result.stdout
                }
                
                print(f"✅ {model_name}: {accuracy:.2%} accuracy ({correct_predictions}/{total_examples})")
                
            else:
                error_msg = f"Modal run failed: {result.stderr}"
                print(f"❌ {model_name}: {error_msg}")
                results[model_name] = {
                    'model_name': model_name,
                    'error': error_msg,
                    'overall_accuracy': 0.0,
                    'modal_stdout': result.stdout,
                    'modal_stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            error_msg = "Modal evaluation timed out (30 minutes)"
            print(f"❌ {model_name}: {error_msg}")
            results[model_name] = {
                'model_name': model_name,
                'error': error_msg,
                'overall_accuracy': 0.0
            }
            
        except Exception as e:
            error_msg = f"Modal evaluation failed: {str(e)}"
            print(f"❌ {model_name}: {error_msg}")
            results[model_name] = {
                'model_name': model_name,
                'error': error_msg,
                'overall_accuracy': 0.0
            }
    
    return results

def evaluate_dataset_unified(
    dataset_file: str,
    models: List[str], 
    output_file: str,
    max_examples: int = MAX_EXAMPLES_DEFAULT
) -> Dict[str, Any]:
    """
    Unified evaluation function that handles both local and Modal models.
    
    Args:
        dataset_file: Path to dataset file  
        models: List of model names to evaluate
        output_file: Path to save evaluation results
        max_examples: Maximum number of examples to evaluate
        
    Returns:
        Dictionary with evaluation statistics for all models
    """
    print(f"🎯 Starting unified evaluation")
    print(f"📂 Dataset: {dataset_file}")
    print(f"🤖 Models: {', '.join(models)}")
    print(f"📊 Max examples: {max_examples}")
    print("=" * 60)
    
    # Load dataset
    examples = load_dataset(dataset_file)
    if not examples:
        print(f"❌ Failed to load dataset from {dataset_file}")
        return {}
    
    # Limit number of examples
    if len(examples) > max_examples:
        print(f"📋 Limiting evaluation to {max_examples} examples (out of {len(examples)})")
        examples = examples[:max_examples]
    
    print(f"📊 Evaluating {len(examples)} examples")
    
    # Separate models by type
    local_models = [m for m in models if m in GEMINI_MODELS]
    qwen_models = [m for m in models if m in QWEN_MODELS]
    
    all_results = {}
    
    # Evaluate local models (Gemini)
    for model_name in local_models:
        print(f"\n🔥 Evaluating {model_name} (local)...")
        model_results = []
        
        for i, example in enumerate(examples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(examples)}")
            
            result = evaluate_single_example_local(example, model_name)
            model_results.append(result)
        
        # Calculate statistics
        stats = calculate_statistics(model_results)
        all_results[model_name] = {
            'model_name': model_name,
            'statistics': stats,
            'detailed_results': model_results,
            'evaluation_method': 'local_api'
        }
        
        print(f"✅ {model_name}: {stats['overall_accuracy']:.2%} accuracy")
    
    # Evaluate Qwen models via Modal
    if qwen_models:
        print(f"\n🚀 Evaluating Qwen models via Modal...")
        modal_results = run_modal_evaluation(qwen_models, dataset_file)
        
        for model_name in qwen_models:
            if model_name in modal_results:
                result = modal_results[model_name]
                if 'error' not in result:
                    all_results[model_name] = {
                        'model_name': model_name,
                        'statistics': {
                            'total_examples': result.get('total_examples', 0),
                            'correct_predictions': result.get('correct_predictions', 0),
                            'overall_accuracy': result.get('overall_accuracy', 0.0),
                            'accuracy_by_type': result.get('accuracy_by_type', {}),
                        },
                        'detailed_results': result.get('detailed_results', []),
                        'evaluation_method': 'modal_vllm'
                    }
                    print(f"✅ {model_name}: {result.get('overall_accuracy', 0.0):.2%} accuracy")
                else:
                    print(f"❌ {model_name}: {result['error']}")
                    all_results[model_name] = {
                        'model_name': model_name,
                        'error': result['error'],
                        'evaluation_method': 'modal_vllm'
                    }
    
    # Save comprehensive results
    output_data = {
        'evaluation_info': {
            'dataset_file': dataset_file,
            'models_evaluated': models,
            'total_examples_in_dataset': len(examples),
            'examples_evaluated': max_examples,
            'timestamp': time.time()
        },
        'model_results': all_results
    }
    
    print(f"\n💾 Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 UNIFIED EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, result in all_results.items():
        if 'error' in result:
            print(f"❌ {model_name}: ERROR - {result['error']}")
        else:
            stats = result.get('statistics', {})
            accuracy = stats.get('overall_accuracy', 0.0)
            correct = stats.get('correct_predictions', 0)
            total = stats.get('total_examples', 0)
            method = result.get('evaluation_method', 'unknown')
            print(f"✅ {model_name} ({method}): {accuracy:.2%} accuracy ({correct}/{total})")
    
    return all_results

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate evaluation statistics from results."""
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    accuracy = correct / total if total > 0 else 0.0
    
    # Statistics by operation type
    by_type = defaultdict(list)
    by_operation = defaultdict(list)
    
    for result in results:
        op_type = result.get('operation_type', 'unknown')
        operation = result.get('operation', 'unknown')
        by_type[op_type].append(result)
        by_operation[operation].append(result)
    
    type_stats = {}
    for op_type, type_results in by_type.items():
        type_correct = sum(1 for r in type_results if r.get('correct', False))
        type_stats[op_type] = {
            'total': len(type_results),
            'correct': type_correct,
            'accuracy': type_correct / len(type_results) if type_results else 0.0
        }
    
    operation_stats = {}
    for operation, op_results in by_operation.items():
        op_correct = sum(1 for r in op_results if r.get('correct', False))
        operation_stats[operation] = {
            'total': len(op_results),
            'correct': op_correct,
            'accuracy': op_correct / len(op_results) if op_results else 0.0
        }
    
    # Error examples
    error_examples = [r for r in results if not r.get('correct', False)][:5]
    
    return {
        'total_examples': total,
        'correct_predictions': correct,
        'overall_accuracy': accuracy,
        'accuracy_by_type': type_stats,
        'accuracy_by_operation': operation_stats,
        'error_examples': error_examples
    }

def load_hf_format(file_path: str) -> List[Dict[str, str]]:
    """Load HuggingFace JSONL format data."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples

def convert_hf_to_structured(hf_examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert HuggingFace format to structured format for evaluation."""
    structured_examples = []
    
    for hf_example in hf_examples:
        prompt = hf_example['prompt']
        completion = hf_example['completion']
        
        # Parse the prompt to extract initial_state and operation
        lines = prompt.split('\n')
        
        # Find the current filesystem state section
        state_start = -1
        state_end = -1
        operation_line = -1
        
        for i, line in enumerate(lines):
            if 'Current filesystem state:' in line:
                state_start = i + 1
            elif 'Operation:' in line:
                state_end = i
                operation_line = i
                break
        
        if state_start == -1 or operation_line == -1:
            # Fallback parsing for malformed prompts
            initial_state = ""
            operation = "unknown"
        else:
            # Extract initial state
            if state_end > state_start:
                initial_state = '\n'.join(lines[state_start:state_end]).strip()
            else:
                initial_state = ""
            
            # Extract operation
            operation = lines[operation_line].replace('Operation:', '').strip()
        
        # Determine operation type based on operation name
        operation_type = "state_change"  # Default assumption
        if any(op in operation.lower() for op in ['readdir', 'stat', 'ls', 'getattr']):
            operation_type = "query"
        
        structured_example = {
            'initial_state': initial_state,
            'operation': operation,
            'result': completion,
            'operation_type': operation_type
        }
        
        structured_examples.append(structured_example)
    
    return structured_examples

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset in either structured JSON or HuggingFace JSONL format."""
    try:
        if file_path.endswith('.jsonl'):
            # Assume HuggingFace format
            print("📋 Detected HuggingFace JSONL format")
            hf_examples = load_hf_format(file_path)
            return convert_hf_to_structured(hf_examples)
        else:
            # Assume structured JSON format
            print("📋 Detected structured JSON format")
            with open(file_path, 'r') as f:
                examples = json.load(f)
            return examples
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

def run_modal_analysis(results_filename: str) -> int:
    """
    Run Modal analysis on evaluation results.
    
    Args:
        results_filename: Name of the results file to analyze
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print(f"🔍 Running Modal analysis on {results_filename}")
    
    try:
        # Use modal run to analyze results (same as your original command)
        cmd = ['modal', 'run', 'modal_run.py::analyze_eval_results', '--results-filename', results_filename]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            print("🎉 Modal analysis complete!")
            print(result.stdout)
            return 0
        else:
            print(f"❌ Modal analysis failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return 1
            
    except subprocess.TimeoutExpired:
        print(f"❌ Modal analysis timed out (10 minutes)")
        return 1
        
    except Exception as e:
        print(f"❌ Modal analysis setup failed: {str(e)}")
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified LLM evaluation for FUSE filesystem operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Gemini and Qwen models
  python -m eval.eval --data data/fs_data.jsonl --models gemini,qwen3-1.7b,qwen3-4b
  
  # Evaluate only Gemini with custom output
  python -m eval.eval --data data/fs_data.jsonl --models gemini --output gemini_results.json
  
  # Evaluate multiple Qwen models via Modal
  python -m eval.eval --data data/fs_data.jsonl --models qwen3-8b,qwen3-14b --max-examples 200
  
  # Analyze existing evaluation results via Modal
  python -m eval.eval --analyze-results qwen3_eval_results_1749927278.json
        """
    )
    
    # Create mutually exclusive group for main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    
    action_group.add_argument(
        '--data', 
        help='Path to dataset file (JSON or JSONL format) - for evaluation mode'
    )
    
    action_group.add_argument(
        '--analyze-results',
        metavar='RESULTS_FILE',
        help='Analyze existing evaluation results via Modal (e.g., qwen3_eval_results_1749927278.json)'
    )
    
    parser.add_argument(
        '--models', 
        help='Comma-separated list of models to evaluate (required for evaluation mode)'
    )
    
    parser.add_argument(
        '--output',
        default='eval_results.json',
        help='Output file for results (default: eval_results.json) - evaluation mode only'
    )
    
    parser.add_argument(
        '--max-examples',
        type=int,
        default=MAX_EXAMPLES_DEFAULT,
        help=f'Maximum number of examples to evaluate (default: {MAX_EXAMPLES_DEFAULT}) - evaluation mode only'
    )
    
    args = parser.parse_args()
    
    # Handle analysis mode
    if args.analyze_results:
        return run_modal_analysis(args.analyze_results)
    
    # Handle evaluation mode
    if not args.data:
        print("❌ --data is required for evaluation mode")
        return 1
    
    if not args.models:
        print("❌ --models is required for evaluation mode")
        return 1
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"❌ Dataset file not found: {args.data}")
        return 1
    
    # Parse models
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models:
        print("❌ No models specified")
        return 1
    
    # Validate models
    invalid_models = []
    for model in models:
        if model not in GEMINI_MODELS and model not in QWEN_MODELS:
            invalid_models.append(model)
    
    if invalid_models:
        print(f"❌ Invalid models: {', '.join(invalid_models)}")
        print(f"✅ Supported models:")
        print(f"   Gemini: {', '.join(GEMINI_MODELS)}")
        print(f"   Qwen (via Modal): {', '.join(QWEN_MODELS)}")
        return 1
    
    # Check API key for Gemini models
    if any(m in GEMINI_MODELS for m in models):
        if not os.environ.get('GEMINI_API_KEY'):
            print("❌ GEMINI_API_KEY environment variable not set")
            print("   Required for Gemini model evaluation")
            return 1
    
    print("🚀 Starting unified evaluation pipeline...")
    
    # Run evaluation
    results = evaluate_dataset_unified(
        dataset_file=args.data,
        models=models,
        output_file=args.output,
        max_examples=args.max_examples
    )
    
    if results:
        print(f"\n🎉 Evaluation complete! Results saved to {args.output}")
        return 0
    else:
        print("\n❌ Evaluation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 