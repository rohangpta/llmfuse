"""
Model evaluation and testing functions.
Contains comprehensive evaluation and analysis functionality.
"""

from eval.common import (
    app, vllm_image, model_cache, vllm_cache, eval_results_volume,
    MODEL_CACHE_PATH, VLLM_CACHE_PATH, EVAL_RESULTS_PATH, MINUTES,
    _get_vllm_response, _get_gemini_response, load_evaluation_dataset,
    calculate_similarity, QWEN3_MODELS
)
import modal
import json
import os
import subprocess
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
import shutil


async def evaluate_model_on_dataset(
    llm, model_name: str, dataset: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate a model on a dataset."""
    print(f"🔍 Evaluating {model_name} on {len(dataset)} examples...")
    
    results = []
    correct_predictions = 0
    total_examples = len(dataset)
    
    for i, example in enumerate(dataset):
        print(f"📝 Processing example {i+1}/{total_examples}")
        
        prompt = example.get("prompt", "")
        expected_output = example.get("completion", "")
        
        if not prompt:
            print(f"⚠️  Skipping example {i+1}: No prompt found")
            continue
            
        try:
            # Get model response
            if hasattr(llm, 'generate_response'):
                predicted_output = await llm.generate_response(prompt)
            else:
                # Direct function call for Gemini
                predicted_output = llm(prompt)
            
            # Calculate similarity
            similarity = calculate_similarity(predicted_output, expected_output)
            
            # Consider it correct if similarity > 0.7
            is_correct = similarity > 0.7
            if is_correct:
                correct_predictions += 1
            
            result = {
                "example_id": i + 1,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "expected": expected_output[:100] + "..." if len(expected_output) > 100 else expected_output,
                "predicted": predicted_output[:100] + "..." if len(predicted_output) > 100 else predicted_output,
                "similarity": similarity,
                "correct": is_correct,
            }
            
            results.append(result)
            
            # Log progress every 10 examples
            if (i + 1) % 10 == 0:
                current_accuracy = correct_predictions / (i + 1)
                print(f"📊 Progress: {i+1}/{total_examples}, Accuracy: {current_accuracy:.2%}")
        
        except Exception as e:
            print(f"❌ Error processing example {i+1}: {str(e)}")
            results.append({
                "example_id": i + 1,
                "error": str(e),
                "correct": False,
            })
    
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    
    evaluation_results = {
        "model_name": model_name,
        "total_examples": total_examples,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "detailed_results": results,
    }
    
    print(f"✅ Evaluation complete for {model_name}: {accuracy:.2%} accuracy")
    return evaluation_results


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
        EVAL_RESULTS_PATH: eval_results_volume,
    },
    timeout=20 * MINUTES,  # 20 minutes per model (faster with H100)
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def eval_single_size(model_name: str, dataset_file: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate a single model size on the given dataset."""
    
    # Load dataset
    if dataset_file is None:
        dataset_file = "/root/data/training_data_100.jsonl"
    
    dataset = load_evaluation_dataset(dataset_file)
    if not dataset:
        return {"error": f"Failed to load dataset from {dataset_file}"}
    
    # Use a subset for faster evaluation
    max_examples = 50
    if len(dataset) > max_examples:
        dataset = dataset[:max_examples]
        print(f"📊 Using first {max_examples} examples for evaluation")
    
    print(f"🚀 Starting evaluation for {model_name} with {len(dataset)} examples")
    
    # Start vLLM server
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name
    
    print(f"🔥 Starting vLLM server for {full_model_name}")
    
    # Clear any corrupted compilation cache first
    cache_dir = "/root/.cache/vllm/torch_compile_cache"
    if os.path.exists(cache_dir):
        print("🧹 Clearing vLLM compilation cache to prevent corruption issues...")
        shutil.rmtree(cache_dir)
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    
    # Start the server
    server_process = subprocess.Popen(cmd)
    
    # Wait for server to be ready with health checks
    print("⏳ Waiting for vLLM server to be ready (up to 10 minutes for first-time compilation)...")
    max_attempts = 60  # 10 minutes with 10-second intervals
    
    for attempt in range(max_attempts):
        await asyncio.sleep(10)
        
        # Try to connect to the server
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8000/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        print(f"✅ vLLM server is ready after {(attempt + 1) * 10} seconds")
                        break
        except Exception as e:
            print(f"🔄 Attempt {attempt + 1}: Server not ready yet ({str(e)[:50]}...)")
            continue
    else:
        print("❌ Server failed to start within timeout period")
        # Try one more time with a simple test request
        try:
            test_response = await _get_vllm_response(
                "http://localhost:8000", "test", model_name
            )
            if not test_response.startswith("ERROR:"):
                print("✅ Server responding to test request")
            else:
                print(f"⚠️  Server test failed: {test_response[:100]}...")
        except Exception as e:
            print(f"⚠️  Server test exception: {e}")
    
    # Give a bit more time for full initialization
    await asyncio.sleep(10)
    
    try:
        # Create LLM wrapper
        class VLLMLLMWrapper:
            def __init__(self, server_url, model_name):
                self.server_url = server_url
                self.model_name = model_name
            
            async def generate_response(self, prompt: str) -> str:
                return await _get_vllm_response(
                    self.server_url, prompt, self.model_name
                )
        
        llm = VLLMLLMWrapper("http://localhost:8000", model_name)
        
        # Run evaluation with detailed logging
        results = await evaluate_model_on_dataset_with_details(llm, model_name, dataset)
        
        # Save detailed results to JSON file
        results_filename = f"eval_results_{model_name}_detailed.json"
        results_path = f"{EVAL_RESULTS_PATH}/{results_filename}"
        
        os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Commit to volume so we can access it
        eval_results_volume.commit()
        
        print(f"💾 Detailed results saved to {results_path}")
        print(f"📊 Final accuracy: {results.get('accuracy', 0):.2%}")
        
        return results
    
    finally:
        # Clean up server more gracefully
        print("🧹 Cleaning up vLLM server...")
        try:
            # First try graceful termination
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
                print("✅ Server terminated gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't terminate gracefully, forcing shutdown...")
                server_process.kill()
                server_process.wait(timeout=5)
                print("✅ Server killed successfully")
        except Exception as e:
            print(f"⚠️  Error during server cleanup: {e}")
            try:
                server_process.kill()
                print("✅ Server force-killed as fallback")
            except:
                print("⚠️  Could not kill server process")


async def evaluate_model_on_dataset_with_details(
    llm, model_name: str, dataset: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate a model on a dataset with detailed logging."""
    print(f"🔍 Evaluating {model_name} on {len(dataset)} examples...")
    
    results = []
    correct_predictions = 0
    total_examples = len(dataset)
    
    for i, example in enumerate(dataset):
        print(f"📝 Processing example {i+1}/{total_examples}")
        
        prompt = example.get("prompt", "")
        expected_output = example.get("completion", "")
        
        if not prompt:
            print(f"⚠️  Skipping example {i+1}: No prompt found")
            continue
            
        try:
            # Get model response
            predicted_output = await llm.generate_response(prompt)
            
            # Calculate similarity
            similarity = calculate_similarity(predicted_output, expected_output)
            
            # Consider it correct if similarity > 0.7
            is_correct = similarity > 0.7
            if is_correct:
                correct_predictions += 1
            
            # Log first few examples in detail
            if i < 3:
                print(f"🔍 Example {i+1} details:")
                print(f"   Prompt (first 100 chars): {prompt[:100]}...")
                print(f"   Expected (first 100 chars): {expected_output[:100]}...")
                print(f"   Predicted (first 100 chars): {predicted_output[:100]}...")
                print(f"   Similarity: {similarity:.3f}")
                print(f"   Correct: {is_correct}")
            
            result = {
                "example_id": i + 1,
                "prompt": prompt,
                "expected": expected_output,
                "predicted": predicted_output,
                "similarity": similarity,
                "correct": is_correct,
            }
            
            results.append(result)
            
            # Log progress every 10 examples
            if (i + 1) % 10 == 0:
                current_accuracy = correct_predictions / (i + 1)
                print(f"📊 Progress: {i+1}/{total_examples}, Accuracy: {current_accuracy:.2%}")
        
        except Exception as e:
            print(f"❌ Error processing example {i+1}: {str(e)}")
            results.append({
                "example_id": i + 1,
                "error": str(e),
                "correct": False,
            })
    
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    
    evaluation_results = {
        "model_name": model_name,
        "total_examples": total_examples,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "detailed_results": results,
    }
    
    print(f"✅ Evaluation complete for {model_name}: {accuracy:.2%} accuracy")
    return evaluation_results


@app.function(
    image=vllm_image,
    volumes={EVAL_RESULTS_PATH: eval_results_volume},
    timeout=40 * MINUTES,  # 40 minutes total timeout (faster with H100s)
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def eval_sizes(dataset_file: Optional[str] = None):
    """Evaluate all Qwen3 model sizes."""
    print("🚀 Starting comprehensive evaluation of all Qwen3 sizes")
    
    model_sizes = ["qwen3-4b", "qwen3-8b"]  # Focus on practical sizes
    results = {}
    
    for model_name in model_sizes:
        print(f"\n🔍 Evaluating {model_name}...")
        try:
            model_results = await eval_single_size.remote.aio(model_name, dataset_file)
            results[model_name] = model_results
            print(f"✅ Completed evaluation for {model_name}")
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # Save consolidated results
    results_filename = f"eval_results_all_sizes.json"
    results_path = f"{EVAL_RESULTS_PATH}/{results_filename}"
    
    os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Commit to volume
    eval_results_volume.commit()
    
    print(f"💾 Results saved to {results_path}")
    
    # Print summary
    print("\n📊 EVALUATION SUMMARY:")
    for model_name, result in results.items():
        if "error" in result:
            print(f"  {model_name:12}: ERROR - {result['error']}")
        else:
            accuracy = result.get("accuracy", 0)
            print(f"  {model_name:12}: {accuracy:.2%} accuracy")
    
    return results


@app.function(
    image=vllm_image,
    volumes={EVAL_RESULTS_PATH: eval_results_volume},
    timeout=10 * MINUTES,
)
def analyze_eval_results(results_filename: str):
    """Analyze and visualize evaluation results."""
    results_path = f"{EVAL_RESULTS_PATH}/{results_filename}"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return f"Error: {results_filename} not found"
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    print(f"📊 Analyzing results from {results_filename}")
    print("=" * 60)
    
    # Overall summary
    model_accuracies = []
    for model_name, result in results.items():
        if "error" not in result and "accuracy" in result:
            accuracy = result["accuracy"]
            model_accuracies.append((model_name, accuracy))
            
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   Accuracy: {accuracy:.2%}")
            print(f"   Correct: {result.get('correct_predictions', 0)}/{result.get('total_examples', 0)}")
    
    # Rank models by performance
    if model_accuracies:
        print(f"\n🏆 MODEL RANKING:")
        sorted_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)
        for i, (model, acc) in enumerate(sorted_models, 1):
            print(f"   {i}. {model}: {acc:.2%}")
    
    # Best model recommendation
    if model_accuracies:
        best_model, best_acc = max(model_accuracies, key=lambda x: x[1])
        print(f"\n⭐ RECOMMENDATION: {best_model} ({best_acc:.2%} accuracy)")
    
    print("=" * 60)
    return f"Analysis complete for {results_filename}"


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
        EVAL_RESULTS_PATH: eval_results_volume,
        "/root/trained_models": modal.Volume.from_name(
            "qwen3-trained-models", create_if_missing=True
        ),
    },
    timeout=30 * MINUTES,
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def eval_trained_model(
    dataset_file: str, max_examples: int = 100
) -> Dict[str, Any]:
    """Evaluate a trained model on the filesystem dataset."""
    print(f"🔍 Evaluating trained model on filesystem tasks")
    
    # Load the evaluation dataset
    dataset = load_evaluation_dataset(dataset_file)
    if not dataset:
        return {"error": f"Failed to load dataset from {dataset_file}"}
    
    # Limit examples for faster evaluation
    if len(dataset) > max_examples:
        dataset = dataset[:max_examples]
        print(f"📊 Using first {max_examples} examples for evaluation")
    
    print(f"📊 Dataset size: {len(dataset)} examples")
    
    # Load the trained model (assuming it's been saved to the trained_models volume)
    model_path = "/root/trained_models/qwen3-8b-sft-3epochs-distributed"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        available_models = []
        if os.path.exists("/root/trained_models"):
            available_models = os.listdir("/root/trained_models")
        return {
            "error": f"Trained model not found at {model_path}",
            "available_models": available_models
        }
    
    print(f"🤖 Loading trained model from {model_path}")
    
    # Start vLLM server with the trained model
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    
    def read_output():
        for line in iter(server_process.stdout.readline, b''):
            line_str = line.decode('utf-8').strip()
            if line_str:
                print(f"[vLLM] {line_str}")
    
    # Start the server
    print("🔥 Starting vLLM server for trained model...")
    server_process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT
    )
    
    # Wait for server to start
    await asyncio.sleep(60)  # Give extra time for model loading
    
    try:
        class TrainedModelLLM:
            def __init__(self, server_url, model_path):
                self.server_url = server_url
                self.model_path = model_path
            
            async def generate_response(self, prompt: str) -> str:
                return await _get_vllm_response(
                    self.server_url, prompt, "trained-model"
                )
        
        llm = TrainedModelLLM("http://localhost:8000", model_path)
        
        # Test server connectivity
        try:
            test_response = await llm.generate_response("Hello, how are you?")
            print(f"🧪 Server test response: {test_response[:100]}...")
        except Exception as e:
            print(f"❌ Server connectivity test failed: {e}")
            return {"error": f"Server not responding: {e}"}
        
        # Run comprehensive evaluation
        print("🔍 Starting comprehensive evaluation...")
        results = await evaluate_model_on_dataset(llm, "trained-qwen3-8b", dataset)
        
        # Save results
        results_filename = f"trained_model_eval_results.json"
        results_path = f"{EVAL_RESULTS_PATH}/{results_filename}"
        
        os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        eval_results_volume.commit()
        
        print(f"💾 Results saved to {results_path}")
        print(f"🎯 Final accuracy: {results['accuracy']:.2%}")
        
        return results
    
    finally:
        # Clean up server more gracefully
        print("🧹 Cleaning up vLLM server...")
        try:
            # First try graceful termination
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
                print("✅ Server terminated gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't terminate gracefully, forcing shutdown...")
                server_process.kill()
                server_process.wait(timeout=5)
                print("✅ Server killed successfully")
        except Exception as e:
            print(f"⚠️  Error during server cleanup: {e}")
            try:
                server_process.kill()
                print("✅ Server force-killed as fallback")
            except:
                print("⚠️  Could not kill server process")


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
        EVAL_RESULTS_PATH: eval_results_volume,
    },
    timeout=30 * MINUTES,
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def eval_with_custom_dataset(
    model_name: str, 
    dataset_file: str,
    max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a model with a custom dataset file."""
    print(f"🔍 Evaluating {model_name} with custom dataset: {dataset_file}")
    
    # Load dataset
    dataset = load_evaluation_dataset(dataset_file)
    if not dataset:
        return {"error": f"Failed to load dataset from {dataset_file}"}
    
    # Limit examples if specified
    if max_examples and len(dataset) > max_examples:
        dataset = dataset[:max_examples]
        print(f"📊 Limited to first {max_examples} examples")
    
    print(f"📊 Dataset size: {len(dataset)} examples")
    
    # Start vLLM server
    if model_name.lower() in QWEN3_MODELS:
        full_model_name = QWEN3_MODELS[model_name.lower()]
    else:
        full_model_name = model_name
    
    print(f"🔥 Starting vLLM server for {full_model_name}")
    
    # Clear any corrupted compilation cache first
    cache_dir = "/root/.cache/vllm/torch_compile_cache"
    if os.path.exists(cache_dir):
        print("🧹 Clearing vLLM compilation cache to prevent corruption issues...")
        shutil.rmtree(cache_dir)
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    
    # Start the server
    server_process = subprocess.Popen(cmd)
    
    # Wait for server to start with health checks
    print("⏳ Waiting for vLLM server to be ready...")
    max_wait_time = 120  # 2 minutes max
    check_interval = 5   # Check every 5 seconds
    
    for attempt in range(max_wait_time // check_interval):
        await asyncio.sleep(check_interval)
        
        # Try to connect to the server
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8000/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        print(f"✅ vLLM server is ready after {(attempt + 1) * check_interval} seconds")
                        break
        except Exception as e:
            print(f"🔄 Attempt {attempt + 1}: Server not ready yet ({str(e)[:50]}...)")
            continue
    else:
        print("❌ Server failed to start within timeout period")
        # Try one more time with a simple test request
        try:
            test_response = await _get_vllm_response(
                "http://localhost:8000", "test", model_name
            )
            if not test_response.startswith("ERROR:"):
                print("✅ Server responding to test request")
            else:
                print(f"⚠️  Server test failed: {test_response[:100]}...")
        except Exception as e:
            print(f"⚠️  Server test exception: {e}")
    
    # Give a bit more time for full initialization
    await asyncio.sleep(10)
    
    try:
        # Create LLM wrapper
        class VLLMLLMWrapper:
            def __init__(self, server_url, model_name):
                self.server_url = server_url
                self.model_name = model_name
            
            async def generate_response(self, prompt: str) -> str:
                return await _get_vllm_response(
                    self.server_url, prompt, self.model_name
                )
        
        llm = VLLMLLMWrapper("http://localhost:8000", model_name)
        
        # Run evaluation
        results = await evaluate_model_on_dataset(llm, model_name, dataset)
        
        # Save results
        results_filename = f"custom_eval_{model_name}_{len(dataset)}_examples.json"
        results_path = f"{EVAL_RESULTS_PATH}/{results_filename}"
        
        os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        eval_results_volume.commit()
        
        print(f"💾 Results saved to {results_path}")
        return results
    
    finally:
        # Clean up server more gracefully
        print("🧹 Cleaning up vLLM server...")
        try:
            # First try graceful termination
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
                print("✅ Server terminated gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't terminate gracefully, forcing shutdown...")
                server_process.kill()
                server_process.wait(timeout=5)
                print("✅ Server killed successfully")
        except Exception as e:
            print(f"⚠️  Error during server cleanup: {e}")
            try:
                server_process.kill()
                print("✅ Server force-killed as fallback")
            except:
                print("⚠️  Could not kill server process") 