"""
Modal app for testing Qwen3 models using vLLM server on cloud infrastructure.

This app runs Qwen3 models via vLLM server with OpenAI-compatible API,
allowing efficient inference and comparison of different model sizes.

This is the main entrypoint that imports functionality from other modules.

Usage:
    modal run eval/modal_eval.py::test_qwen3
    modal run eval/modal_eval.py::test_qwen3_single --model-name="qwen3-8b"
    modal run eval/modal_eval.py::compare_gemini_qwen3
    modal run eval/modal_eval.py::serve_qwen3 --model-name="qwen3-8b"
"""

# Import all the functionality from the other modules
from eval.common import (
    app, print_available_models, get_available_qwen3_models, download_model,
    _get_vllm_response, _get_gemini_response, QWEN3_MODELS, VLLM_PORT, MINUTES,
    vllm_image, model_cache, vllm_cache, MODEL_CACHE_PATH, VLLM_CACHE_PATH,
    eval_results_volume
)
from eval.qwen_models import (
    start_vllm_server, serve_qwen3, serve_qwen3_0_6b, serve_qwen3_1_7b, 
    serve_qwen3_4b, serve_qwen3_8b, serve_qwen3_14b, serve_qwen3_32b
)
from eval.evaluation import (
    eval_single_size, eval_sizes, analyze_eval_results, eval_trained_model,
    eval_with_custom_dataset, evaluate_model_on_dataset
)

import asyncio
import modal
from typing import Dict, Any, List, Optional
import json


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=1800,  # 30 minutes
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def test_qwen3_with_vllm():
    """Test all available Qwen3 models with vLLM servers."""
    print("🚀 Testing all Qwen3 models with vLLM")
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about mountains.",
        "How do you make a cup of tea?",
        "What are the benefits of renewable energy?",
    ]
    
    results = {}
    
    # Test smaller models (practical for testing)
    test_models = ["qwen3-4b", "qwen3-8b"] 
    
    for model_name in test_models:
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            # Start vLLM server for this model
            server_result = await start_vllm_server.remote.aio(model_name)
            print(f"Server started: {server_result}")
            
            # Test with sample prompts
            model_results = []
            for prompt in test_prompts:
                response = await _get_vllm_response(
                    "http://localhost:8000", prompt, model_name
                )
                model_results.append({
                    "prompt": prompt,
                    "response": response[:200] + "..." if len(response) > 200 else response
                })
            
            results[model_name] = {
                "status": "success",
                "results": model_results
            }
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Print summary
    print("\n📊 TESTING SUMMARY:")
    for model, result in results.items():
        status = result["status"]
        print(f"  {model:12}: {status.upper()}")
    
    return results


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=1800,  # 30 minutes
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def test_qwen3_single_vllm(model_name: str = "qwen3-8b"):
    """Test a single Qwen3 model with vLLM server."""
    print(f"🚀 Testing single model: {model_name}")
    
    # Validate model name
    if model_name.lower() not in QWEN3_MODELS:
        available = list(QWEN3_MODELS.keys())
        return {
            "error": f"Model {model_name} not found. Available: {available}"
        }
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning in one sentence.",
        "List 3 programming languages.",
        "What is 2 + 2?",
        "Describe the color blue.",
    ]
    
    try:
        # Start vLLM server
        print(f"🔥 Starting vLLM server for {model_name}")
        server_result = await start_vllm_server.remote.aio(model_name)
        print(f"Server status: {server_result}")
        
        # Test with prompts
        results = []
        for i, prompt in enumerate(test_prompts, 1):
            print(f"📝 Testing prompt {i}/{len(test_prompts)}: {prompt[:50]}...")
            
            response = await _get_vllm_response(
                "http://localhost:8000", prompt, model_name
            )
            
            results.append({
                "prompt": prompt,
                "response": response,
                "response_length": len(response),
                "contains_error": "ERROR:" in response
            })
        
        # Calculate success rate
        successful_responses = sum(1 for r in results if not r["contains_error"])
        success_rate = successful_responses / len(results)
        
        return {
            "model_name": model_name,
            "success_rate": success_rate,
            "total_prompts": len(test_prompts),
            "successful_responses": successful_responses,
            "results": results
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return {
            "model_name": model_name,
            "error": str(e)
        }


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_PATH: model_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
    timeout=2400,  # 40 minutes
    secrets=[
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
async def compare_gemini_qwen3_vllm():
    """Compare Gemini with Qwen3 models on the same prompts."""
    print("🚀 Comparing Gemini with Qwen3 models")
    
    # Test prompts for comparison
    comparison_prompts = [
        "Explain the concept of artificial intelligence.",
        "What are the main causes of climate change?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is the difference between machine learning and deep learning?",
    ]
    
    results = {}
    
    # Test Gemini first
    print("\n🔍 Testing Gemini...")
    try:
        gemini_results = []
        for prompt in comparison_prompts:
            response = _get_gemini_response(prompt)
            gemini_results.append({
                "prompt": prompt,
                "response": response[:300] + "..." if len(response) > 300 else response
            })
        
        results["gemini-2.5-flash"] = {
            "status": "success",
            "results": gemini_results
        }
    except Exception as e:
        print(f"❌ Error testing Gemini: {str(e)}")
        results["gemini-2.5-flash"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test Qwen3 models
    test_models = ["qwen3-4b", "qwen3-8b"]
    
    for model_name in test_models:
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            # Start vLLM server
            await start_vllm_server.remote.aio(model_name)
            
            # Test with prompts
            model_results = []
            for prompt in comparison_prompts:
                response = await _get_vllm_response(
                    "http://localhost:8000", prompt, model_name
                )
                model_results.append({
                    "prompt": prompt,
                    "response": response[:300] + "..." if len(response) > 300 else response
                })
            
            results[model_name] = {
                "status": "success",
                "results": model_results
            }
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Print comparison summary
    print("\n📊 COMPARISON SUMMARY:")
    for model, result in results.items():
        status = result["status"]
        print(f"  {model:20}: {status.upper()}")
    
    return results


@app.local_entrypoint()
def main():
    """Local entrypoint to display help and available commands."""
    print("🤖 Qwen3 vLLM Testing Suite")
    print("=" * 50)
    
    print("\n🚀 Available Commands:")
    print("  modal run eval/modal_eval.py::test_qwen3_with_vllm            # Test all models")
    print("  modal run eval/modal_eval.py::test_qwen3_single_vllm --model-name='qwen3-8b'  # Test single model")
    print("  modal run eval/modal_eval.py::compare_gemini_qwen3_vllm       # Compare with Gemini")
    
    print("\n🌐 Model Serving Commands:")
    print("  modal run eval/modal_eval.py::serve_qwen3             # Default (qwen3-8b)")
    print("  modal run eval/modal_eval.py::serve_qwen3_0_6b        # Qwen3 0.6B")
    print("  modal run eval/modal_eval.py::serve_qwen3_1_7b        # Qwen3 1.7B")
    print("  modal run eval/modal_eval.py::serve_qwen3_4b          # Qwen3 4B")
    print("  modal run eval/modal_eval.py::serve_qwen3_8b          # Qwen3 8B")
    print("  modal run eval/modal_eval.py::serve_qwen3_14b         # Qwen3 14B")
    print("  modal run eval/modal_eval.py::serve_qwen3_32b         # Qwen3 32B")
    
    print("\n🔍 Evaluation Commands:")
    print("  modal run eval/modal_eval.py::eval_single_size --model-name='qwen3-8b'      # Evaluate single model")
    print("  modal run eval/modal_eval.py::eval_sizes                                     # Evaluate all practical sizes")
    print("  modal run eval/modal_eval.py::analyze_eval_results --results-filename='...' # Analyze results")
    
    print("\n📊 Utility Commands:")
    print("  modal run eval/modal_eval.py::download_model --model-name='qwen3-8b'        # Download model")
    print("  modal run eval/modal_eval.py::print_available_models                        # List models")
    
    print_available_models()


# Re-export key functions for backwards compatibility
test_qwen3 = test_qwen3_with_vllm
test_qwen3_single = test_qwen3_single_vllm
compare_gemini_qwen3 = compare_gemini_qwen3_vllm


@app.function(
    volumes={"/root/eval_results": eval_results_volume},
    timeout=5 * MINUTES,
)
def download_eval_results(filename: str = "eval_results_qwen3-4b_detailed.json"):
    """Download detailed evaluation results JSON file."""
    import os
    import json
    
    results_path = f"/root/eval_results/{filename}"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        available_files = []
        if os.path.exists("/root/eval_results"):
            available_files = os.listdir("/root/eval_results")
        return {
            "error": f"File {filename} not found",
            "available_files": available_files
        }
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    print(f"📊 Downloaded results for {results.get('model_name', 'unknown')}")
    print(f"📈 Accuracy: {results.get('accuracy', 0):.2%}")
    print(f"📝 Total examples: {results.get('total_examples', 0)}")
    
    # Print the first few detailed results for inspection
    detailed_results = results.get('detailed_results', [])
    print(f"\n🔍 First 3 detailed results:")
    for i, result in enumerate(detailed_results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Expected (first 200 chars): {result.get('expected', '')[:200]}...")
        print(f"Predicted (first 200 chars): {result.get('predicted', '')[:200]}...")
        print(f"Similarity: {result.get('similarity', 0):.3f}")
        print(f"Correct: {result.get('correct', False)}")
    
    return results
