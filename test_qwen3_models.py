#!/usr/bin/env python3
"""
Test script for the updated model interface supporting both Gemini and Qwen3 models.

This script demonstrates how to use different Qwen3 model sizes and compare them
with Gemini models for filesystem-related tasks.
"""

import sys
import os

# Add src directory to path so we can import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import (
    get_model_response,
    get_available_qwen3_models,
    print_available_models,
    validate_api_key,
)


def test_qwen3_sizes():
    """Test different Qwen3 model sizes on filesystem prediction tasks."""

    # Test prompts related to filesystem operations
    test_prompts = [
        "What happens when I run 'ls -la' in a directory?",
        "Explain the difference between absolute and relative paths.",
        "What is the purpose of file permissions in Unix systems?",
    ]

    # Test different Qwen3 sizes
    qwen3_models_to_test = [
        "qwen3-0.6b",  # Smallest, fastest
        "qwen3-1.7b",  # Small
        "qwen3-4b",  # Medium
        # "qwen3-8b",    # Larger (uncomment if you have enough GPU memory)
        # "qwen3-14b",   # Even larger
        # "qwen3-32b",   # Largest dense model
    ]

    print("=" * 80)
    print("QWEN3 MODEL SIZE COMPARISON TEST")
    print("=" * 80)

    for prompt in test_prompts:
        print(f"\n📝 PROMPT: {prompt}")
        print("-" * 80)

        for model_name in qwen3_models_to_test:
            print(f"\n🤖 Testing {model_name.upper()}:")
            try:
                response = get_model_response(
                    prompt=prompt,
                    model_name=model_name,
                    temperature=0.1,  # Slightly creative but still focused
                    max_output_tokens=200,  # Keep responses concise for comparison
                )
                print(f"Response: {response}")
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")

        print("\n" + "=" * 80)


def compare_gemini_vs_qwen3():
    """Compare Gemini and Qwen3 responses side by side."""

    if not validate_api_key():
        print("⚠️  Gemini API key not configured. Skipping Gemini comparison.")
        return

    prompt = "Describe what a filesystem is and give examples of common filesystem operations."

    print("=" * 80)
    print("GEMINI vs QWEN3 COMPARISON")
    print("=" * 80)
    print(f"📝 PROMPT: {prompt}")
    print("-" * 80)

    # Test Gemini
    print("\n🔥 GEMINI RESPONSE:")
    try:
        gemini_response = get_model_response(prompt)
        print(gemini_response)
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

    # Test Qwen3
    print("\n🤖 QWEN3-8B RESPONSE:")
    try:
        qwen3_response = get_model_response(prompt, model_name="qwen3-8b")
        print(qwen3_response)
    except Exception as e:
        print(f"❌ Qwen3 Error: {e}")


def main():
    """Run all tests."""

    print("🚀 Starting Model Interface Tests")
    print("=" * 80)

    # Show available models
    print("\n📋 AVAILABLE MODELS:")
    print_available_models()

    # Test Qwen3 sizes
    print("\n" + "=" * 80)
    test_qwen3_sizes()

    # Compare Gemini vs Qwen3
    print("\n" + "=" * 80)
    compare_gemini_vs_qwen3()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
