#!/usr/bin/env python3
"""
Comprehensive test suite for LLM-based arithmetic coding compression.

This script tests the full roundtrip compression/decompression pipeline using
Qwen 4B model and displays detailed compression statistics.
"""

import sys
import time
import traceback
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, 'src')

from llmencode import LLMEncode
from arithmetic_coding import test_arithmetic_coding


def test_basic_arithmetic_coding():
    """Test the basic arithmetic coding implementation."""
    print("🧮 Testing Basic Arithmetic Coding")
    print("=" * 50)
    
    success = test_arithmetic_coding()
    
    if success:
        print("✅ Basic arithmetic coding test passed!")
    else:
        print("❌ Basic arithmetic coding test failed!")
    
    return success


def test_llm_compression_simple():
    """Test LLM compression with simple text."""
    print("\n🤖 Testing LLM Compression - Simple Text")
    print("=" * 50)
    
    simple_text = "Hello world! This is a test of LLM-based compression using arithmetic coding."
    
    try:
        encoder = LLMEncode(model_name="qwen3-4b", max_tokens=50)
        result = encoder.test_roundtrip(simple_text, verbose=True)
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Simple text test failed: {e}")
        traceback.print_exc()
        return False


def test_llm_compression_complex():
    """Test LLM compression with more complex text."""
    print("\n📚 Testing LLM Compression - Complex Text")
    print("=" * 50)
    
    complex_text = """
    Arithmetic coding is a form of entropy encoding used in lossless data compression.
    Unlike Huffman coding, which assigns fixed-length codes to symbols, arithmetic coding
    represents data as a single fraction between 0 and 1. The technique was introduced
    by Jorma Rissanen in 1976 and was later improved by various researchers.
    
    In the context of language model compression, arithmetic coding can leverage the
    probability distributions predicted by neural networks to achieve compression ratios
    that approach the theoretical entropy limit. This is particularly effective when
    the language model has been trained on similar text domains.
    
    The key insight is that language models implicitly learn to compress text by
    predicting the next token based on context. By using these predictions as
    probability distributions for arithmetic coding, we can achieve compression
    that adapts to the specific patterns in the input text.
    """
    
    try:
        encoder = LLMEncode(model_name="qwen3-4b", max_tokens=200)
        result = encoder.test_roundtrip(complex_text, verbose=True)
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Complex text test failed: {e}")
        traceback.print_exc()
        return False


def test_llm_compression_repetitive():
    """Test LLM compression with repetitive text (should compress well)."""
    print("\n🔄 Testing LLM Compression - Repetitive Text")
    print("=" * 50)
    
    repetitive_text = """
    The quick brown fox jumps over the lazy dog. 
    The quick brown fox jumps over the lazy dog.
    The quick brown fox jumps over the lazy dog.
    The quick brown fox jumps over the lazy dog.
    This repetitive text should compress very well because the language model
    can predict the repeated patterns with high confidence.
    """
    
    try:
        encoder = LLMEncode(model_name="qwen3-4b", max_tokens=100)
        result = encoder.test_roundtrip(repetitive_text, verbose=True)
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Repetitive text test failed: {e}")
        traceback.print_exc()
        return False


def test_file_operations():
    """Test file save/load operations."""
    print("\n💾 Testing File Operations")
    print("=" * 50)
    
    test_text = "This text will be compressed and saved to a file, then loaded and decompressed."
    
    try:
        encoder = LLMEncode(model_name="qwen3-4b", max_tokens=50)
        
        # Save to file
        print("Saving compressed data to file...")
        output_file = "test_compressed.pkl"
        stats = encoder.encode_to_file(test_text, output_file)
        
        print(f"✓ Saved to {output_file}")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        
        # Load from file
        print("Loading and decompressing from file...")
        decoded_text = encoder.decode_from_file(output_file)
        
        print(f"✓ Loaded and decompressed")
        print(f"  Original:  {test_text}")
        print(f"  Decoded:   {decoded_text}")
        
        # Clean up
        import os
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"✓ Cleaned up {output_file}")
        
        # Check if they match
        success = test_text.strip() in decoded_text.strip()
        
        if success:
            print("✅ File operations test passed!")
        else:
            print("❌ File operations test failed - text doesn't match!")
        
        return success
        
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        traceback.print_exc()
        return False


def benchmark_compression_ratio():
    """Benchmark compression ratios on different types of text."""
    print("\n📊 Compression Ratio Benchmark")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Technical Text',
            'text': """
            Machine learning algorithms require large datasets to train effectively.
            Deep neural networks use backpropagation to optimize their parameters.
            Transformers use attention mechanisms to process sequential data.
            """,
            'expected_ratio': 2.0,
        },
        {
            'name': 'Natural Language',
            'text': """
            The sun was shining brightly on the beautiful summer day. Birds were
            singing in the trees and children were playing in the park. It was
            a perfect day for a picnic with family and friends.
            """,
            'expected_ratio': 1.5,
        },
        {
            'name': 'Highly Predictable',
            'text': """
            To be or not to be, that is the question. Whether 'tis nobler in the
            mind to suffer the slings and arrows of outrageous fortune, or to take
            arms against a sea of troubles, and by opposing end them.
            """,
            'expected_ratio': 3.0,
        },
    ]
    
    results = []
    
    try:
        encoder = LLMEncode(model_name="qwen3-4b", max_tokens=100)
        
        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            print("-" * 30)
            
            start_time = time.time()
            result = encoder.test_roundtrip(test_case['text'], verbose=False)
            end_time = time.time()
            
            if result['success']:
                stats = result['stats']
                compression_time = end_time - start_time
                
                print(f"✓ Success")
                print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
                print(f"  Theoretical ratio: {stats['theoretical_compression_ratio']:.2f}x")
                print(f"  Efficiency: {stats['compression_efficiency']:.2f}")
                print(f"  Processing time: {compression_time:.2f}s")
                print(f"  Tokens: {stats['num_tokens']}")
                
                results.append({
                    'name': test_case['name'],
                    'success': True,
                    'compression_ratio': stats['compression_ratio'],
                    'theoretical_ratio': stats['theoretical_compression_ratio'],
                    'efficiency': stats['compression_efficiency'],
                    'time': compression_time,
                    'tokens': stats['num_tokens'],
                })
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                })
        
        # Summary
        print("\n📋 Benchmark Summary")
        print("=" * 50)
        
        successful_tests = [r for r in results if r['success']]
        
        if successful_tests:
            avg_ratio = sum(r['compression_ratio'] for r in successful_tests) / len(successful_tests)
            avg_efficiency = sum(r['efficiency'] for r in successful_tests) / len(successful_tests)
            total_time = sum(r['time'] for r in successful_tests)
            
            print(f"Successful tests: {len(successful_tests)}/{len(results)}")
            print(f"Average compression ratio: {avg_ratio:.2f}x")
            print(f"Average efficiency: {avg_efficiency:.2f}")
            print(f"Total processing time: {total_time:.2f}s")
            
            return len(successful_tests) == len(results)
        else:
            print("No successful tests!")
            return False
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 LLM Arithmetic Coding Compression Test Suite")
    print("=" * 70)
    print("Testing Qwen 4B model with arithmetic coding compression")
    print("Inspired by Fabrice Bellard's work and 'Language Modeling is Compression'")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Arithmetic Coding", test_basic_arithmetic_coding),
        ("Simple Text Compression", test_llm_compression_simple),
        ("Complex Text Compression", test_llm_compression_complex),
        ("Repetitive Text Compression", test_llm_compression_repetitive),
        ("File Operations", test_file_operations),
        ("Compression Benchmark", benchmark_compression_ratio),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏁 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LLM compression is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 