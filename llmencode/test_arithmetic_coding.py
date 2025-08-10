"""
Comprehensive tests for arithmetic coding implementation.

Tests the ArithmeticCoder class for correctness, robustness, and edge cases.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from llmencode.arithmetic_coding import ArithmeticCoder, estimate_compression_ratio


class TestArithmeticCoding:
    """Test suite for arithmetic coding implementation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.coder = ArithmeticCoder()
    
    def test_basic_arithmetic_coding(self):
        """Test basic encode/decode functionality."""
        # Simple test case
        symbols = [0, 1, 0, 2]
        probabilities = [
            [0.7, 0.2, 0.1],  # For first symbol (chose 0)
            [0.3, 0.6, 0.1],  # For second symbol (chose 1) 
            [0.8, 0.1, 0.1],  # For third symbol (chose 0)
            [0.1, 0.1, 0.8],  # For fourth symbol (chose 2)
        ]
        
        # Encode
        encoded = self.coder.encode(symbols, probabilities)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        
        # Decode
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
    
    def test_single_symbol(self):
        """Test encoding/decoding a single symbol."""
        symbols = [1]
        probabilities = [[0.2, 0.8]]
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
    
    def test_uniform_distribution(self):
        """Test with uniform probability distributions."""
        symbols = [0, 1, 2, 3, 0, 1]
        probabilities = [[0.25, 0.25, 0.25, 0.25] for _ in range(len(symbols))]
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
    
    def test_skewed_distribution(self):
        """Test with highly skewed probability distributions."""
        symbols = [0, 0, 0, 1, 0]
        probabilities = [[0.99, 0.01] for _ in range(len(symbols))]
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
    
    def test_large_alphabet(self):
        """Test with large vocabulary size."""
        vocab_size = 100
        symbols = [i % vocab_size for i in range(20)]
        
        # Create random probability distributions
        np.random.seed(42)  # For reproducibility
        probabilities = []
        for _ in range(len(symbols)):
            probs = np.random.exponential(1.0, vocab_size)
            probs = probs / probs.sum()  # Normalize
            probabilities.append(probs.tolist())
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
    
    def test_long_sequence(self):
        """Test with long sequences to verify no precision loss."""
        np.random.seed(123)  # For reproducibility
        
        # Create a long sequence (500 symbols)
        sequence_length = 500
        vocab_size = 10
        symbols = np.random.randint(0, vocab_size, sequence_length).tolist()
        
        # Create varying probability distributions
        probabilities = []
        for i in range(sequence_length):
            probs = np.random.dirichlet(np.ones(vocab_size))  # Random distribution
            probabilities.append(probs.tolist())
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
        
        # Check compression effectiveness
        original_size = len(symbols) * 2  # 2 bytes per symbol
        compressed_size = len(encoded)
        compression_ratio = original_size / compressed_size
        assert compression_ratio > 1.0  # Should achieve some compression
    
    def test_iterative_decoding(self):
        """Test iterative decoding functionality."""
        symbols = [0, 1, 2, 1, 0]
        probabilities = [
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.3, 0.4, 0.3],
            [0.7, 0.2, 0.1],
        ]
        
        # Encode
        encoded = self.coder.encode(symbols, probabilities)
        
        # Decode iteratively
        self.coder.decode_iterative_init(encoded)
        decoded_symbols = []
        for prob_dist in probabilities:
            symbol = self.coder.decode_iterative_next(prob_dist)
            decoded_symbols.append(symbol)
        
        assert decoded_symbols == symbols
    
    def test_compression_ratio_estimation(self):
        """Test compression ratio estimation function."""
        symbols = [0, 0, 1, 0, 0]  # Mostly symbol 0
        probabilities = [[0.8, 0.2] for _ in range(len(symbols))]
        
        estimated_ratio = estimate_compression_ratio(symbols, probabilities)
        assert estimated_ratio > 1.0  # Should be compressible
        
        # Test with uniform distribution (less compressible)
        uniform_probs = [[0.5, 0.5] for _ in range(len(symbols))]
        uniform_ratio = estimate_compression_ratio(symbols, uniform_probs)
        
        # Skewed distribution should compress better than uniform
        assert estimated_ratio > uniform_ratio
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with very small probabilities
        symbols = [0, 1]
        probabilities = [
            [0.999999, 0.000001],
            [0.000001, 0.999999],
        ]
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
        
        # Test with minimal probability differences
        symbols = [0, 1]
        probabilities = [
            [0.500001, 0.499999],
            [0.499999, 0.500001],
        ]
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        assert decoded == symbols
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Mismatched lengths
        with pytest.raises(ValueError):
            self.coder.encode([0, 1], [[0.5, 0.5]])  # Too few prob distributions
        
        # Symbol out of range
        with pytest.raises(ValueError):
            self.coder.encode([5], [[0.5, 0.5]])  # Symbol 5 not in vocab of size 2
        
        # Too few probability distributions for decoding
        with pytest.raises(ValueError):
            self.coder.decode(b'\x00', [[0.5, 0.5]], 2)  # Need 2 distributions, only have 1
    
    def test_different_precision_bits(self):
        """Test with different precision settings."""
        symbols = [0, 1, 0]
        probabilities = [[0.7, 0.3] for _ in range(len(symbols))]
        
        # Test with different precision levels
        for precision in [32, 48, 62]:
            coder = ArithmeticCoder(precision_bits=precision)
            encoded = coder.encode(symbols, probabilities)
            decoded = coder.decode(encoded, probabilities, len(symbols))
            assert decoded == symbols
    
    def test_reproducibility(self):
        """Test that encoding is deterministic."""
        symbols = [0, 1, 2, 1, 0]
        probabilities = [[0.4, 0.3, 0.3] for _ in range(len(symbols))]
        
        # Encode multiple times
        encoded1 = self.coder.encode(symbols, probabilities)
        encoded2 = self.coder.encode(symbols, probabilities)
        
        assert encoded1 == encoded2
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        symbols = []
        probabilities = []
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, 0)
        assert decoded == symbols
    
    def test_compression_effectiveness(self):
        """Test that compression is actually effective."""
        # Create a highly redundant sequence
        symbols = [0] * 50 + [1] * 10  # Mostly 0s
        probabilities = [[0.9, 0.1] for _ in range(len(symbols))]
        
        encoded = self.coder.encode(symbols, probabilities)
        decoded = self.coder.decode(encoded, probabilities, len(symbols))
        
        assert decoded == symbols
        
        # Check compression ratio
        original_size = len(symbols) * 2  # 2 bytes per symbol
        compressed_size = len(encoded)
        compression_ratio = original_size / compressed_size
        
        # Should achieve significant compression for this redundant sequence
        assert compression_ratio > 5.0


def test_arithmetic_coding_integration():
    """Integration test for the complete arithmetic coding workflow."""
    print("Testing Arithmetic Coding Implementation")
    print("=" * 50)
    
    coder = ArithmeticCoder()
    
    # Test symbols and their probability distributions
    symbols = [0, 1, 0, 2, 1, 0, 0, 1]
    probabilities = [
        [0.7, 0.2, 0.1],  # High probability for 0
        [0.3, 0.6, 0.1],  # High probability for 1
        [0.8, 0.1, 0.1],  # High probability for 0
        [0.1, 0.1, 0.8],  # High probability for 2
        [0.2, 0.7, 0.1],  # High probability for 1
        [0.9, 0.05, 0.05], # Very high probability for 0
        [0.8, 0.15, 0.05], # High probability for 0
        [0.1, 0.8, 0.1],  # High probability for 1
    ]
    
    print(f"Original symbols: {symbols}")
    print(f"Vocabulary size: {len(probabilities[0])}")
    
    # Encode
    encoded = coder.encode(symbols, probabilities)
    print(f"Encoded to {len(encoded)} bytes: {encoded.hex()}")
    
    # Decode
    decoded = coder.decode(encoded, probabilities, len(symbols))
    print(f"Decoded symbols: {decoded}")
    
    # Check correctness
    success = symbols == decoded
    print(f"Perfect roundtrip: {success}")
    
    if success:
        # Calculate compression metrics
        original_size = len(symbols) * 2  # 2 bytes per symbol
        compressed_size = len(encoded)
        actual_ratio = original_size / compressed_size
        
        estimated_ratio = estimate_compression_ratio(symbols, probabilities)
        
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes") 
        print(f"Actual compression ratio: {actual_ratio:.2f}x")
        print(f"Estimated compression ratio: {estimated_ratio:.2f}x")
        print("✓ All tests passed!")
    else:
        print("✗ Roundtrip test failed!")
    
    # For pytest, assert instead of return
    assert success, "Integration test failed - symbols don't match after roundtrip"


if __name__ == "__main__":
    # Run integration test
    success = test_arithmetic_coding_integration()
    
    if success:
        print("\n🎉 Integration test passed!")
        
        # Run all pytest tests
        print("\n🔄 Running comprehensive test suite...")
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            __file__, '-v'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n🎉 All comprehensive tests passed!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
    else:
        print("\n❌ Integration tests failed.") 