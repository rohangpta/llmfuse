"""
LLM-based text compression using arithmetic coding.

This module provides the main LLMEncode class that uses language model probabilities
to achieve text compression through arithmetic coding.
"""

import os
from typing import Dict, Any, Optional, Tuple
import pickle
import base64

try:
    from .model import get_model_logprobs, compress_with_model_probs, decompress_with_model_probs
    from .arithmetic_coding import ArithmeticCoder, estimate_compression_ratio
except ImportError:
    # For direct execution
    from model import get_model_logprobs, compress_with_model_probs, decompress_with_model_probs
    from arithmetic_coding import ArithmeticCoder, estimate_compression_ratio


class LLMEncode:
    """
    Main class for LLM-based text compression using arithmetic coding.
    
    This class leverages language model probability distributions to compress text
    by using the model's predictions to guide arithmetic coding decisions.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-4b",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        precision_bits: int = 128,
    ):
        """
        Initialize the LLM encoder.
        
        Args:
            model_name: Name of the language model to use
            max_tokens: Maximum number of tokens to process (limit to 1k as requested)
            temperature: Must be 0.0 for deterministic results
            precision_bits: Precision for arithmetic coding
        """
        if temperature != 0.0:
            raise ValueError("Temperature must be 0.0 for deterministic compression/decompression")
        
        if max_tokens > 1000:
            raise ValueError("Max tokens limited to 1000 as per requirements")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.coder = ArithmeticCoder(precision_bits=precision_bits)
        
        # Validate model availability
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model can be loaded."""
        try:
            from transformers import AutoTokenizer
            # Try to load tokenizer to validate model
            tokenizer = AutoTokenizer.from_pretrained(self._get_full_model_name())
        except Exception as e:
            raise ValueError(f"Cannot load model {self.model_name}: {e}")
    
    def _get_full_model_name(self) -> str:
        """Get the full HuggingFace model name."""
        try:
            from .model import QWEN3_MODELS
        except ImportError:
            from model import QWEN3_MODELS
        if self.model_name.lower() in QWEN3_MODELS:
            return QWEN3_MODELS[self.model_name.lower()]
        return self.model_name
    
    def encode(self, text: str) -> bytes:
        """
        Encode text using LLM-based compression.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed bytes
        """
        if len(text) == 0:
            return b''
        
        # Use model-based compression
        compressed_bytes, metadata = compress_with_model_probs(
            text, 
            model_name=self.model_name, 
            max_tokens=self.max_tokens,
            verbose=False
        )
        
        # Store metadata for decoding
        self._last_metadata = metadata
        
        return compressed_bytes
    
    def decode(self, compressed_bytes: bytes) -> str:
        """
        Decode compressed bytes back to text.
        
        Args:
            compressed_bytes: Compressed data
            
        Returns:
            Decompressed text
        """
        if len(compressed_bytes) == 0:
            return ''
        
        if not hasattr(self, '_last_metadata'):
            raise ValueError("No metadata available for decoding. Must encode first.")
        
        # Use model-based decompression
        return decompress_with_model_probs(
            compressed_bytes, 
            self._last_metadata,
            verbose=False
        )
    
    def encode_to_file(self, text: str, output_path: str) -> Dict[str, Any]:
        """
        Encode text and save to file.
        
        Args:
            text: Input text to compress
            output_path: Path to save compressed data
        
        Returns:
            Encoding statistics
        """
        result = self.encode(text)
        
        # Save to file using pickle for reliable serialization
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"Compressed data saved to: {output_path}")
        
        return result['stats']
    
    def decode_from_file(self, input_path: str) -> str:
        """
        Load and decode compressed data from file.
        
        Args:
            input_path: Path to compressed data file
        
        Returns:
            Decompressed text
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Compressed file not found: {input_path}")
        
        # Load from file
        with open(input_path, 'rb') as f:
            encoded_result = pickle.load(f)
        
        return self.decode(encoded_result)
    
    def test_roundtrip(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Test encoding and decoding roundtrip on given text.
        
        Args:
            text: Text to test
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with test results
        """
        if verbose:
            print("=" * 60)
            print("LLM Compression Roundtrip Test")
            print("=" * 60)
            print(f"Model: {self.model_name}")
            print(f"Max tokens: {self.max_tokens}")
            print(f"Input text length: {len(text)} characters")
            print(f"Input text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
            print("-" * 60)
        
        try:
            # Encode
            if verbose:
                print("Encoding...")
            
            compressed_bytes, metadata = compress_with_model_probs(
                text, 
                model_name=self.model_name, 
                max_tokens=self.max_tokens,
                verbose=verbose
            )
            
            if verbose:
                print("✓ Encoded successfully")
                print(f"  Original size: {len(text)} bytes")
                print(f"  Compressed size: {len(compressed_bytes)} bytes")
                
                # Calculate compression ratio
                if len(compressed_bytes) > 0:
                    compression_ratio = len(text) / len(compressed_bytes)
                    print(f"  Compression ratio: {compression_ratio:.2f}x")
                    
                    # Estimate theoretical ratio
                    from .arithmetic_coding import estimate_compression_ratio
                    # We'd need to recompute probabilities for this, so skip for now
                    
                print(f"  Tokens processed: {metadata['num_tokens']}")
            
            # Decode
            if verbose:
                print("Decoding...")
            
            decoded_text = decompress_with_model_probs(
                compressed_bytes, 
                metadata,
                verbose=verbose
            )
            
            # Check if roundtrip was successful
            success = (text == decoded_text)
            
            if verbose:
                if success:
                    print("✓ Perfect roundtrip!")
                else:
                    print("❌ Roundtrip failed!")
                    print(f"Original:  {repr(text)}")
                    print(f"Decoded:   {repr(decoded_text)}")
                    
                    # Show character differences
                    if len(text) != len(decoded_text):
                        print(f"Length mismatch: {len(text)} vs {len(decoded_text)}")
                    else:
                        for i, (c1, c2) in enumerate(zip(text, decoded_text)):
                            if c1 != c2:
                                print(f"First difference at position {i}: {repr(c1)} vs {repr(c2)}")
                                break
            
            # Calculate statistics
            original_size = len(text)
            compressed_size = len(compressed_bytes)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            return {
                'success': success,
                'original_text': text,
                'decoded_text': decoded_text,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'num_tokens': metadata['num_tokens'],
                'model_name': self.model_name,
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Roundtrip test failed: {type(e).__name__}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'original_text': text,
                'decoded_text': None,
                'original_size': len(text),
                'compressed_size': 0,
                'compression_ratio': 0,
                'num_tokens': 0,
                'model_name': self.model_name,
            }


def main():
    """
    Demo of the LLMEncode functionality.
    """
    print("=" * 70)
    print("LLMEncode Demo - LLM-based Text Compression")
    print("=" * 70)
    
    # Test text
    test_text = """
    The quick brown fox jumps over the lazy dog. This is a classic pangram sentence 
    that contains every letter of the English alphabet at least once. Language models 
    can predict this text quite well due to its familiarity, which should lead to 
    good compression ratios when using model probabilities for arithmetic coding.
    """
    
    try:
        # Initialize encoder with Qwen 4B
        encoder = LLMEncode(model_name="qwen3-4b", max_tokens=200)
        
        # Run roundtrip test
        result = encoder.test_roundtrip(test_text, verbose=True)
        
        if result['success']:
            print("\n🎉 Demo completed successfully!")
        else:
            print(f"\n💥 Demo failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    main() 