#!/usr/bin/env python3
"""
LLM-based text compression using arithmetic coding.

This module provides the main LLMEncode class that uses language model probabilities
to achieve text compression through arithmetic coding.

Usage:
    python llmencode.py "Hello world"
    python llmencode.py --help
"""



import os
import argparse
import sys
from typing import Dict, Any, Optional, Tuple
import pickle
import base64

try:
    # When imported as a package
    from common.model import get_model_logprobs, compress_with_model_probs, decompress_with_model_probs
    from .arithmetic_coding import ArithmeticCoder, estimate_compression_ratio
except ImportError:
    # When run directly, add parent directory to path
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from common.model import get_model_logprobs, compress_with_model_probs, decompress_with_model_probs
    from llmencode.arithmetic_coding import ArithmeticCoder, estimate_compression_ratio


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
        from common.model import get_qwen3_4b_id
        canonical_id = get_qwen3_4b_id()
        if self.model_name.lower() in ("qwen3-4b", canonical_id.lower()):
            return canonical_id
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
        compressed_bytes = self.encode(text)

        # Save compressed bytes and metadata to file using pickle
        with open(output_path, 'wb') as f:
            pickle.dump({
                'compressed': compressed_bytes,
                'metadata': self._last_metadata
            }, f)

        return {
            'original_size': len(text),
            'compressed_size': len(compressed_bytes),
            'compression_ratio': len(text) / len(compressed_bytes) if compressed_bytes else 0
        }
    
    def decode_from_file(self, input_path: str) -> str:
        """
        Load and decode compressed data from file.
        
        Args:
            input_path: Path to compressed data file
        
        Returns:
            Decompressed text
        """
        from pathlib import Path
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Compressed file not found: {input_path}")

        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        self._last_metadata = data['metadata']
        return self.decode(data['compressed'])
    
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
                    
                    # Calculate compression effectiveness
                    
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


def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface for LLMEncode."""
    parser = argparse.ArgumentParser(
        description="LLM-based text compression using arithmetic coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s encode "Hello world"
  %(prog)s decode 48656c6c6f20776f726c64
  %(prog)s test "Hello world"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode text')
    encode_parser.add_argument('text', help='Text to encode')
    encode_parser.add_argument('--model', default='qwen3-4b', help='Model to use')
    encode_parser.add_argument('--max-tokens', type=int, default=1000, help='Max tokens')
    encode_parser.add_argument('--output-format', choices=['hex', 'base64'], default='hex')
    encode_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode compressed data')
    decode_parser.add_argument('data', help='Compressed data to decode')
    decode_parser.add_argument('--model', default='qwen3-4b', help='Model to use')
    decode_parser.add_argument('--max-tokens', type=int, default=1000, help='Max tokens')
    decode_parser.add_argument('--input-format', choices=['hex', 'base64'], default='hex')
    decode_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test roundtrip compression')
    test_parser.add_argument('text', help='Text to test')
    test_parser.add_argument('--model', default='qwen3-4b', help='Model to use')
    test_parser.add_argument('--max-tokens', type=int, default=1000, help='Max tokens')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        encoder = LLMEncode(model_name=args.model, max_tokens=args.max_tokens)
        
        if args.command == 'encode':
            if args.verbose:
                print(f"Encoding with model: {args.model}")
                print(f"Input text: {repr(args.text)}")
            
            compressed_bytes = encoder.encode(args.text)
            
            if args.output_format == 'hex':
                output = compressed_bytes.hex()
            else:  # base64
                output = base64.b64encode(compressed_bytes).decode('ascii')
            
            if args.verbose:
                metadata = encoder._last_metadata
                print(f"Original size: {len(args.text)} bytes")
                print(f"Compressed size: {len(compressed_bytes)} bytes")
                print(f"Compression ratio: {len(args.text) / len(compressed_bytes):.2f}x")
                print(f"Tokens: {metadata['num_tokens']}")
                print(f"Compressed data ({args.output_format}):")
            
            print(output)
            
        elif args.command == 'decode':
            if args.input_format == 'hex':
                compressed_bytes = bytes.fromhex(args.data.replace(' ', ''))
            else:  # base64
                compressed_bytes = base64.b64decode(args.data)
            
            # Note: This is a limitation - we need metadata for decoding
            # In practice, metadata would be stored alongside the compressed data
            print("❌ Decoding requires metadata from encoding session", file=sys.stderr)
            print("Use the 'test' command for roundtrip testing", file=sys.stderr)
            sys.exit(1)
            
        elif args.command == 'test':
            if args.verbose:
                print(f"Testing roundtrip with model: {args.model}")
            
            result = encoder.test_roundtrip(args.text, verbose=args.verbose)
            
            if result['success']:
                print(f"✅ Perfect roundtrip!")
                print(f"📊 {result['original_size']} bytes → {result['compressed_size']} bytes")
                print(f"🚀 Compression ratio: {result['compression_ratio']:.2f}x")
            else:
                print(f"❌ Roundtrip failed!")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 
