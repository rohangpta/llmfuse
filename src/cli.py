#!/usr/bin/env python3
"""
Command-line interface for LLM-based compression.

Usage:
    llmencode encode "Hello world"
    llmencode decode "828d4693a0"
    llmencode test "Hello world"
"""

import argparse
import sys
import base64
import json
from typing import Dict, Any

from .llmencode import LLMEncode


def encode_command(args) -> None:
    """Handle the encode subcommand."""
    try:
        encoder = LLMEncode(model_name=args.model, max_tokens=args.max_tokens)
        
        if args.verbose:
            print(f"Encoding with model: {args.model}")
            print(f"Input text: {repr(args.text)}")
            print(f"Input length: {len(args.text)} characters")
        
        # Encode the text
        compressed_bytes = encoder.encode(args.text)
        
        # Get metadata for display
        metadata = encoder._last_metadata
        
        if args.output_format == 'hex':
            output = compressed_bytes.hex()
        elif args.output_format == 'base64':
            output = base64.b64encode(compressed_bytes).decode('ascii')
        else:  # binary
            output = ' '.join(format(b, '08b') for b in compressed_bytes)
        
        # Display results
        if args.verbose:
            print(f"\n✅ Encoding successful!")
            print(f"Original size: {len(args.text)} bytes")
            print(f"Compressed size: {len(compressed_bytes)} bytes")
            print(f"Compression ratio: {len(args.text) / len(compressed_bytes):.2f}x")
            print(f"Tokens processed: {metadata['num_tokens']}")
            print(f"Model: {metadata['model_name']}")
            print(f"\nCompressed data ({args.output_format}):")
        
        print(output)
        
        # Save metadata if requested
        if args.save_metadata:
            metadata_file = args.save_metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            if args.verbose:
                print(f"Metadata saved to: {metadata_file}")
                
    except Exception as e:
        print(f"❌ Encoding failed: {e}", file=sys.stderr)
        sys.exit(1)


def decode_command(args) -> None:
    """Handle the decode subcommand."""
    try:
        encoder = LLMEncode(model_name=args.model, max_tokens=args.max_tokens)
        
        # Parse input based on format
        if args.input_format == 'hex':
            compressed_bytes = bytes.fromhex(args.data.replace(' ', ''))
        elif args.input_format == 'base64':
            compressed_bytes = base64.b64decode(args.data)
        else:  # binary
            binary_str = args.data.replace(' ', '')
            if len(binary_str) % 8 != 0:
                raise ValueError("Binary string length must be multiple of 8")
            compressed_bytes = bytes(int(binary_str[i:i+8], 2) for i in range(0, len(binary_str), 8))
        
        # Load metadata if provided
        if args.metadata:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
            encoder._last_metadata = metadata
        else:
            # For CLI usage, we need metadata. This is a limitation.
            print("❌ Metadata required for decoding. Use --metadata flag or encode/decode in same session.", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            print(f"Decoding with model: {args.model}")
            print(f"Compressed size: {len(compressed_bytes)} bytes")
            print(f"Expected tokens: {metadata['num_tokens']}")
        
        # Decode the data
        decoded_text = encoder.decode(compressed_bytes)
        
        if args.verbose:
            print(f"\n✅ Decoding successful!")
            print(f"Decoded length: {len(decoded_text)} characters")
            print(f"\nDecoded text:")
        
        print(decoded_text)
        
    except Exception as e:
        print(f"❌ Decoding failed: {e}", file=sys.stderr)
        sys.exit(1)


def test_command(args) -> None:
    """Handle the test subcommand (encode + decode roundtrip)."""
    try:
        encoder = LLMEncode(model_name=args.model, max_tokens=args.max_tokens)
        
        print(f"🧪 Testing roundtrip compression")
        print(f"Model: {args.model}")
        print(f"Input: {repr(args.text)}")
        print("=" * 50)
        
        # Run roundtrip test
        result = encoder.test_roundtrip(args.text, verbose=args.verbose)
        
        if result['success']:
            print(f"✅ Perfect roundtrip!")
            print(f"📊 {result['original_size']} bytes → {result['compressed_size']} bytes")
            print(f"🚀 Compression ratio: {result['compression_ratio']:.2f}x")
            print(f"🔢 Tokens: {result['num_tokens']}")
            
            # Compare with gzip
            import gzip
            gzip_compressed = gzip.compress(args.text.encode('utf-8'))
            gzip_ratio = len(args.text) / len(gzip_compressed)
            advantage = result['compression_ratio'] / gzip_ratio
            
            print(f"📦 vs gzip: {gzip_ratio:.2f}x (our advantage: {advantage:.1f}x)")
            
        else:
            print(f"❌ Roundtrip failed!")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Expected: {repr(result['original_text'])}")
                print(f"Got: {repr(result.get('decoded_text', 'None'))}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed: {e}", file=sys.stderr)
        sys.exit(1)


def benchmark_command(args) -> None:
    """Handle the benchmark subcommand."""
    test_cases = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "My Name is Rohan Gupta",
        "In computing, compression reduces data size using fewer bits than the original representation.",
    ]
    
    print(f"🏁 Benchmarking LLM compression")
    print(f"Model: {args.model}")
    print("=" * 70)
    
    encoder = LLMEncode(model_name=args.model, max_tokens=args.max_tokens)
    
    for i, text in enumerate(test_cases, 1):
        try:
            result = encoder.test_roundtrip(text, verbose=False)
            
            if result['success']:
                # Compare with gzip
                import gzip
                gzip_compressed = gzip.compress(text.encode('utf-8'))
                gzip_ratio = len(text) / len(gzip_compressed) if len(gzip_compressed) > 0 else 0
                advantage = result['compression_ratio'] / gzip_ratio if gzip_ratio > 0 else float('inf')
                
                print(f"{i}. {text[:40]}{'...' if len(text) > 40 else ''}")
                print(f"   LLM: {result['compression_ratio']:.1f}x | gzip: {gzip_ratio:.1f}x | advantage: {advantage:.1f}x")
            else:
                print(f"{i}. {text[:40]}{'...' if len(text) > 40 else ''}")
                print(f"   ❌ Failed")
                
        except Exception as e:
            print(f"{i}. {text[:40]}{'...' if len(text) > 40 else ''}")
            print(f"   ❌ Error: {e}")
    
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-based text compression using arithmetic coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmencode encode "Hello world"
  llmencode encode "Hello world" --format hex --verbose
  llmencode decode "828d4693a0" --metadata metadata.json
  llmencode test "Hello world"
  llmencode benchmark
        """
    )
    
    # Global options
    parser.add_argument('--model', default='qwen3-0.6b', 
                       help='Model to use (default: qwen3-0.6b)')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to process (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Encode subcommand
    encode_parser = subparsers.add_parser('encode', help='Encode text')
    encode_parser.add_argument('text', help='Text to encode')
    encode_parser.add_argument('--format', dest='output_format', 
                              choices=['hex', 'base64', 'binary'], default='hex',
                              help='Output format (default: hex)')
    encode_parser.add_argument('--save-metadata', metavar='FILE',
                              help='Save metadata to file for later decoding')
    
    # Decode subcommand  
    decode_parser = subparsers.add_parser('decode', help='Decode compressed data')
    decode_parser.add_argument('data', help='Compressed data to decode')
    decode_parser.add_argument('--format', dest='input_format',
                              choices=['hex', 'base64', 'binary'], default='hex', 
                              help='Input format (default: hex)')
    decode_parser.add_argument('--metadata', metavar='FILE', required=True,
                              help='Metadata file from encoding')
    
    # Test subcommand
    test_parser = subparsers.add_parser('test', help='Test encode/decode roundtrip')
    test_parser.add_argument('text', help='Text to test')
    
    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark tests')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate handler
    if args.command == 'encode':
        encode_command(args)
    elif args.command == 'decode':
        decode_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)


if __name__ == '__main__':
    main() 