"""
Arithmetic coding implementation for LLM-based compression with proper renormalization.

This module implements arithmetic coding that can work with probability distributions
from language models to achieve compression based on model predictions.
Uses proper renormalization to handle arbitrarily long sequences without precision loss.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from fractions import Fraction


class ArithmeticCoder:
    """
    Arithmetic coding implementation with proper renormalization for long sequences.
    
    Uses high-precision arithmetic with bit-level renormalization to handle
    sequences of any length without precision loss.
    """
    
    def __init__(self, precision_bits: int = 62):
        """
        Initialize the arithmetic coder.
        
        Args:
            precision_bits: Number of bits for internal precision (default 62 for safe int64)
        """
        self.precision_bits = precision_bits
        self.max_value = (1 << precision_bits) - 1
        self.half = 1 << (precision_bits - 1)
        self.quarter = 1 << (precision_bits - 2)
        self.three_quarter = 3 * self.quarter
        
    def encode(self, symbols: List[int], probabilities: List[List[float]]) -> bytes:
        """
        Encode a sequence of symbols using arithmetic coding with renormalization.
        
        Args:
            symbols: List of symbol indices to encode
            probabilities: List of probability distributions for each position
        
        Returns:
            Encoded bytes
        """
        if len(symbols) != len(probabilities):
            raise ValueError("Number of symbols must match number of probability distributions")
        
        # Initialize range
        low = 0
        high = self.max_value
        
        # Bit buffer for output
        output_bits = []
        pending_bits = 0
        
        for symbol, prob_dist in zip(symbols, probabilities):
            if symbol >= len(prob_dist):
                raise ValueError(f"Symbol {symbol} not in probability distribution of size {len(prob_dist)}")
            
            # Convert probabilities to cumulative probabilities (scaled to integer range)
            cumulative_probs = self._to_cumulative_probabilities_int(prob_dist)
            
            # Update range based on symbol probability
            range_size = high - low + 1
            
            if symbol == 0:
                symbol_low = 0
            else:
                symbol_low = cumulative_probs[symbol - 1]
            
            symbol_high = cumulative_probs[symbol]
            
            # Update range
            high = low + (range_size * symbol_high) // self.max_value - 1
            low = low + (range_size * symbol_low) // self.max_value
            
            # Renormalization - output bits when range becomes narrow
            while True:
                if high < self.half:
                    # Output 0 bit
                    output_bits.append(0)
                    for _ in range(pending_bits):
                        output_bits.append(1)
                    pending_bits = 0
                elif low >= self.half:
                    # Output 1 bit
                    output_bits.append(1)
                    for _ in range(pending_bits):
                        output_bits.append(0)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarter:
                    # Near half - increment pending bits
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                
                # Scale up the range
                low = 2 * low
                high = 2 * high + 1
        
        # Output final bits
        pending_bits += 1
        if low < self.quarter:
            output_bits.append(0)
            for _ in range(pending_bits):
                output_bits.append(1)
        else:
            output_bits.append(1)
            for _ in range(pending_bits):
                output_bits.append(0)
        
        # Convert bits to bytes
        return self._bits_to_bytes(output_bits)
    
    def decode(self, encoded_bytes: bytes, probabilities: List[List[float]], num_symbols: int) -> List[int]:
        """
        Decode bytes back to symbols using arithmetic coding with renormalization.
        
        Args:
            encoded_bytes: Bytes to decode
            probabilities: List of probability distributions for each position
            num_symbols: Number of symbols to decode
        
        Returns:
            Decoded symbol sequence
        """
        if len(probabilities) < num_symbols:
            raise ValueError("Not enough probability distributions for requested number of symbols")
        
        # Convert bytes to bit stream
        input_bits = self._bytes_to_bits(encoded_bytes)
        bit_index = 0
        
        # Initialize range and value
        low = 0
        high = self.max_value
        value = 0
        
        # Read initial value
        for i in range(self.precision_bits):
            if bit_index < len(input_bits):
                value = (value << 1) | input_bits[bit_index]
                bit_index += 1
            else:
                value = value << 1
        
        symbols = []
        
        for i in range(num_symbols):
            prob_dist = probabilities[i]
            cumulative_probs = self._to_cumulative_probabilities_int(prob_dist)
            
            # Find which symbol corresponds to the current value
            range_size = high - low + 1
            scaled_value = ((value - low + 1) * self.max_value - 1) // range_size
            
            # Find the symbol
            symbol = self._find_symbol_int(scaled_value, cumulative_probs)
            symbols.append(symbol)
            
            # Update range for next iteration
            if symbol == 0:
                symbol_low = 0
            else:
                symbol_low = cumulative_probs[symbol - 1]
            
            symbol_high = cumulative_probs[symbol]
            
            high = low + (range_size * symbol_high) // self.max_value - 1
            low = low + (range_size * symbol_low) // self.max_value
            
            # Renormalization - read bits when range becomes narrow
            while True:
                if high < self.half:
                    # Do nothing
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                    value -= self.half
                elif low >= self.quarter and high < self.three_quarter:
                    low -= self.quarter
                    high -= self.quarter
                    value -= self.quarter
                else:
                    break
                
                # Scale up the range and read next bit
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value
                
                if bit_index < len(input_bits):
                    value |= input_bits[bit_index]
                    bit_index += 1
        
        return symbols
    
    def _to_cumulative_probabilities_int(self, probabilities: List[float]) -> List[int]:
        """Convert probability distribution to cumulative probabilities as integers."""
        # Normalize probabilities to ensure they sum to 1
        total = sum(probabilities)
        if total == 0:
            # Uniform distribution fallback
            normalized_probs = [1.0 / len(probabilities)] * len(probabilities)
        else:
            normalized_probs = [p / total for p in probabilities]
        
        # Apply minimum probability threshold to prevent zero probabilities
        min_prob = 1.0 / (self.max_value * len(probabilities))
        normalized_probs = [max(p, min_prob) for p in normalized_probs]
        
        # Renormalize after applying minimum
        total_after_min = sum(normalized_probs)
        normalized_probs = [p / total_after_min for p in normalized_probs]
        
        # Convert to integer cumulative probabilities
        cumulative = []
        cumsum = 0
        
        for i, prob in enumerate(normalized_probs):
            cumsum += int(prob * self.max_value)
            # Ensure we don't exceed max_value due to rounding
            if i == len(normalized_probs) - 1:
                cumsum = self.max_value
            cumulative.append(cumsum)
        
        return cumulative
    
    def _find_symbol_int(self, value: int, cumulative_probs: List[int]) -> int:
        """Find which symbol corresponds to a given value in cumulative probability space."""
        for i, cum_prob in enumerate(cumulative_probs):
            if value < cum_prob:
                return i
        return len(cumulative_probs) - 1
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert a list of bits to bytes."""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
        
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= bits[i + j] << (7 - j)
            result.append(byte_val)
        
        return bytes(result)
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to a list of bits."""
        bits = []
        for byte_val in data:
            for i in range(8):
                bits.append((byte_val >> (7 - i)) & 1)
        return bits
    
    # Legacy methods for backward compatibility
    def _to_cumulative_probabilities(self, probabilities: List[float]) -> List[Fraction]:
        """Legacy method - convert to integer version."""
        int_cumulative = self._to_cumulative_probabilities_int(probabilities)
        return [Fraction(x, self.max_value) for x in int_cumulative]
    
    def _find_symbol(self, value: Fraction, cumulative_probs: List[Fraction]) -> int:
        """Legacy method - convert to integer version."""
        int_value = int(value * self.max_value)
        int_cumulative = [int(x * self.max_value) for x in cumulative_probs]
        return self._find_symbol_int(int_value, int_cumulative)
    
    def _fraction_to_bytes(self, fraction: Fraction) -> bytes:
        """Legacy method for compatibility."""
        # This is now handled by the bit-based encoding
        scaled_value = int(fraction * (1 << 63))
        return scaled_value.to_bytes(8, byteorder='big', signed=False)
    
    def _bytes_to_fraction(self, data: bytes) -> Fraction:
        """Legacy method for compatibility."""
        if len(data) != 8:
            raise ValueError("Expected 8 bytes for decoding")
        scaled_value = int.from_bytes(data, byteorder='big', signed=False)
        return Fraction(scaled_value, 1 << 63)

    def decode_iterative_init(self, encoded_bytes: bytes) -> None:
        """
        Initialize iterative decoding state.
        
        Args:
            encoded_bytes: Bytes to decode
        """
        # Convert bytes to bit stream
        self._input_bits = self._bytes_to_bits(encoded_bytes)
        self._bit_index = 0
        
        # Initialize range and value
        self._low = 0
        self._high = self.max_value
        self._value = 0
        
        # Read initial value
        for i in range(self.precision_bits):
            if self._bit_index < len(self._input_bits):
                self._value = (self._value << 1) | self._input_bits[self._bit_index]
                self._bit_index += 1
            else:
                self._value = self._value << 1
    
    def decode_iterative_next(self, probabilities: List[float]) -> int:
        """
        Decode the next symbol using iterative decoding.
        
        Args:
            probabilities: Probability distribution for the next symbol
            
        Returns:
            Decoded symbol
        """
        cumulative_probs = self._to_cumulative_probabilities_int(probabilities)
        
        # Find which symbol corresponds to the current value
        range_size = self._high - self._low + 1
        scaled_value = ((self._value - self._low + 1) * self.max_value - 1) // range_size
        
        # Find the symbol
        symbol = self._find_symbol_int(scaled_value, cumulative_probs)
        
        # Update range for next iteration
        if symbol == 0:
            symbol_low = 0
        else:
            symbol_low = cumulative_probs[symbol - 1]
        
        symbol_high = cumulative_probs[symbol]
        
        self._high = self._low + (range_size * symbol_high) // self.max_value - 1
        self._low = self._low + (range_size * symbol_low) // self.max_value
        
        # Renormalization - read bits when range becomes narrow
        while True:
            if self._high < self.half:
                # Do nothing
                pass
            elif self._low >= self.half:
                self._low -= self.half
                self._high -= self.half
                self._value -= self.half
            elif self._low >= self.quarter and self._high < self.three_quarter:
                self._low -= self.quarter
                self._high -= self.quarter
                self._value -= self.quarter
            else:
                break
            
            # Scale up the range and read next bit
            self._low = 2 * self._low
            self._high = 2 * self._high + 1
            self._value = 2 * self._value
            
            if self._bit_index < len(self._input_bits):
                self._value |= self._input_bits[self._bit_index]
                self._bit_index += 1
        
        return symbol


def estimate_compression_ratio(symbols: List[int], probabilities: List[List[float]]) -> float:
    """
    Estimate the compression ratio that would be achieved with given probabilities.
    
    Args:
        symbols: Symbol sequence
        probabilities: Probability distributions for each position
    
    Returns:
        Estimated compression ratio (original_size / compressed_size)
    """
    if len(symbols) != len(probabilities):
        raise ValueError("Number of symbols must match number of probability distributions")
    
    # Calculate entropy-based estimate
    total_entropy = 0.0
    
    for symbol, prob_dist in zip(symbols, probabilities):
        if symbol >= len(prob_dist):
            continue
            
        # Normalize probabilities
        total_prob = sum(prob_dist)
        if total_prob == 0:
            continue
            
        normalized_probs = [p / total_prob for p in prob_dist]
        symbol_prob = normalized_probs[symbol]
        
        if symbol_prob > 0:
            # Add entropy contribution: -log2(p)
            total_entropy += -np.log2(symbol_prob)
    
    # Estimate compressed size in bits
    estimated_compressed_bits = total_entropy
    
    # Original size (assuming 16 bits per token ID)
    original_bits = len(symbols) * 16
    
    if estimated_compressed_bits == 0:
        return float('inf')
    
    return original_bits / estimated_compressed_bits


 