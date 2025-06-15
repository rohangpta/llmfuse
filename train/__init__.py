"""
Training package for Qwen3 models.

This package contains:
- modal_sft: Supervised fine-tuning functions using Modal
- generate_data: Data generation utilities
- generate_data_test: Tests for data generation
- train: Distributed training script
"""

# We don't import everything by default since these are large Modal functions
# Import when needed 