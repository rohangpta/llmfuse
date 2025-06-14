"""
LLM model interface for filesystem operation prediction.

This module provides a clean interface to both Gemini and Hugging Face models
for generating predictions about filesystem operations and states.
"""

# Standard library imports
import os
from typing import Dict, Any, List, Optional, Union

# Third-party imports
import google.generativeai as genai

# Model configuration constants
DEFAULT_MODEL = "models/gemini-2.5-flash-preview-05-20"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192
API_KEY_ENV_VAR = "GEMINI_API_KEY"

# Safety settings for unrestricted content generation
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
]

# Qwen3 model sizes mapping
QWEN3_MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    # Base models
    "qwen3-0.6b-base": "Qwen/Qwen3-0.6B-Base",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
    "qwen3-14b-base": "Qwen/Qwen3-14B-Base",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B-Base",
}


def _is_huggingface_model(model_name: str) -> bool:
    """
    Determine if the model is a Hugging Face model based on naming patterns.

    Args:
        model_name: Name of the model

    Returns:
        True if it's a Hugging Face model, False otherwise
    """
    # Check if it's a Qwen3 model (case insensitive)
    if model_name.lower() in QWEN3_MODELS:
        return True

    # Check if it follows HF naming convention (org/model)
    if "/" in model_name and not model_name.startswith("models/"):
        return True

    # Check if it's a known Gemini model pattern
    if model_name.startswith("models/gemini") or model_name.startswith("gemini"):
        return False

    return False


def _get_huggingface_response(
    prompt: str,
    model_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    **kwargs,
) -> str:
    """
    Get response from a Hugging Face model.

    Args:
        prompt: Input prompt for the model
        model_name: Name of the Hugging Face model
        temperature: Sampling temperature
        max_output_tokens: Maximum number of tokens in the response
        **kwargs: Additional generation parameters

    Returns:
        Model's text response

    Raises:
        ImportError: If transformers or torch is not installed
        Exception: If model generation fails
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError as e:
        error_msg = f"Hugging Face transformers not installed. Install with: pip install transformers torch"
        print(f"ERROR: {error_msg}")
        return f"ERROR: {error_msg}"

    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        model_name = QWEN3_MODELS[model_name.lower()]

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        # Handle different prompt formats
        if "qwen3" in model_name.lower():
            # Qwen3 uses chat format
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to True for thinking mode
            )
        else:
            # For other models, use prompt directly
            text = prompt

        # Tokenize input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode response
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        error_msg = (
            f"Hugging Face model generation failed: {type(e).__name__}: {str(e)}"
        )
        print(f"ERROR: {error_msg}")
        return f"ERROR: {error_msg}"


def _get_gemini_response(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """
    Get response from a Gemini model (original implementation).

    Args:
        prompt: Input prompt for the model
        model_name: Name of the Gemini model to use
        temperature: Sampling temperature (0.0 for deterministic output)
        top_p: Top-p sampling parameter
        max_output_tokens: Maximum number of tokens in the response

    Returns:
        Model's text response

    Raises:
        ValueError: If API key is not configured
        Exception: If model generation fails
    """
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"Environment variable {API_KEY_ENV_VAR} is not set")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )

    try:
        response = model.generate_content(
            contents=prompt,
            generation_config=config,
            safety_settings=SAFETY_SETTINGS,
        )

        # Check for blocked content
        if not response.parts:
            error_msg = (
                "Model response was blocked (safety filters or other restrictions)"
            )
            print(f"ERROR: {error_msg}")
            return f"ERROR: {error_msg}"

        return response.text

    except Exception as e:
        error_msg = f"API call failed: {type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"ERROR: {error_msg}"


def get_model_response(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    **kwargs,
) -> str:
    """
    Get a response from either Gemini or Hugging Face models for a given prompt.

    This function automatically detects the model provider based on the model name
    and routes to the appropriate backend.

    Args:
        prompt: Input prompt for the model
        model_name: Name of the model to use. Can be:
            - Gemini models: "models/gemini-2.5-flash-preview-05-20" (default)
            - Qwen3 models: "qwen3-8b", "qwen3-32b", etc.
            - Full HF model names: "Qwen/Qwen3-8B", "microsoft/DialoGPT-medium", etc.
        temperature: Sampling temperature (0.0 for deterministic output)
        top_p: Top-p sampling parameter (Gemini only)
        max_output_tokens: Maximum number of tokens in the response
        **kwargs: Additional generation parameters for HF models

    Returns:
        Model's text response

    Raises:
        ValueError: If API key is not configured (for Gemini models)
        ImportError: If required libraries are not installed
        Exception: If model generation fails

    Examples:
        # Use Gemini (default)
        response = get_model_response("Hello, world!")

        # Use Qwen3-8B
        response = get_model_response("Hello, world!", model_name="qwen3-8b")

        # Use full HF model name
        response = get_model_response("Hello, world!", model_name="Qwen/Qwen3-32B")
    """
    if _is_huggingface_model(model_name):
        return _get_huggingface_response(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
    else:
        return _get_gemini_response(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )


def query_model(prompt: str) -> str:
    """
    Legacy function name for backward compatibility.

    Args:
        prompt: Input prompt for the model

    Returns:
        Model's text response
    """
    return get_model_response(prompt)


def validate_api_key() -> bool:
    """
    Check if the Gemini API key is properly configured.

    Returns:
        True if API key is available, False otherwise
    """
    return API_KEY_ENV_VAR in os.environ


def get_available_models() -> List[str]:
    """
    Get list of available Gemini models.

    Returns:
        List of model names

    Raises:
        ValueError: If API key is not configured
    """
    if not validate_api_key():
        raise ValueError(f"Environment variable {API_KEY_ENV_VAR} is not set")

    genai.configure(api_key=os.environ[API_KEY_ENV_VAR])

    try:
        models = genai.list_models()
        return [
            model.name
            for model in models
            if "generateContent" in model.supported_generation_methods
        ]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def get_available_qwen3_models() -> List[str]:
    """
    Get list of available Qwen3 model shortcuts.

    Returns:
        List of Qwen3 model names that can be used with get_model_response
    """
    return list(QWEN3_MODELS.keys())


def print_available_models() -> None:
    """
    Print all available model options.
    """
    print("Available Qwen3 models:")
    for short_name, full_name in QWEN3_MODELS.items():
        print(f"  {short_name:<20} -> {full_name}")

    print("\nGemini models:")
    print(f"  {DEFAULT_MODEL} (default)")
    print("  Or any other Gemini model name starting with 'models/gemini'")

    print(
        "\nYou can also use any Hugging Face model name directly (e.g., 'microsoft/DialoGPT-medium')"
    )


def main() -> None:
    """
    Example usage of the model interface.

    Demonstrates basic functionality with both Gemini and Hugging Face models.
    """
    print("=" * 60)
    print("Model Interface Demo")
    print("=" * 60)

    # Show available models
    print_available_models()
    print("\n" + "=" * 60)

    # Test prompt
    prompt = "Hello! Can you explain what a filesystem is in simple terms?"
    print(f"Test prompt: {prompt}")
    print("\n" + "-" * 60)

    # Test with Gemini (if API key is available)
    if validate_api_key():
        print("Testing with Gemini model...")
        try:
            response = get_model_response(prompt)
            print(f"Gemini response: {response[:200]}...")
        except Exception as e:
            print(f"Gemini error: {e}")
    else:
        print("Gemini API key not configured. Skipping Gemini test.")

    print("\n" + "-" * 60)

    # Test with Qwen3 models (if transformers is available)
    try:
        import transformers

        print("Testing with Qwen3 models...")

        # Test different Qwen3 sizes
        test_models = ["qwen3-0.6b", "qwen3-1.7b"]  # Start with smaller models

        for model_name in test_models:
            print(f"\nTesting {model_name}...")
            try:
                response = get_model_response(prompt, model_name=model_name)
                print(f"{model_name} response: {response[:200]}...")
            except Exception as e:
                print(f"{model_name} error: {e}")

    except ImportError:
        print(
            "Transformers not installed. Install with: pip install transformers torch"
        )
        print("To test Hugging Face models.")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
