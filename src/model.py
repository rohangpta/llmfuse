"""
LLM model interface for filesystem operation prediction.

This module provides a clean interface to the Gemini model for generating
predictions about filesystem operations and states.
"""
# Standard library imports
import os
from typing import Dict, Any, List, Optional

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

def get_model_response(
    prompt: str, 
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
) -> str:
    """
    Get a response from the Gemini model for a given prompt.
    
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
            error_msg = "Model response was blocked (safety filters or other restrictions)"
            print(f"ERROR: {error_msg}")
            return f"ERROR: {error_msg}"

        return response.text
    
    except Exception as e:
        error_msg = f"API call failed: {type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"ERROR: {error_msg}"

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
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def main() -> None:
    """
    Example usage of the model interface.
    
    Demonstrates basic functionality and API key validation.
    """
    if not validate_api_key():
        print(f"Error: Please set the {API_KEY_ENV_VAR} environment variable.")
        return
        
    prompt = "Hello my name is Rohan. What is my name?"
    print(f"Sending prompt: {prompt}")
    
    try:
        response_text = get_model_response(prompt)
        print("Model response:")
        print(response_text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
