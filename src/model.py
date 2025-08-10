"""
LLM model interface for filesystem operation prediction.

This module provides a clean interface to both Gemini and Hugging Face models
for generating predictions about filesystem operations and states.
"""
# HF model cache

# Standard library imports
import os
from typing import Dict, Any, List, Optional, Union, Tuple

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

# Import arithmetic coding components (will be imported when needed)


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


def get_tokenizer(model_name: str):
    """
    Get tokenizer for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tokenizer instance
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Transformers not installed. Run: pip install transformers torch")
    
    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        model_name = QWEN3_MODELS[model_name.lower()]
    
    return AutoTokenizer.from_pretrained(model_name)


def get_model(model_name: str):
    """
    Get model for a given model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model instance
    """
    try:
        from transformers import AutoModelForCausalLM
        import torch
    except ImportError:
        raise ImportError("Transformers not installed. Run: pip install transformers torch")
    
    # Convert simplified model names to full HF model names
    if model_name.lower() in QWEN3_MODELS:
        model_name = QWEN3_MODELS[model_name.lower()]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    return model


def get_model_logprobs(context: str, model, tokenizer, verbose: bool = False) -> List[float]:
    """
    Get probability distribution for next token given context.
    
    Args:
        context: Input context text
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        verbose: Whether to print debug information
        
    Returns:
        List of probabilities for each token in vocabulary
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    try:
        # Tokenize context
        input_tokens = tokenizer.encode(context, return_tensors="pt").to(model.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_tokens)
            logits = outputs.logits[0, -1, :]  # Get logits for last position
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Apply minimal probability floor for arithmetic coding stability
            probs_np = probs.cpu().numpy()
            prob_floor = 1e-15
            probs_np = np.maximum(probs_np, prob_floor)
            
            # Renormalize
            probs_np = probs_np / probs_np.sum()
            
            if verbose:
                # Show top predictions
                top_indices = np.argsort(probs_np)[-5:][::-1]
                top_tokens = [(idx, f"{probs_np[idx]:.6f}", repr(tokenizer.decode([idx]))) 
                             for idx in top_indices]
                print(f"[COMPRESSION DEBUG] Top 5 predicted tokens: {top_tokens}")
            
            return probs_np.tolist()
    
    except Exception as e:
        raise Exception(f"Model prediction failed: {type(e).__name__}: {str(e)}")





def compress_with_model_probs(text: str, model_name: str = "qwen3-0.6b", 
                            max_tokens: int = 1000, verbose: bool = False) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress text using model-predicted probabilities with arithmetic coding.
    
    Args:
        text: Input text to compress
        model_name: Model to use for probability prediction
        max_tokens: Maximum number of tokens to process
        verbose: Whether to print debug information
    
    Returns:
        Tuple of (compressed_bytes, metadata_dict)
    """
    if verbose:
        print(f"[COMPRESSION DEBUG] Input text: {repr(text)}")
    
    # Tokenize input
    tokenizer = get_tokenizer(model_name)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if len(input_ids) > max_tokens:
        raise ValueError(f"Input has {len(input_ids)} tokens, exceeds max_tokens={max_tokens}")
    
    if verbose:
        print(f"[COMPRESSION DEBUG] Input tokens: {input_ids}")
        token_texts = [tokenizer.decode([token_id]) for token_id in input_ids]
        print(f"[COMPRESSION DEBUG] Token texts: {token_texts}")
    
    # Use start token approach for consistent context
    start_token = "<START>"
    if verbose:
        print(f"[COMPRESSION DEBUG] Using start token: {repr(start_token)}")
    
    # Get model probabilities for each token
    probabilities = []
    model = get_model(model_name)
    
    for i, token_id in enumerate(input_ids):
        # Build context: start_token + tokens up to position i-1
        if i == 0:
            context = start_token
        else:
            # Decode previous tokens and append to start token
            prev_tokens_text = tokenizer.decode(input_ids[:i], skip_special_tokens=True)
            context = start_token + prev_tokens_text
        
        if verbose:
            print(f"[COMPRESSION DEBUG] Position {i}: Using context {repr(context)} to predict token {token_id}")
        
        # Get probabilities from model
        probs = get_model_logprobs(context, model, tokenizer, verbose=verbose)
        probabilities.append(probs)
        
        # Debug: show what we're trying to compress
        token_text = tokenizer.decode([token_id])
        if verbose:
            print(f"[COMPRESSION DEBUG] Token to compress: {token_id} ({repr(token_text)})")
            
            # Show target token probability
            if token_id < len(probs):
                target_prob = probs[token_id]
                print(f"[COMPRESSION DEBUG] Target token probability: {target_prob:.10f}")
            else:
                print(f"[COMPRESSION DEBUG] Target token {token_id} not in vocabulary!")
            
            # Show distribution stats
            print(f"[COMPRESSION DEBUG] Min probability in distribution: {min(probs):.10f}")
            print(f"[COMPRESSION DEBUG] Max probability in distribution: {max(probs):.10f}")
    
    # Use arithmetic coding to compress
    from llmencode.arithmetic_coding import ArithmeticCoder
    coder = ArithmeticCoder()
    
    try:
        compressed_bytes = coder.encode(input_ids, probabilities)
    except Exception as e:
        raise ValueError(f"Arithmetic coding failed: {e}")
    
    # Store metadata
    metadata = {
        'original_length': len(text),
        'num_tokens': len(input_ids),
        'model_name': model_name,
        'start_token': start_token,
        'compressed_length': len(compressed_bytes)
    }
    
    return compressed_bytes, metadata


def decompress_with_model_probs(compressed_bytes: bytes, metadata: Dict[str, Any], 
                               verbose: bool = False) -> str:
    """
    Decompress bytes back to text using model-predicted probabilities.
    
    Args:
        compressed_bytes: Compressed data
        metadata: Metadata from compression
        verbose: Whether to print debug information
    
    Returns:
        Decompressed text
    """
    model_name = metadata['model_name']
    num_tokens = metadata['num_tokens']
    start_token = metadata['start_token']
    
    if verbose:
        print(f"[DECOMPRESSION DEBUG] Starting with {num_tokens} compressed tokens")
        print(f"[DECOMPRESSION DEBUG] Start token: {repr(start_token)}")
    
    # Get model and tokenizer
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)
    
    # Use iterative arithmetic decoding to decode tokens one by one
    # This allows us to use each decoded token to build the proper context for the next one
    
    from llmencode.arithmetic_coding import ArithmeticCoder
    coder = ArithmeticCoder()
    
    # Initialize iterative decoding
    coder.decode_iterative_init(compressed_bytes)
    
    decoded_token_ids = []
    
    for i in range(num_tokens):
        # Build context using tokens decoded so far (same as encoding)
        if i == 0:
            context = start_token
        else:
            prev_tokens_text = tokenizer.decode(decoded_token_ids, skip_special_tokens=True)
            context = start_token + prev_tokens_text
        
        if verbose:
            print(f"[DECOMPRESSION DEBUG] Position {i}: Using context {repr(context)}")
        
        # Get probability distribution for this position (same as encoding)
        probs = get_model_logprobs(context, model, tokenizer, verbose=False)
        
        # Decode the next token using this probability distribution
        try:
            token_id = coder.decode_iterative_next(probs)
            decoded_token_ids.append(token_id)
            
            if verbose:
                token_text = tokenizer.decode([token_id])
                print(f"[DECOMPRESSION DEBUG] Decoded token {i}: {token_id} ({repr(token_text)})")
                
        except Exception as e:
            raise ValueError(f"Iterative decoding failed at position {i}: {e}")
    
    if verbose:
        print(f"[DECOMPRESSION DEBUG] Decoded token IDs: {decoded_token_ids}")
    
    # Convert tokens back to text
    try:
        # Remove the start token context by decoding just the actual tokens
        decoded_text = tokenizer.decode(decoded_token_ids, skip_special_tokens=True)
        return decoded_text
    except Exception as e:
        raise ValueError(f"Token decoding failed: {e}")


