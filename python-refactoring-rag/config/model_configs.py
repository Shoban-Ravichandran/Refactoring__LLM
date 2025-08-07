"""LLM model configurations and setup."""

import os
from dataclasses import dataclass
from typing import List
from .settings import LLMProvider


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str
    provider: LLMProvider
    api_key: str
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30


def get_groq_models() -> List[LLMConfig]:
    """Get Groq model configurations."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    return [
        LLMConfig(
            model_name="llama3-70b-8192",
            provider=LLMProvider.GROQ,
            api_key=api_key,
            max_tokens=4000,
            temperature=0.1
        ),
        LLMConfig(
            model_name="gemma2-9b-it",
            provider=LLMProvider.GROQ,
            api_key=api_key,
            max_tokens=4000,
            temperature=0.1
        ),
        LLMConfig(
            model_name="qwen/qwen3-32b",
            provider=LLMProvider.GROQ,
            api_key=api_key,
            max_tokens=4000,
            temperature=0.1
        ),
        LLMConfig(
            model_name="moonshotai/kimi-k2-instruct",
            provider=LLMProvider.GROQ,
            api_key=api_key,
            max_tokens=4000,
            temperature=0.1
        ),
        LLMConfig(
            model_name="deepseek-r1-distill-llama-70b",
            provider=LLMProvider.GROQ,
            api_key=api_key,
            max_tokens=4000,
            temperature=0.1
        )
    ]


def get_openai_models() -> List[LLMConfig]:
    """Get OpenAI model configurations."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return [
        LLMConfig(
            model_name="gpt-4",
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1
        ),
        LLMConfig(
            model_name="gpt-3.5-turbo",
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1
        )
    ]


def get_anthropic_models() -> List[LLMConfig]:
    """Get Anthropic model configurations."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    return [
        LLMConfig(
            model_name="claude-3-opus-20240229",
            provider=LLMProvider.ANTHROPIC,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1
        ),
        LLMConfig(
            model_name="claude-3-sonnet-20240229",
            provider=LLMProvider.ANTHROPIC,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1
        )
    ]


def get_default_llm_configs() -> List[LLMConfig]:
    """Get default LLM configurations (Groq models)."""
    try:
        return get_groq_models()
    except ValueError:
        # Fallback to empty list if no API key available
        return []


def get_all_available_models() -> List[LLMConfig]:
    """Get all available model configurations."""
    models = []
    
    # Try to get models from each provider
    for provider_func in [get_groq_models, get_openai_models, get_anthropic_models]:
        try:
            models.extend(provider_func())
        except ValueError:
            # Skip providers without API keys
            continue
    
    return models


def validate_model_config(config: LLMConfig) -> bool:
    """Validate a model configuration."""
    if not config.api_key:
        return False
    if not config.model_name:
        return False
    if config.max_tokens <= 0:
        return False
    if not 0 <= config.temperature <= 2:
        return False
    
    return True


def filter_valid_configs(configs: List[LLMConfig]) -> List[LLMConfig]:
    """Filter out invalid model configurations."""
    return [config for config in configs if validate_model_config(config)]