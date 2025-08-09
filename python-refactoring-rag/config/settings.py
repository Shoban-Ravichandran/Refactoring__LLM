"""Configuration settings and constants for the RAG system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
from pathlib import Path


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ContentType(Enum):
    """Content types for processing."""
    CODE = "code"
    TEXT = "text"
    MIXED = "mixed"


@dataclass
class CodeChunkConfig:
    """Configuration for code chunking."""
    max_lines: int = 200
    min_lines: int = 5
    overlap_lines: int = 5
    preserve_functions: bool = True
    preserve_classes: bool = True
    include_imports: bool = True


@dataclass
class PDFConfig:
    """Configuration for PDF processing."""
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 50
    extract_code_blocks: bool = True
    use_pymupdf: bool = True


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_context_length: int = 3000
    include_metrics: bool = True
    focus_areas: List[str] = field(default_factory=lambda: [
        'complexity', 'performance', 'readability', 'patterns'
    ])



@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Remove BLEU and ROUGE flags
    include_codebleu: bool = True
    include_efficiency_metrics: bool = True
    
    # Core quality metric weights (should sum to ~0.8)
    answer_relevance_weight: float = 0.20
    faithfulness_weight: float = 0.20
    completeness_weight: float = 0.20
    
    # Code quality metric weight
    codebleu_weight: float = 0.20
    
    # Efficiency metric weights (should sum to ~0.2)
    latency_weight: float = 0.10  # Lower latency is better
    token_usage_weight: float = 0.10  # Lower token usage is better
    
    # Efficiency thresholds for scoring
    max_acceptable_latency_ms: float = 5000.0  # 5 seconds
    max_acceptable_tokens: int = 2000  # 2000 tokens
    
    # Evaluation behavior settings
    normalize_efficiency_metrics: bool = True  # Normalize latency and tokens to 0-1 scale
    penalize_slow_responses: bool = True  # Apply penalty for responses over threshold


@dataclass
class OptimizationConfig:
    """Configuration for NSGA-II optimization."""
    population_size: int = 150
    n_generations: int = 250
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2  # adaptive if supported
    crossover_eta: float = 15
    mutation_eta: float = 20


# Default configurations
DEFAULT_CODE_CHUNK_CONFIG = CodeChunkConfig()
DEFAULT_PDF_CONFIG = PDFConfig()
DEFAULT_RAG_CONFIG = RAGConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_OPTIMIZATION_CONFIG = OptimizationConfig()

# System constants
EMBEDDING_MODEL_DEFAULT = "jinaai/jina-embeddings-v2-base-code"
QDRANT_COLLECTION_NAME = "code_refactoring_chunks"
DEFAULT_DATASET_EXAMPLES = 1500
MAX_CHUNKS_PER_EXAMPLE = 4

# Refactoring patterns
REFACTORING_PATTERNS = {
    'extract_method': {
        'name': 'Extract Method',
        'description': 'Break down long functions into smaller, focused methods',
        'category': 'complexity_reduction',
        'complexity_reduction': 0.6,
        'readability_improvement': 0.8
    },
    'extract_class': {
        'name': 'Extract Class',
        'description': 'Create classes to group related functionality',
        'category': 'organization',
        'complexity_reduction': 0.4,
        'readability_improvement': 0.7
    },
    'replace_conditional_with_polymorphism': {
        'name': 'Replace Conditional with Polymorphism',
        'description': 'Use polymorphism instead of complex conditionals',
        'category': 'design_patterns',
        'complexity_reduction': 0.5,
        'readability_improvement': 0.6
    },
    'introduce_parameter_object': {
        'name': 'Introduce Parameter Object',
        'description': 'Group related parameters into objects',
        'category': 'parameter_management',
        'complexity_reduction': 0.3,
        'readability_improvement': 0.9
    },
    'replace_loop_with_comprehension': {
        'name': 'Replace Loop with Comprehension',
        'description': 'Use Python list/dict comprehensions',
        'category': 'pythonic_patterns',
        'complexity_reduction': 0.4,
        'readability_improvement': 0.8
    }
}

# Code quality thresholds
CODE_QUALITY_THRESHOLDS = {
    'complexity': {
        'low': 5,
        'medium': 10,
        'high': 20
    },
    'maintainability': {
        'poor': 40,
        'fair': 60,
        'good': 80
    },
    'readability': {
        'poor': 40,
        'fair': 60,
        'good': 80
    }
}


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return {
        'code_chunk': DEFAULT_CODE_CHUNK_CONFIG,
        'pdf': DEFAULT_PDF_CONFIG,
        'rag': DEFAULT_RAG_CONFIG,
        'evaluation': DEFAULT_EVALUATION_CONFIG,
        'optimization': DEFAULT_OPTIMIZATION_CONFIG
    }