# Python Code Refactoring RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for Python code refactoring suggestions, featuring multi-objective optimization for model selection and enhanced evaluation metrics.

## Features

- **Multi-LLM Support**: Integration with multiple language models (Groq, OpenAI, Anthropic)
- **Advanced Retrieval**: Code-specific embeddings with enhanced query processing
- **Multi-Objective Optimization**: NSGA-II algorithm for optimal model selection
- **Comprehensive Evaluation**: RAG metrics including BLEU, ROUGE, context relevance, and faithfulness
- **Code Quality Analysis**: Cyclomatic complexity, maintainability index, and code smell detection
- **PDF Processing**: Extract and process refactoring knowledge from PDF documents
- **Vector Storage**: Qdrant integration for efficient similarity search

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd python-refactoring-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```python
from main import main
from config.settings import get_default_config

# Run the complete system
results = main()

# Access the best model recommendation
best_model = results['nsga2_results']['best_model']
print(f"Recommended model: {best_model}")
```

## Project Structure

```
python-refactoring-rag/
├── config/           # Configuration and settings
├── data/            # Data processing and generation
├── models/          # Core algorithms and evaluation
├── services/        # Business logic and orchestration
├── utils/           # Helper utilities
└── tests/           # Test suite
```

## Configuration

The system uses environment variables for configuration:

- `GROQ_API_KEY`: Groq API key for LLM access
- `QDRANT_URL`: Qdrant cloud URL (optional)
- `QDRANT_API_KEY`: Qdrant API key (optional)

## Usage Examples

### Basic Refactoring Suggestion

```python
from services.rag_service import RefactoringRAGSystem
from config.model_configs import get_default_llm_configs

# Initialize system
system = RefactoringRAGSystem(get_default_llm_configs())
system.setup()

# Get refactoring suggestions
query = "How can I reduce complexity in nested loops?"
suggestions = system.get_refactoring_suggestions(query)
```

### Custom Evaluation

```python
from models.evaluation.rag_evaluator import RAGEvaluator
from config.settings import EvaluationConfig

evaluator = RAGEvaluator()
config = EvaluationConfig()

# Evaluate custom test cases
results = evaluator.evaluate_models(test_cases, config, model_names)
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Formatting

```bash
black .
isort .
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.