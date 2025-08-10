# Python Code Refactoring RAG System

## Introduction

This repository presents a comprehensive Retrieval-Augmented Generation (RAG) system designed specifically for Python code refactoring recommendations. The system combines advanced information retrieval techniques with multiple large language models (LLMs) to provide intelligent, context-aware refactoring suggestions. By leveraging a curated dataset of refactoring patterns and expert knowledge from technical literature, the system addresses the critical need for automated code quality improvement tools in software development.

The system employs multi-objective optimization through NSGA-II algorithms to determine optimal model configurations and implements sophisticated evaluation metrics including CodeBLEU, ROUGE, and custom RAG-specific measures. This approach ensures that refactoring recommendations are not only semantically relevant but also technically sound and practically applicable.

## Features

### Core Capabilities

- **Multi-LLM Integration**: Supports multiple language models including Groq, OpenAI, and Anthropic APIs with intelligent model selection
- **Advanced Retrieval System**: Code-specific embeddings with enhanced query processing and semantic search capabilities
- **Multi-Objective Optimization**: NSGA-II algorithm implementation for optimal model selection based on performance, consistency, and efficiency
- **Comprehensive Evaluation Framework**: Includes BLEU, ROUGE, CodeBLEU, context relevance, and faithfulness metrics
- **Code Quality Analysis**: Automated analysis of cyclomatic complexity, maintainability index, and code smell detection
- **PDF Knowledge Integration**: Extraction and processing of refactoring knowledge from technical documents
- **Vector Database Storage**: Qdrant integration for efficient similarity search and chunk management

### Technical Architecture

- **Modular Design**: Clean separation of concerns with dedicated modules for data processing, model management, and evaluation
- **Scalable Vector Storage**: Support for both local and cloud-based Qdrant deployments
- **Intelligent Caching**: Optimized embedding generation with caching mechanisms to reduce computational overhead
- **Robust Error Handling**: Comprehensive error management and fallback mechanisms for production reliability
- **Interactive Interface**: Rich console interface with multi-line input support and response management

## Installation

### Prerequisites

- Python 3.8 or higher
- Git for repository cloning
- API keys for desired LLM providers (Groq, OpenAI, or Anthropic)

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd python-refactoring-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```bash
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
QDRANT_URL=your_qdrant_cloud_url  # Optional
QDRANT_API_KEY=your_qdrant_api_key  # Optional
```

### Dependencies

The system requires several key dependencies:

- **Core Libraries**: numpy, pandas, scikit-learn
- **Vector Database**: qdrant-client
- **Embeddings**: sentence-transformers, transformers, torch
- **LLM Providers**: groq, openai, anthropic
- **Optimization**: pymoo for NSGA-II implementation
- **Evaluation**: nltk, rouge-score for text metrics
- **Document Processing**: PyPDF2, PyMuPDF for PDF extraction
- **Code Analysis**: radon, astunparse for Python code metrics

## Usage Examples

### Basic System Initialization

```python
from main import main
from config.settings import get_default_config

# Run the complete system with evaluation and optimization
results = main()

# Access the optimized model recommendation
best_model = results['nsga2_results']['best_model']
print(f"Recommended model: {best_model}")
```

### Direct RAG System Usage

```python
from services.rag_service import RefactoringRAGSystem
from config.model_configs import get_default_llm_configs

# Initialize system
system = RefactoringRAGSystem(get_default_llm_configs())
system.setup()

# Process datasets and knowledge sources
system.process_dataset("path/to/dataset.json")
system.process_pdfs(["path/to/expert_knowledge.pdf"])

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

# Evaluate with custom test cases
results = evaluator.evaluate_models(test_cases, config, model_names)
```

### Model Optimization

```python
from models.optimization.nsga2_optimizer_2 import run_fixed_nsga2_optimization

# Run multi-objective optimization
optimization_results = run_fixed_nsga2_optimization(evaluation_results)
best_model = optimization_results['best_model']
```

## Dataset Details

### Synthetic Dataset Generation

The system includes a comprehensive dataset generator that creates 2,200 realistic Python code refactoring examples across 10 distinct refactoring patterns:

- **Extract Method**: Breaking down monolithic functions into smaller, focused methods
- **Extract Class**: Organizing related functionality into cohesive classes
- **Replace Conditional with Polymorphism**: Using inheritance and polymorphism instead of complex conditionals
- **Introduce Parameter Object**: Grouping related parameters into structured objects
- **Replace Loop with Comprehension**: Converting manual loops to Pythonic comprehensions
- **Add Type Hints**: Including type annotations for improved code clarity
- **Improve Error Handling**: Adding proper exception handling and validation
- **Eliminate Code Duplication**: Removing redundant code through extraction and abstraction
- **Improve Naming**: Using descriptive, meaningful variable and function names
- **Optimize Performance**: Improving algorithmic efficiency and resource usage

### Domain Coverage

The dataset spans 12 application domains to ensure broad applicability:

- Database operations
- Data processing and analysis
- Machine learning workflows
- User management systems
- Scientific computing
- System administration
- File management utilities
- Web development frameworks
- Game development logic
- API integration patterns
- E-commerce applications
- Financial calculations

### Quality Metrics

Each example includes comprehensive metadata:
- Cyclomatic complexity measurements (before and after refactoring)
- Maintainability index scores
- Code smell detection and classification
- Refactoring benefits and expected improvements
- Domain-specific context and application scenarios

## Methodology

### Retrieval-Augmented Generation Architecture

The system implements a sophisticated RAG pipeline combining multiple components:

1. **Query Processing**: Enhanced query understanding with semantic expansion and intent detection
2. **Embedding Generation**: Code-specific embeddings using specialized models (Jina Embeddings v2 for code)
3. **Similarity Search**: Vector-based retrieval with Qdrant for efficient nearest neighbor search
4. **Context Enhancement**: Intelligent chunk selection and ranking based on relevance scores
5. **Generation**: Multi-model response generation with grounded prompts

### Multi-Objective Optimization

The NSGA-II optimization framework evaluates models across four key objectives:

- **Performance**: Weighted combination of answer relevance, faithfulness, and completeness
- **Consistency**: Reliability and variance in model responses across test cases
- **Code Quality**: CodeBLEU scores measuring syntactic and semantic code similarity
- **Efficiency**: Response latency and token usage optimization

### Evaluation Framework

The system employs a comprehensive evaluation methodology:

- **Answer Relevance**: Measures how well responses address the specific query
- **Faithfulness**: Assesses whether responses are grounded in retrieved context
- **Response Completeness**: Evaluates thoroughness and actionability of suggestions
- **CodeBLEU**: Specialized BLEU variant for code similarity measurement
- **Context Relevance**: Quality of retrieved examples and their applicability

## Results

### Model Performance Analysis

The evaluation framework assessed multiple LLM configurations across comprehensive test scenarios. Key findings include:

- **Model Diversity**: Significant performance variations across different refactoring patterns
- **Optimization Impact**: NSGA-II optimization achieved measurable improvements in composite scores
- **Context Quality**: High-quality retrieval proved crucial for generation performance
- **Pattern Specificity**: Different models excelled at different refactoring patterns

### Optimization Outcomes

The multi-objective optimization process demonstrated:

- **Pareto Front Discovery**: Identification of optimal trade-offs between competing objectives
- **Model Selection**: Data-driven selection of best-performing configurations
- **Performance Gains**: Quantifiable improvements in evaluation metrics post-optimization
- **Efficiency Balance**: Optimal balance between response quality and computational efficiency

### System Validation

Comprehensive testing validated:

- **Scalability**: Efficient processing of large document collections and code repositories
- **Reliability**: Robust error handling and graceful degradation under various conditions
- **Usability**: Intuitive interface design enabling effective human-AI collaboration
- **Extensibility**: Modular architecture supporting future enhancements and additional models

## Limitations

### Current Constraints

Several limitations should be considered when deploying this system:

- **Language Specificity**: Current implementation focuses exclusively on Python code refactoring
- **Domain Knowledge**: Performance depends on the quality and coverage of training examples
- **Model Dependencies**: Requires access to external LLM APIs with associated costs and rate limits
- **Computational Requirements**: Vector embeddings and similarity search require substantial computational resources
- **Context Window Limitations**: Large codebases may exceed model context limits

### Evaluation Scope

The evaluation framework has inherent limitations:

- **Subjective Nature**: Code quality assessment involves subjective human judgment
- **Test Coverage**: Evaluation limited to specific refactoring patterns and scenarios
- **Ground Truth**: Synthetic dataset may not capture all real-world refactoring complexities
- **Metric Limitations**: Automated metrics may not fully capture semantic code improvements

## Future Work

### Technical Enhancements

Several avenues for future development have been identified:

- **Multi-Language Support**: Extension to additional programming languages (Java, JavaScript, C++)
- **Advanced Code Analysis**: Integration of static analysis tools for deeper code understanding
- **Dynamic Optimization**: Adaptive model selection based on query characteristics and user feedback
- **Incremental Learning**: Continuous improvement through user interaction and feedback incorporation

### Methodological Improvements

- **Enhanced Evaluation**: Development of more sophisticated code quality metrics
- **User Studies**: Comprehensive evaluation with software developers in real-world scenarios
- **Temporal Analysis**: Investigation of refactoring pattern evolution and best practice changes
- **Domain Adaptation**: Specialized models for specific application domains or codebases

### System Capabilities

- **IDE Integration**: Development of plugins for popular integrated development environments
- **Batch Processing**: Automated refactoring suggestions for entire codebases
- **Version Control Integration**: Integration with Git workflows for refactoring history tracking


## License

This project is licensed under the MIT License. See the LICENSE file for complete terms and conditions.

