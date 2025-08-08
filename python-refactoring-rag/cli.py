#!/usr/bin/env python3
"""
Command-line interface for Python Code Refactoring RAG System

Provides command-line access to various system operations.
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path="key.env")
    except ImportError:
        logger.warning("python-dotenv not available")


def generate_dataset(args):
    """Generate legacy code dataset."""
    from data.generators.legacy_code_generator import main as generate_main
    
    print("Generating legacy code dataset...")
    generate_main()
    print("Dataset generation complete!")


def process_pdfs(args):
    """Process PDF files."""
    import os
    from config.model_configs import get_default_llm_configs
    from config.settings import get_default_config, EXPERT_KNOWLEDGE_DIR
    from services.rag_service import RefactoringRAGSystem
    
    # Initialize system
    llm_configs = get_default_llm_configs()
    if not llm_configs:
        logger.error("No valid LLM configurations found")
        return
    
    # Get Qdrant configuration
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    system.setup(config=get_default_config())
    
    # Process PDFs
    pdf_files = list(EXPERT_KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in expert_knowledge directory")
        return
    
    pdf_paths = [str(pdf) for pdf in pdf_files]
    print(f"Processing {len(pdf_files)} PDF files...")
    
    chunks = system.process_pdfs(pdf_paths, force_reindex=args.force)
    print(f"Processed {chunks} chunks from PDFs")


def run_evaluation(args):
    """Run system evaluation."""
    import os
    from config.model_configs import get_default_llm_configs
    from config.settings import get_default_config
    from services.rag_service import RefactoringRAGSystem
    from models.evaluation.rag_evaluator import RAGEvaluator
    
    # Initialize system
    llm_configs = get_default_llm_configs()
    if not llm_configs:
        logger.error("No valid LLM configurations found")
        return
    
    # Get Qdrant configuration
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    system.setup(config=get_default_config())
    
    # Load data
    from config.settings import DATASETS_DIR, EXPERT_KNOWLEDGE_DIR
    
    dataset_file = DATASETS_DIR / "python_legacy_refactoring_dataset.json"
    if dataset_file.exists():
        system.process_dataset(str(dataset_file))
    
    pdf_files = list(EXPERT_KNOWLEDGE_DIR.glob("*.pdf"))
    if pdf_files:
        pdf_paths = [str(pdf) for pdf in pdf_files]
        system.process_pdfs(pdf_paths)
    
    # Run evaluation
    print("Running evaluation...")
    evaluator = RAGEvaluator()
    
    test_cases = [
        {
            'query': 'How can I simplify this complex function?',
            'original_code': 'def complex_func():\n    # complex logic here\n    pass'
        }
    ]
    
    available_models = system.llm_provider.get_available_models()
    evaluation_results = {}
    
    for model_name in available_models:
        print(f"Evaluating {model_name}...")
        model_results = []
        
        for test_case in test_cases:
            try:
                suggestion = system.get_refactoring_suggestions(
                    query=test_case['query'],
                    model_name=model_name,
                    user_code=test_case.get('original_code')
                )
                
                similar_chunks = system.retrieval_service.search_with_enhanced_query(
                    test_case['query'], 5
                )
                
                result = evaluator.evaluate_single_query(
                    query=test_case['query'],
                    context_chunks=similar_chunks,
                    answer=suggestion
                )
                
                model_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        evaluation_results[model_name] = model_results
    
    print("Evaluation complete!")
    
    # Display results
    for model_name, results in evaluation_results.items():
        successful = [r for r in results if r['success']]
        print(f"{model_name}: {len(successful)}/{len(results)} successful")


def run_interactive(args):
    """Run interactive mode."""
    import os
    from config.model_configs import get_default_llm_configs
    from config.settings import get_default_config
    from services.rag_service import RefactoringRAGSystem
    from services.interactive_service import InteractiveService
    
    # Initialize system
    llm_configs = get_default_llm_configs()
    if not llm_configs:
        logger.error("No valid LLM configurations found")
        return
    
    # Get Qdrant configuration
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    system.setup(config=get_default_config())
    
    # Load data if available
    from config.settings import DATASETS_DIR, EXPERT_KNOWLEDGE_DIR
    
    dataset_file = DATASETS_DIR / "python_legacy_refactoring_dataset.json"
    if dataset_file.exists():
        print("Loading dataset...")
        system.process_dataset(str(dataset_file))
    
    pdf_files = list(EXPERT_KNOWLEDGE_DIR.glob("*.pdf"))
    if pdf_files:
        print("Loading PDF files...")
        pdf_paths = [str(pdf) for pdf in pdf_files]
        system.process_pdfs(pdf_paths)
    
    # Run interactive mode
    interactive_service = InteractiveService(system)
    interactive_service.run()


def show_status(args):
    """Show system status."""
    from config.settings import DATASETS_DIR, EXPERT_KNOWLEDGE_DIR
    
    print("Python Code Refactoring RAG System - Status")
    print("=" * 50)
    
    # Check dataset
    dataset_file = DATASETS_DIR / "python_legacy_refactoring_dataset.json"
    print(f"Dataset: {'✓' if dataset_file.exists() else '✗'} {dataset_file}")
    
    # Check PDF files
    pdf_files = list(EXPERT_KNOWLEDGE_DIR.glob("*.pdf"))
    print(f"PDF files: {len(pdf_files)} found")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Check API keys
    import os
    api_keys = {
        'GROQ_API_KEY': bool(os.getenv('GROQ_API_KEY')),
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
        'ANTHROPIC_API_KEY': bool(os.getenv('ANTHROPIC_API_KEY'))
    }
    
    print(f"\nAPI Keys:")
    for key, available in api_keys.items():
        print(f"  {key}: {'✓' if available else '✗'}")
    
    # Check directories
    from config.settings import ensure_directories
    ensure_directories()
    print(f"\nDirectories: ✓ Created")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Python Code Refactoring RAG System CLI")
    
    # Setup environment
    setup_environment()
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate dataset command
    gen_parser = subparsers.add_parser('generate-dataset', help='Generate legacy code dataset')
    gen_parser.set_defaults(func=generate_dataset)
    
    # Process PDFs command
    pdf_parser = subparsers.add_parser('process-pdfs', help='Process PDF files')
    pdf_parser.add_argument('--force', action='store_true', help='Force reprocessing')
    pdf_parser.set_defaults(func=process_pdfs)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Run system evaluation')
    eval_parser.set_defaults(func=run_evaluation)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run interactive mode')
    interactive_parser.set_defaults(func=run_interactive)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.set_defaults(func=show_status)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()