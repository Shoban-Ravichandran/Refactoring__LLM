"""Main entry point for the Python Code Refactoring RAG System."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any,List


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="key.env")
except ImportError:
    logger.warning("python-dotenv not available. Ensure environment variables are set manually.")

# Import system components
from config.model_configs import get_default_llm_configs
from config.settings import PDFConfig, get_default_config
from services.rag_service import RefactoringRAGSystem
from models.evaluation.rag_evaluator import RAGEvaluator
from models.optimization.nsga2_optimizer import run_nsga2_optimization
from utils.logging_utils import setup_enhanced_logging


def setup_system() -> RefactoringRAGSystem:
    """Initialize and setup the RAG system."""
    logger.info("Initializing Python Code Refactoring RAG System...")
    
    # Get LLM configurations
    try:
        llm_configs = get_default_llm_configs()
        if not llm_configs:
            logger.error("No valid LLM configurations found. Please check your API keys.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(llm_configs)} LLM configurations")
        
    except Exception as e:
        logger.error(f"Error loading LLM configurations: {e}")
        sys.exit(1)
    
    # Initialize RAG system
    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # Setup with enhanced configuration
    config = get_default_config()
    config['pdf'] = PDFConfig(
        max_chunk_size=1000,
        min_chunk_size=100,
        overlap_size=50,
        extract_code_blocks=True,
        use_pymupdf=True
    )
    
    system.setup(config=config)
    
    return system


def process_data(system: RefactoringRAGSystem, 
                dataset_path: str = None,
                pdf_paths: list = None,
                force_reindex: bool = False) -> Dict[str, int]:
    """Process and index data sources."""
    results = {'dataset_chunks': 0, 'pdf_chunks': 0}
    
    # Process dataset if provided
    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Processing dataset: {dataset_path}")
        dataset_chunks = system.process_dataset(dataset_path, force_reindex)
        results['dataset_chunks'] = dataset_chunks
        logger.info(f"Processed {dataset_chunks} chunks from dataset")
    elif dataset_path:
        logger.warning(f"Dataset file not found: {dataset_path}")
    
    # Process PDFs if provided
    if pdf_paths:
        valid_pdfs = [path for path in pdf_paths if Path(path).exists()]
        logger.info(f"Found {len(valid_pdfs)} valid PDF files to process which are: {valid_pdfs}")
        if valid_pdfs:
            logger.info(f"Processing {len(valid_pdfs)} PDF files")
            pdf_chunks = system.process_pdfs(valid_pdfs)
            results['pdf_chunks'] = pdf_chunks
            logger.info(f"Processed {pdf_chunks} chunks from PDFs")
        else:
            logger.warning("No valid PDF files found")
    
    return results


def run_evaluation(system: RefactoringRAGSystem) -> Dict[str, Any]:
    """Run system evaluation with test cases."""
    logger.info("Running system evaluation...")
    
    # Define test cases
    test_cases = [
        {
            'query': 'How can I refactor this nested function to be more readable?',
            'original_code': '''def process_orders(orders):
    results = []
    for order in orders:
        if order.get('status') == 'active':
            for item in order.get('items', []):
                if item.get('price', 0) > 100:
                    results.append(item)
    return results''',
            'reference_answer': """To refactor nested functions for better readability, extract the inner logic into separate functions. Here's the approach:

**Why this helps:**
- Reduces complexity and cognitive load
- Makes code easier to understand and test
- Follows the Single Responsibility Principle

**Refactoring approach:**
```python
def is_high_value_item(item):
    return item.get('price', 0) > 100

def get_active_order_items(order):
    if order.get('status') == 'active':
        return order.get('items', [])
    return []

def process_orders(orders):
    results = []
    for order in orders:
        items = get_active_order_items(order)
        for item in items:
            if is_high_value_item(item):
                results.append(item)
    return results
```

This refactoring extracts nested conditions into descriptive functions."""
        },
        {
            'query': 'How to reduce cyclomatic complexity?',
            'original_code': '''def calculate_discount(customer, amount, product_type):
    if customer['is_premium']:
        if product_type == 'electronics':
            return amount * 0.15 if amount > 1000 else amount * 0.10
        elif product_type == 'clothing':
            return amount * 0.20 if amount > 500 else amount * 0.15
    else:
        return amount * 0.05 if product_type == 'electronics' else amount * 0.03''',
            'reference_answer': """To reduce cyclomatic complexity, replace conditional logic with data structures. Here's how:

**Why this helps:**
- Reduces the number of decision points in code
- Makes discount rules easier to maintain and extend
- Eliminates nested conditionals

**Refactoring approach:**
```python
def calculate_discount(customer, amount, product_type):
    discount_rules = {
        ('premium', 'electronics'): (0.15, 0.10, 1000),
        ('premium', 'clothing'): (0.20, 0.15, 500),
        ('regular', 'electronics'): (0.05, 0.05, 0),
        ('regular', 'clothing'): (0.03, 0.03, 0)
    }
    
    customer_type = 'premium' if customer['is_premium'] else 'regular'
    key = (customer_type, product_type)
    
    if key in discount_rules:
        high_rate, low_rate, threshold = discount_rules[key]
        return amount * (high_rate if amount > threshold else low_rate)
    
    return 0
```

This approach uses a lookup table instead of nested conditionals."""
        }
    ]
    
    # Run evaluation for each available model
    evaluator = RAGEvaluator()
    available_models = system.llm_provider.get_available_models()
    
    evaluation_results = {}
    
    for model_name in available_models:
        logger.info(f"Evaluating model: {model_name}")
        model_results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Testing case {i+1}: {test_case['query'][:50]}...")
            
            try:
                # Get suggestion from system
                suggestion = system.get_refactoring_suggestions(
                    query=test_case['query'],
                    model_name=model_name,
                    user_code=test_case.get('original_code')
                )
                
                # Get context chunks for evaluation
                similar_chunks = system.retrieval_service.search_with_enhanced_query(
                    test_case['query'], 5
                )
                
                # Evaluate the result
                result = evaluator.evaluate_single_query(
                    query=test_case['query'],
                    context_chunks=similar_chunks,
                    answer=suggestion,
                    reference_answer=test_case.get('reference_answer')
                )
                
                result['model_name'] = model_name
                model_results.append(result)
                
                # Print brief results
                if result['success'] and result['rag_metrics']:
                    rag = result['rag_metrics']
                    print(f"  Model: {model_name}")
                    print(f"  Context Relevance: {rag.context_relevance:.3f}")
                    print(f"  Answer Relevance: {rag.answer_relevance:.3f}")
                    print(f"  Faithfulness: {rag.faithfulness:.3f}")
                    print(f"  BLEU Score: {rag.bleu_score:.4f}")
                    print(f"  ROUGE-L Score: {rag.rouge_l_score:.4f}")
                    print("-" * 50)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on case {i+1}: {e}")
                model_results.append({
                    'query': test_case['query'],
                    'model_name': model_name,
                    'success': False,
                    'error': str(e)
                })
        
        evaluation_results[model_name] = model_results
    
    return evaluation_results


def run_model_optimization(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run NSGA-II optimization for model selection."""
    logger.info("Running NSGA-II multi-objective optimization...")
    
    try:
        optimization_results = run_nsga2_optimization(evaluation_results)
        
        print("\n" + "="*60)
        print("NSGA-II OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Best Model: {optimization_results['best_model']}")
        print(f"Pareto Front Size: {optimization_results['pareto_front_size']}")
        print(f"Optimization Time: {optimization_results['optimization_time_seconds']:.2f}s")
        print(f"Algorithm: {optimization_results['algorithm']}")
        
        best_objectives = optimization_results.get('best_model_objectives', {})
        if best_objectives:
            print("\nBest Model Objectives:")
            for metric, value in best_objectives.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"Error in NSGA-II optimization: {e}")
        return {'error': str(e)}


# Optional: Install rich library for enhanced UI
# pip install rich

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def interactive_mode_rich(system: RefactoringRAGSystem):
    """Interactive mode using the Rich library for better UI and syntax highlighting."""
    if not RICH_AVAILABLE:
        print("Rich library not available. Install with: pip install rich")
        return interactive_mode_advanced(system)

    console = Console()

    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]Python Code Refactoring RAG System[/bold blue]\n"
        "[yellow]Interactive Mode with Rich UI[/yellow]",
        title="Welcome"
    ))

    # Display available commands
    console.print("\n[bold green]Commands:[/bold green]")
    console.print("  • [cyan]quit/exit[/cyan] - Exit")
    console.print("  • [cyan]stats[/cyan] - System statistics")
    console.print("  • [cyan]health[/cyan] - Health check")
    console.print("  • [cyan]help[/cyan] - Show help")

    console.print("\n[bold green]Multi-line Input:[/bold green]")
    console.print("  • Type [yellow]###END###[/yellow] to submit")
    console.print("  • Press [yellow]CTRL+D[/yellow] to submit")
    console.print("  • Double [yellow]ENTER[/yellow] for short inputs")

    session = 0

    while True:
        try:
            session += 1
            console.print(f"\n[bold magenta]Session {session}[/bold magenta]")

            console.print("\n[bold yellow]Enter your query or code:[/bold yellow]")

            lines = []
            line_num = 1
            empty_count = 0

            while True:
                try:
                    prompt_text = f"[dim]{line_num:2d}|[/dim] "
                    line = Prompt.ask(prompt_text, default="")

                    if line.strip() == '###END###':
                        break

                    if not line.strip():
                        empty_count += 1
                        if empty_count >= 2:
                            console.print("[dim]Detected double empty line. Submitting...[/dim]")
                            break
                    else:
                        empty_count = 0

                    lines.append(line)
                    line_num += 1

                except EOFError:
                    console.print("\n[dim]EOF detected. Submitting...[/dim]")
                    break

            query = '\n'.join(lines).strip()

            if not query:
                continue

            # Command handling
            command = query.lower()

            if command in ['quit', 'exit']:
                console.print("[bold red]Exiting interactive mode.[/bold red]")
                break

            elif command == 'stats':
                stats = system.get_system_stats()
                console.print(Panel(str(stats), title="System Statistics"))
                continue

            elif command == 'health':
                health = system.health_check()
                health_text = Text()
                for component, status in health.items():
                    status_text = "HEALTHY" if status else "UNHEALTHY"
                    color = "green" if status else "red"
                    health_text.append(f"{component}: ", style="white")
                    health_text.append(f"{status_text}\n", style=color)
                console.print(Panel(health_text, title="Health Check"))
                continue

            elif command == 'help':
                help_text = """[bold green]Commands:[/bold green]
• quit/exit - Exit interactive mode
• stats - Show system statistics
• health - Perform system health check
• help - Display help menu

[bold green]Input Tips:[/bold green]
• Paste code or ask natural language questions
• Use ###END### or CTRL+D to submit multi-line input
• Double ENTER also works for short queries
"""
                console.print(Panel(help_text, title="Help"))
                continue

            # Input summary
            console.print(f"\n[bold blue]Processing your input...[/bold blue]")
            console.print(f"[dim]Input length: {len(query)} characters, {len(lines)} lines[/dim]")

            if any(kw in query.lower() for kw in ['def ', 'class ', 'import ', 'for ']):
                preview = query[:200] + "..." if len(query) > 200 else query
                syntax = Syntax(preview, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title="Code Preview"))

            # Get and display suggestions
            suggestions = system.get_refactoring_suggestions(query)

            if isinstance(suggestions, dict):
                for model_name, suggestion in suggestions.items():
                    is_truncated = len(suggestion) > 10000
                    display_text = suggestion[:10000] + "\n\n[... truncated ...]" if is_truncated else suggestion
                    title = f"{model_name} Suggestion"
                    if is_truncated:
                        title += f" ({len(suggestion)} characters total)"
                    console.print(Panel(display_text, title=title))
            else:
                is_truncated = len(suggestions) > 10000
                display_text = suggestions[:10000] + "\n\n[... truncated ...]" if is_truncated else suggestions
                title = "Suggestion"
                if is_truncated:
                    title += f" ({len(suggestions)} characters total)"
                console.print(Panel(display_text, title=title))

            console.print("[bold green]Query completed.[/bold green]")

        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
            break
        except Exception as error:
            console.print(f"[bold red]Error: {error}[/bold red]")
            console.print("[yellow]Use 'help' for available commands.[/yellow]")

def get_multiline_input(prompt: str = "Enter your input") -> str:
    """
    Get multi-line input from user using various end conditions.

    Modes:
    - Press CTRL+D (Linux/Mac) or CTRL+Z (Windows) to submit
    - Type '###END###' on a new line to submit
    - Press ENTER twice to submit
    """
    print(f"\n{prompt}:")
    print("Multi-line input supported:")
    print("  - Press CTRL+D/CTRL+Z to submit")
    print("  - Type '###END###' on a new line to submit")
    print("  - Press ENTER twice to submit")
    print("-" * 50)

    lines = []
    empty_line_count = 0
    line_number = 1

    try:
        while True:
            try:
                line = input(f"{line_number:2d}| ")
                if line.strip() == '###END###':
                    break
                if not line.strip():
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        print("Detected double empty line. Submitting input...")
                        break
                    lines.append(line)
                else:
                    empty_line_count = 0
                    lines.append(line)
                line_number += 1
            except EOFError:
                print("\nDetected EOF. Submitting input...")
                break
    except KeyboardInterrupt:
        print("\nInput cancelled by user.")
        return ""

    return '\n'.join(lines).strip()


def interactive_mode_advanced(system):
    """
    Advanced interactive mode with multi-line input, command parsing, and system integration.
    """
    print("\n" + "=" * 80)
    print("ADVANCED INTERACTIVE MODE - Python Code Refactoring RAG System")
    print("=" * 80)

    print("\nAvailable Commands:")
    print("  • quit / exit / q     - Exit interactive mode")
    print("  • stats               - Show system statistics")
    print("  • health              - Perform system health check")
    print("  • clear               - Clear the screen")
    print("  • help                - Display help message")

    print("\nInput Options:")
    print("  • Use CTRL+D/CTRL+Z, '###END###', or double ENTER to submit multi-line input")

    session_count = 0

    while True:
        try:
            session_count += 1
            print(f"\n{'=' * 20} Session {session_count} {'=' * 20}")

            query = get_multiline_input("Enter your refactoring query or code")
            if not query:
                print("No input provided. Please try again.")
                continue

            query_lower = query.lower().strip()

            if query_lower in ['quit', 'exit', 'q']:
                print("Exiting. Goodbye!")
                break

            elif query_lower == 'stats':
                stats = system.get_system_stats()
                print("\nSystem Statistics")
                print("-" * 40)
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for sub_key, sub_value in value.items():
                            print(f"  {sub_key}: {sub_value}")
                    else:
                        print(f"{key}: {value}")
                continue

            elif query_lower == 'health':
                health = system.health_check()
                print("\nSystem Health Check")
                print("-" * 40)
                for component, status in health.items():
                    status_text = "HEALTHY" if status else "UNHEALTHY"
                    print(f"{component}: {status_text}")
                continue

            elif query_lower == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            elif query_lower == 'help':
                print("\nHelp")
                print("-" * 40)
                print("Commands:")
                print("  • quit / exit / q     - Exit interactive mode")
                print("  • stats               - Show system statistics")
                print("  • health              - Perform system health check")
                print("  • clear               - Clear the terminal screen")
                print("  • help                - Display this help message")
                print("\nInput Tips:")
                print("  • Paste Python code or ask questions naturally")
                print("  • Provide context for best results")
                continue

            print("\nProcessing query...")
            print(f"Input Length: {len(query)} characters, {len(query.splitlines())} lines")
            print("-" * 60)

            has_code = any(keyword in query.lower() for keyword in 
                          ['def ', 'class ', 'import ', 'for ', 'if ', 'while ', '```'])

            if has_code:
                print("Detected code input. Generating refactoring suggestions...")
            else:
                print("Detected natural language query. Generating conceptual suggestions...")

            suggestions = system.get_refactoring_suggestions(query)

            print("\n" + "=" * 60)
            print("REFACTORING SUGGESTIONS")
            print("=" * 60)

            if isinstance(suggestions, dict):
                for i, (model, suggestion) in enumerate(suggestions.items(), 1):
                    print(f"\nModel {i}: {model}")
                    print("-" * 40)

                    if len(suggestion) > 1500:
                        lines = suggestion.split('\n')
                        if len(lines) > 30:
                            print('\n'.join(lines[:30]))
                            print(f"\n... [Output truncated to 30 of {len(lines)} lines]")
                            print(f"Total characters: {len(suggestion)}")
                        else:
                            print(suggestion[:1500])
                            print(f"\n... [Output truncated to 1500 characters]")
                    else:
                        print(suggestion)
            else:
                if len(suggestions) > 1500:
                    print(suggestions[:1500])
                    print(f"\n... [Output truncated to 1500 characters]")
                else:
                    print(suggestions)

            print("\nQuery completed successfully.")

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Please try again or type 'help' for guidance.")

        try:
            continue_choice = input("\nContinue with another query? (y/n/help): ").strip().lower()
            if continue_choice in ['n', 'no', 'quit', 'exit']:
                print("Session ended. Goodbye!")
                break
            elif continue_choice in ['help', 'h']:
                print("\nQuick Help:")
                print("  • Type your query or paste code")
                print("  • Submit with double ENTER, ###END###, or CTRL+D")
        except KeyboardInterrupt:
            print("\nSession ended. Goodbye!")
            break



def interactive_mode(system: RefactoringRAGSystem):
    """Select the most appropriate interactive mode based on available libraries."""
    if RICH_AVAILABLE:
        interactive_mode_rich(system)
    else:
        interactive_mode_advanced(system)



def main():
    """Main entry point."""
    print("Python Code Refactoring RAG System")
    print("=" * 50)
    
    # Configuration
    DATASET_PATH = "E:\Shoban\Shoban-NCI\Practicum\Python_Refactoring\Python_refactoring\python-refactoring-rag\Inputs\python_legacy_refactoring_dataset_2.json"
    PDF_PATHS = [
        "E:\Shoban\Shoban-NCI\Practicum\Python_Refactoring\Python_refactoring\python-refactoring-rag\Inputs\clean-code-in-python.pdf",
        "E:\Shoban\Shoban-NCI\Practicum\Python_Refactoring\Python_refactoring\python-refactoring-rag\Inputs\\the-clean-coder-a-code-of-conduct-for-professional-programmers.pdf" 
    ]
    
    try:
        # Setup system
        system = setup_system()
        
        # Check system health
        health = system.health_check()
        if not health['overall']:
            logger.error("System health check failed. Please check the logs.")
            return
        
        print("System initialized successfully!")
        
        # Process data sources
        processing_results = process_data(
            system, 
            DATASET_PATH, 
            PDF_PATHS, 
            force_reindex=False
        )
        
        total_chunks = processing_results['dataset_chunks'] + processing_results['pdf_chunks']
        print(f"Data processing complete: {total_chunks} total chunks indexed")
        
        # Show system statistics
        stats = system.get_system_stats()
        print(f"\nSystem ready with {stats['vector_store'].get('points_count', 0)} indexed chunks")
        
        # Run evaluation
        print("\n" + "="*50)
        print("RUNNING SYSTEM EVALUATION")
        print("="*50)
        
        evaluation_results = run_evaluation(system)
        
        # Run optimization
        optimization_results = run_model_optimization(evaluation_results)
        
        # Show final results
        if 'error' not in optimization_results:
            print(f"\nRecommended Model: {optimization_results['best_model']}")
            print("System is ready for production use!")
        else:
            print(f"\nOptimization failed: {optimization_results['error']}")
            print("System is still functional for basic queries.")
        
        # Interactive mode
        interactive_mode(system)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def demo_mode():
    """Run a simple demo without full setup."""
    print("Python Code Refactoring RAG System - Demo Mode")
    print("=" * 50)
    
    try:
        system = setup_system()
        
        # Simple demo queries
        demo_queries = [
            "How can I make this function more readable?",
            "What's the best way to reduce code complexity?",
            "How to eliminate code duplication?"
        ]
        
        print("Running demo queries...")
        
        for query in demo_queries:
            print(f"\nQuery: {query}")
            try:
                suggestions = system.get_refactoring_suggestions(query)
                if isinstance(suggestions, dict) and suggestions:
                    # Show first model's suggestion
                    first_model = list(suggestions.keys())[0]
                    suggestion = suggestions[first_model]
                    print(f"Suggestion ({first_model}): {suggestion[:200]}...")
                else:
                    print(f"Suggestion: {str(suggestions)[:200]}...")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nDemo complete!")
        
    except Exception as e:
        logger.error(f"Error in demo mode: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Code Refactoring RAG System")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--interactive", action="store_true", help="Run only interactive mode")
    parser.add_argument("--dataset", type=str, help="Path to dataset file")
    parser.add_argument("--pdfs", nargs="+", help="Paths to PDF files")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing of data")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if args.demo:
        demo_mode()
    elif args.interactive:
        system = setup_system()
        if args.dataset:
            process_data(system, args.dataset, args.pdfs, args.force_reindex)
        interactive_mode(system)
    else:
        main()