"""Enhanced main entry point with comprehensive evaluation and improved interactive mode."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

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

# Try to import enhanced display manager
try:
    from interactive_display_manager import create_interactive_session
    ENHANCED_DISPLAY_AVAILABLE = True
except ImportError:
    ENHANCED_DISPLAY_AVAILABLE = False
    logger.warning("Enhanced display manager not found. Using built-in interactive mode.")


def get_enhanced_test_cases():
    """Extended set of test cases for comprehensive evaluation."""
    return [
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
        },
        {
            'query': 'How can I eliminate code duplication in this class?',
            'original_code': '''class UserProcessor:
    def process_admin_user(self, user_data):
        if not user_data.get('email'):
            raise ValueError("Email required")
        if not user_data.get('name'):
            raise ValueError("Name required")
        user = {
            'email': user_data['email'].lower(),
            'name': user_data['name'].strip(),
            'role': 'admin',
            'permissions': ['read', 'write', 'delete']
        }
        return user
    
    def process_regular_user(self, user_data):
        if not user_data.get('email'):
            raise ValueError("Email required")
        if not user_data.get('name'):
            raise ValueError("Name required")
        user = {
            'email': user_data['email'].lower(),
            'name': user_data['name'].strip(),
            'role': 'user',
            'permissions': ['read']
        }
        return user''',
            'reference_answer': """Extract common validation and processing logic into shared methods:

```python
class UserProcessor:
    def _validate_user_data(self, user_data):
        if not user_data.get('email'):
            raise ValueError("Email required")
        if not user_data.get('name'):
            raise ValueError("Name required")
    
    def _create_base_user(self, user_data, role, permissions):
        self._validate_user_data(user_data)
        return {
            'email': user_data['email'].lower(),
            'name': user_data['name'].strip(),
            'role': role,
            'permissions': permissions
        }
    
    def process_admin_user(self, user_data):
        return self._create_base_user(user_data, 'admin', ['read', 'write', 'delete'])
    
    def process_regular_user(self, user_data):
        return self._create_base_user(user_data, 'user', ['read'])
```"""
        },
        {
            'query': 'How to improve the naming and readability of this function?',
            'original_code': '''def proc_data(d):
    r = []
    for x in d:
        if x['s'] == 'a' and x['v'] > 0:
            tmp = {'id': x['id'], 'val': x['v'] * 2, 'cat': 'pos'}
            r.append(tmp)
        elif x['s'] == 'a' and x['v'] <= 0:
            tmp = {'id': x['id'], 'val': abs(x['v']), 'cat': 'neg'}
            r.append(tmp)
    return r''',
            'reference_answer': """Improve naming and add clear structure:

```python
def process_active_data_entries(data_entries):
    \"\"\"Process active data entries and categorize by value.\"\"\"
    processed_results = []
    
    for entry in data_entries:
        if entry['status'] != 'active':
            continue
            
        processed_entry = {
            'id': entry['id'],
            'value': entry['value'] * 2 if entry['value'] > 0 else abs(entry['value']),
            'category': 'positive' if entry['value'] > 0 else 'negative'
        }
        processed_results.append(processed_entry)
    
    return processed_results
```"""
        },
        {
            'query': 'How can I optimize this slow loop performance?',
            'original_code': '''def find_common_elements(list1, list2):
    common = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in common:
                common.append(item1)
    return common''',
            'reference_answer': """Use set operations for O(n) performance:

```python
def find_common_elements(list1, list2):
    \"\"\"Find common elements using set intersection for better performance.\"\"\"
    return list(set(list1) & set(list2))

# Alternative preserving order:
def find_common_elements_ordered(list1, list2):
    \"\"\"Find common elements preserving order from first list.\"\"\"
    set2 = set(list2)
    seen = set()
    common = []
    
    for item in list1:
        if item in set2 and item not in seen:
            common.append(item)
            seen.add(item)
    
    return common
```"""
        },
        {
            'query': 'How to replace this complex conditional with polymorphism?',
            'original_code': '''def process_shape(shape_type, dimensions):
    if shape_type == 'circle':
        return 3.14159 * dimensions['radius'] ** 2
    elif shape_type == 'rectangle':
        return dimensions['width'] * dimensions['height']
    elif shape_type == 'triangle':
        return 0.5 * dimensions['base'] * dimensions['height']
    else:
        raise ValueError("Unknown shape type")''',
            'reference_answer': """Use polymorphism with classes:

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def calculate_area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def calculate_area(self):
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def calculate_area(self):
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def calculate_area(self):
        return 0.5 * self.base * self.height

def process_shape(shape: Shape):
    return shape.calculate_area()
```"""
        }
    ]


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
    """Process and index data sources with intelligent skip logic."""
    results = {'dataset_chunks': 0, 'pdf_chunks': 0}
    
    # Process dataset if provided
    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Processing dataset: {dataset_path}")
        dataset_chunks = system.process_dataset(dataset_path, force_reindex)
        results['dataset_chunks'] = dataset_chunks
        logger.info(f"Processed {dataset_chunks} chunks from dataset")
    elif dataset_path:
        logger.warning(f"Dataset file not found: {dataset_path}")
    
    # Process PDFs if provided - now with intelligent skip logic
    if pdf_paths:
        valid_pdfs = [path for path in pdf_paths if Path(path).exists()]
        if valid_pdfs:
            logger.info(f"Found {len(valid_pdfs)} valid PDF files: {[Path(p).name for p in valid_pdfs]}")
            
            # Use the enhanced PDF processing with skip logic
            pdf_chunks = system.process_pdfs(valid_pdfs, force_reindex)
            results['pdf_chunks'] = pdf_chunks
            
            if pdf_chunks > 0:
                logger.info(f"Processed {pdf_chunks} chunks from PDFs")
            else:
                logger.info("No new PDF chunks processed (already up to date)")
        else:
            logger.warning("No valid PDF files found")
    
    return results


def run_evaluation(system: RefactoringRAGSystem) -> Dict[str, Any]:
    """Run system evaluation with enhanced test cases."""
    logger.info("Running comprehensive system evaluation...")
    
    # Get enhanced test cases
    test_cases = get_enhanced_test_cases()
    
    # Run evaluation for each available model
    evaluator = RAGEvaluator()
    available_models = system.llm_provider.get_available_models()
    
    evaluation_results = {}
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {len(available_models)} MODELS ON {len(test_cases)} TEST CASES")
    print(f"{'='*80}")
    
    for model_name in available_models:
        logger.info(f"Evaluating model: {model_name}")
        model_results = []
        
        print(f"\n🔍 Testing Model: {model_name}")
        print("-" * 50)
        
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
                    print(f"  ✓ Case {i+1}: Context:{rag.context_relevance:.3f} | "
                          f"Answer:{rag.answer_relevance:.3f} | Faithful:{rag.faithfulness:.3f} | "
                          f"BLEU:{rag.bleu_score:.4f} | ROUGE:{rag.rouge_l_score:.4f}")
                else:
                    print(f"  ✗ Case {i+1}: Failed")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on case {i+1}: {e}")
                model_results.append({
                    'query': test_case['query'],
                    'model_name': model_name,
                    'success': False,
                    'error': str(e)
                })
                print(f"  ✗ Case {i+1}: Error - {str(e)[:50]}...")
        
        evaluation_results[model_name] = model_results
    
    return evaluation_results


def display_evaluation_summary(evaluation_results: Dict[str, Any]):
    """Display comprehensive evaluation summary."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Calculate aggregate metrics per model
    model_aggregates = {}
    
    for model_name, results in evaluation_results.items():
        successful_results = [r for r in results if r['success'] and r.get('rag_metrics')]
        
        if successful_results:
            # Calculate averages
            avg_context_rel = sum(r['rag_metrics'].context_relevance for r in successful_results) / len(successful_results)
            avg_answer_rel = sum(r['rag_metrics'].answer_relevance for r in successful_results) / len(successful_results)
            avg_faithfulness = sum(r['rag_metrics'].faithfulness for r in successful_results) / len(successful_results)
            avg_completeness = sum(r['rag_metrics'].response_completeness for r in successful_results) / len(successful_results)
            avg_bleu = sum(r['rag_metrics'].bleu_score for r in successful_results) / len(successful_results)
            avg_rouge = sum(r['rag_metrics'].rouge_l_score for r in successful_results) / len(successful_results)
            
            model_aggregates[model_name] = {
                'context_relevance': avg_context_rel,
                'answer_relevance': avg_answer_rel,
                'faithfulness': avg_faithfulness,
                'response_completeness': avg_completeness,
                'bleu_score': avg_bleu,
                'rouge_l_score': avg_rouge,
                'success_rate': len(successful_results) / len(results)
            }
        else:
            model_aggregates[model_name] = {
                'context_relevance': 0.0,
                'answer_relevance': 0.0,
                'faithfulness': 0.0,
                'response_completeness': 0.0,
                'bleu_score': 0.0,
                'rouge_l_score': 0.0,
                'success_rate': 0.0
            }
    
    # Display results table
    print(f"{'Model':<30} {'Context':<8} {'Answer':<8} {'Faith':<8} {'Complete':<8} {'BLEU':<8} {'ROUGE':<8} {'Success':<8}")
    print("-" * 88)
    
    for model_name, metrics in model_aggregates.items():
        print(f"{model_name:<30} "
              f"{metrics['context_relevance']:<8.3f} "
              f"{metrics['answer_relevance']:<8.3f} "
              f"{metrics['faithfulness']:<8.3f} "
              f"{metrics['response_completeness']:<8.3f} "
              f"{metrics['bleu_score']:<8.4f} "
              f"{metrics['rouge_l_score']:<8.4f} "
              f"{metrics['success_rate']:<8.2%}")
    
    return model_aggregates


def run_model_optimization(evaluation_results: Dict[str, Any], 
                          pre_optimization_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Run NSGA-II optimization and show before/after comparison."""
    logger.info("Running NSGA-II multi-objective optimization...")
    
    try:
        optimization_results = run_nsga2_optimization(evaluation_results)
        
        print(f"\n{'='*80}")
        print("NSGA-II OPTIMIZATION RESULTS & COMPARISON")
        print(f"{'='*80}")
        
        best_model = optimization_results['best_model']
        print(f"🏆 Best Model Selected: {best_model}")
        print(f"📊 Pareto Front Size: {optimization_results['pareto_front_size']}")
        print(f"⏱️  Optimization Time: {optimization_results['optimization_time_seconds']:.2f}s")
        print(f"🔧 Algorithm: {optimization_results['algorithm']}")
        
        # Show before/after metrics comparison
        print(f"\n{'='*50}")
        print("METRICS COMPARISON: BEFORE vs AFTER OPTIMIZATION")
        print(f"{'='*50}")
        
        best_metrics = optimization_results.get('best_model_objectives', {})
        pre_metrics = pre_optimization_metrics.get(best_model, {})
        
        if best_metrics and pre_metrics:
            print(f"{'Metric':<20} {'Before':<10} {'After':<10} {'Change':<10}")
            print("-" * 50)
            
            for metric in ['context_relevance', 'answer_relevance', 'faithfulness', 
                          'response_completeness', 'bleu_score', 'rouge_l_score']:
                if metric in best_metrics and metric in pre_metrics:
                    before_val = pre_metrics[metric]
                    after_val = best_metrics[metric]
                    change = after_val - before_val
                    change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
                    
                    print(f"{metric.replace('_', ' ').title():<20} "
                          f"{before_val:<10.4f} "
                          f"{after_val:<10.4f} "
                          f"{change_str:<10}")
        
        # Show all Pareto solutions
        print(f"\n{'='*50}")
        print("TOP 5 PARETO OPTIMAL SOLUTIONS")
        print(f"{'='*50}")
        
        pareto_solutions = optimization_results.get('all_pareto_solutions', [])[:5]
        for i, solution in enumerate(pareto_solutions, 1):
            model = solution.get('model', 'Unknown')
            objectives = solution.get('objectives', {})
            avg_score = sum(objectives.values()) / len(objectives) if objectives else 0
            print(f"{i}. {model:<25} (Avg Score: {avg_score:.4f})")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"Error in NSGA-II optimization: {e}")
        return {'error': str(e)}


# Optional: Install rich library for enhanced UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def display_comprehensive_response(system: RefactoringRAGSystem, query: str, 
                                 best_model: str = None, user_code: str = None):
    """Display both best model and all models' responses."""
    if not RICH_AVAILABLE:
        return display_comprehensive_response_basic(system, query, best_model, user_code)
    
    console = Console()
    
    console.print(f"\n[bold blue]Processing comprehensive response...[/bold blue]")
    console.print(f"[dim]Query: {query[:100]}{'...' if len(query) > 100 else ''}[/dim]")
    
    # Get suggestions from all models
    all_suggestions = system.get_refactoring_suggestions(query, user_code=user_code)
    
    if isinstance(all_suggestions, str):
        # Single model response
        console.print(Panel(all_suggestions, title="Response"))
        return
    
    # Display best model first if available
    if best_model and best_model in all_suggestions:
        best_suggestion = all_suggestions[best_model]
        
        # Truncate if too long but show full content
        display_text = best_suggestion
        title = f"🏆 BEST MODEL: {best_model}"
        
        console.print(Panel(display_text, title=title, border_style="green"))
        
        # Ask if user wants to see all models
        console.print(f"\n[yellow]Best model response shown above.[/yellow]")
        show_all = Prompt.ask("Show all models' responses? (y/n)", default="n")
        
        if show_all.lower() in ['y', 'yes']:
            console.print(f"\n[bold cyan]ALL MODELS' RESPONSES:[/bold cyan]")
            for model_name, suggestion in all_suggestions.items():
                if model_name != best_model:  # Skip best model as already shown
                    title = f"📋 {model_name}"
                    # Create scrollable content for long responses
                    display_text = suggestion
                    console.print(Panel(display_text, title=title, border_style="blue"))
    else:
        # Show all models
        console.print(f"\n[bold cyan]ALL MODELS' RESPONSES:[/bold cyan]")
        for model_name, suggestion in all_suggestions.items():
            title = f"📋 {model_name}"
            is_best = model_name == best_model
            border_style = "green" if is_best else "blue"
            
            if is_best:
                title = f"🏆 {title} (BEST)"
            
            display_text = suggestion
            console.print(Panel(display_text, title=title, border_style=border_style))


def display_comprehensive_response_basic(system: RefactoringRAGSystem, query: str, 
                                       best_model: str = None, user_code: str = None):
    """Basic display without Rich library."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESPONSE")
    print(f"{'='*80}")
    print(f"Query: {query}")
    
    # Get suggestions from all models
    all_suggestions = system.get_refactoring_suggestions(query, user_code=user_code)
    
    if isinstance(all_suggestions, str):
        print(f"\nResponse:\n{all_suggestions}")
        return
    
    # Display best model first
    if best_model and best_model in all_suggestions:
        print(f"\n🏆 BEST MODEL ({best_model}):")
        print("-" * 50)
        print(all_suggestions[best_model])
        
        response = input(f"\nShow all models' responses? (y/n): ").lower()
        if response in ['y', 'yes']:
            print(f"\n{'='*60}")
            print("ALL MODELS' RESPONSES")
            print(f"{'='*60}")
            
            for model_name, suggestion in all_suggestions.items():
                if model_name != best_model:
                    print(f"\n📋 {model_name}:")
                    print("-" * 40)
                    print(suggestion)
    else:
        for model_name, suggestion in all_suggestions.items():
            marker = "🏆" if model_name == best_model else "📋"
            print(f"\n{marker} {model_name}:")
            print("-" * 40)
            print(suggestion)


def interactive_mode_enhanced(system: RefactoringRAGSystem, best_model: str = None):
    """Enhanced interactive mode with best model optimization."""
    if not RICH_AVAILABLE:
        return interactive_mode_basic_enhanced(system, best_model)

    console = Console()

    # Display welcome message
    console.print(Panel.fit(
        f"[bold blue]Python Code Refactoring RAG System[/bold blue]\n"
        f"[yellow]Enhanced Interactive Mode[/yellow]\n"
        f"[green]Optimized Best Model: {best_model or 'Not determined'}[/green]",
        title="Welcome"
    ))

    # Display available commands
    console.print("\n[bold green]Commands:[/bold green]")
    console.print("  • [cyan]quit/exit[/cyan] - Exit")
    console.print("  • [cyan]stats[/cyan] - System statistics")
    console.print("  • [cyan]health[/cyan] - Health check")
    console.print("  • [cyan]best[/cyan] - Show only best model response")
    console.print("  • [cyan]all[/cyan] - Show all models' responses")
    console.print("  • [cyan]help[/cyan] - Show help")

    console.print("\n[bold green]Multi-line Input:[/bold green]")
    console.print("  • Type [yellow]###END###[/yellow] to submit")
    console.print("  • Press [yellow]CTRL+D[/yellow] to submit")
    console.print("  • Double [yellow]ENTER[/yellow] for short inputs")

    session = 0
    response_mode = "comprehensive"  # "best", "all", or "comprehensive"

    while True:
        try:
            session += 1
            console.print(f"\n[bold magenta]Session {session}[/bold magenta]")

            console.print(f"\n[bold yellow]Enter your query or code:[/bold yellow]")

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

            elif command == 'best':
                response_mode = "best"
                console.print("[green]Mode set to: Best model only[/green]")
                continue

            elif command == 'all':
                response_mode = "all"
                console.print("[green]Mode set to: All models[/green]")
                continue

            elif command == 'help':
                help_text = """[bold green]Commands:[/bold green]
• quit/exit - Exit interactive mode
• stats - Show system statistics
• health - Perform system health check
• best - Show only best model responses
• all - Show all models' responses
• help - Display help menu

[bold green]Input Tips:[/bold green]
• Paste code or ask natural language questions
• Use ###END### or CTRL+D to submit multi-line input
• Double ENTER also works for short queries
"""
                console.print(Panel(help_text, title="Help"))
                continue

            # Process the query
            console.print(f"\n[bold blue]Processing your query...[/bold blue]")
            console.print(f"[dim]Input length: {len(query)} characters, {len(lines)} lines[/dim]")
            console.print(f"[dim]Response mode: {response_mode}[/dim]")

            # Detect code in input
            has_code = any(kw in query.lower() for kw in ['def ', 'class ', 'import ', 'for ', 'if ', 'while ', '```'])
            user_code = query if has_code else None

            if response_mode == "best" and best_model:
                # Show only best model
                suggestion = system.get_refactoring_suggestions(query, model_name=best_model, user_code=user_code)
                title = f"🏆 BEST MODEL: {best_model}"
                console.print(Panel(suggestion, title=title, border_style="green"))
            
            elif response_mode == "all":
                # Show all models
                all_suggestions = system.get_refactoring_suggestions(query, user_code=user_code)
                if isinstance(all_suggestions, dict):
                    for model_name, suggestion in all_suggestions.items():
                        is_best = model_name == best_model
                        title = f"🏆 {model_name}" if is_best else f"📋 {model_name}"
                        border_style = "green" if is_best else "blue"
                        console.print(Panel(suggestion, title=title, border_style=border_style))
                else:
                    console.print(Panel(all_suggestions, title="Response"))
            
            else:
                # Comprehensive mode (default)
                display_comprehensive_response(system, query, best_model, user_code)

            console.print("[bold green]Query completed.[/bold green]")

        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
            break
        except Exception as error:
            console.print(f"[bold red]Error: {error}[/bold red]")
            console.print("[yellow]Use 'help' for available commands.[/yellow]")


def interactive_mode_basic_enhanced(system: RefactoringRAGSystem, best_model: str = None):
    """Enhanced basic interactive mode without Rich."""
    print(f"\n{'='*80}")
    print("ENHANCED INTERACTIVE MODE - Python Code Refactoring RAG System")
    print(f"{'='*80}")
    print(f"Optimized Best Model: {best_model or 'Not determined'}")

    print("\nAvailable Commands:")
    print("  • quit / exit / q     - Exit interactive mode")
    print("  • stats               - Show system statistics")
    print("  • health              - Perform system health check")
    print("  • best                - Show only best model response")
    print("  • all                 - Show all models' responses")
    print("  • pdf-status          - Show PDF processing status") 
    print("  • help                - Display help message")

    session_count = 0
    response_mode = "comprehensive"

    while True:
        try:
            session_count += 1
            print(f"\n{'='*20} Session {session_count} {'='*20}")

            query = input("\nEnter your refactoring query or code: ").strip()
            
            if not query:
                print("No input provided. Please try again.")
                continue

            command = query.lower()

            if command in ['quit', 'exit', 'q']:
                print("Exiting. Goodbye!")
                break  # This should properly exit the main loop

            elif command == 'best':
                response_mode = "best"
                print("Mode set to: Best model only")
                continue

            elif command == 'all':
                response_mode = "all"
                print("Mode set: All models")
                continue

            elif command == 'stats':
                stats = system.get_system_stats()
                print("\nSystem Statistics")
                print("-" * 40)
                for key, value in stats.items():
                    print(f"{key}: {value}")
                continue

            elif command == 'health':
                health = system.health_check()
                print("\nSystem Health Check")
                print("-" * 40)
                for component, status in health.items():
                    status_text = "HEALTHY" if status else "UNHEALTHY"
                    print(f"{component}: {status_text}")
                continue

            # Process query based on mode
            print(f"\nProcessing query in '{response_mode}' mode...")
            
            has_code = any(kw in query.lower() for kw in ['def ', 'class ', 'import ', 'for ', 'if '])
            user_code = query if has_code else None
            
            if response_mode == "best" and best_model:
                suggestion = system.get_refactoring_suggestions(query, model_name=best_model, user_code=user_code)
                print(f"\n🏆 BEST MODEL ({best_model}):")
                print("-" * 50)
                print(suggestion)
            
            elif response_mode == "all":
                display_comprehensive_response_basic(system, query, best_model, user_code)
            
            else:
                display_comprehensive_response_basic(system, query, best_model, user_code)

            print("\nQuery completed successfully.")

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")


def interactive_mode(system: RefactoringRAGSystem, best_model: str = None):
    """Select the most appropriate interactive mode with enhanced display management."""
    if ENHANCED_DISPLAY_AVAILABLE:
        # Use the new enhanced interactive session manager
        session = create_interactive_session(system, best_model)
        session.run()
    elif RICH_AVAILABLE:
        # Fallback to built-in Rich mode
        interactive_mode_enhanced(system, best_model)
    else:
        # Fallback to basic mode
        interactive_mode_basic_enhanced(system, best_model)


def main():
    """Enhanced main entry point."""
    print("Python Code Refactoring RAG System - Enhanced Edition")
    print("=" * 70)
    
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
        
        # Run comprehensive evaluation
        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE SYSTEM EVALUATION")
        print(f"{'='*70}")
        
        evaluation_results = run_evaluation(system)
        
        # Display evaluation summary and store pre-optimization metrics
        pre_optimization_metrics = display_evaluation_summary(evaluation_results)
        
        # Run optimization with comparison
        optimization_results = run_model_optimization(evaluation_results, pre_optimization_metrics)
        
        # Determine best model and set it in the system
        best_model = None
        if 'error' not in optimization_results:
            best_model = optimization_results['best_model']
            system.set_best_model(best_model, optimization_results)  # Set in system
            print(f"\n🎯 SYSTEM READY!")
            print(f"📊 Recommended Model: {best_model}")
            print("✅ System is optimized for production use!")
        else:
            print(f"\n⚠️  Optimization failed: {optimization_results['error']}")
            print("⚡ System is still functional for basic queries.")
        
        # Enhanced interactive mode with all features
        interactive_mode(system, best_model)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()