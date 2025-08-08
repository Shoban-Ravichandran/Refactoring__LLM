import logging
import os
import sys
from pathlib import Path

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
from config.settings import get_default_config, ensure_directories, DATASETS_DIR, EXPERT_KNOWLEDGE_DIR
from services.rag_service import RefactoringRAGSystem
from models.evaluation.rag_evaluator import RAGEvaluator
from models.optimization.nsga2_optimizer import run_nsga2_optimization
from services.interactive_service import InteractiveService

def get_enhanced_test_cases():
    """Get comprehensive test cases for evaluation."""
    return [
        {
            'query': 'How can I refactor this long function to be more readable and maintainable?',
            'original_code': '''def process_customer_order(order_data):
    # Input validation
    if not order_data:
        return {'error': 'No order data provided'}
    if not order_data.get('customer_id'):
        return {'error': 'Customer ID required'}
    if not order_data.get('items') or len(order_data['items']) == 0:
        return {'error': 'Order must contain items'}
    
    # Calculate totals
    subtotal = 0
    for item in order_data['items']:
        if item.get('quantity', 0) <= 0:
            return {'error': f'Invalid quantity for item {item.get("name", "unknown")}'}
        if item.get('price', 0) <= 0:
            return {'error': f'Invalid price for item {item.get("name", "unknown")}'}
        item_total = item['quantity'] * item['price']
        subtotal += item_total
    
    # Apply discounts
    discount = 0
    if order_data.get('discount_code'):
        if order_data['discount_code'] == 'SAVE10':
            discount = subtotal * 0.10
        elif order_data['discount_code'] == 'SAVE20':
            discount = subtotal * 0.20
        elif order_data['discount_code'] == 'NEWCUSTOMER':
            discount = min(subtotal * 0.15, 50)
    
    # Calculate tax
    tax_rate = 0.08
    if order_data.get('shipping_state') == 'CA':
        tax_rate = 0.10
    elif order_data.get('shipping_state') == 'NY':
        tax_rate = 0.09
    
    discounted_subtotal = subtotal - discount
    tax = discounted_subtotal * tax_rate
    total = discounted_subtotal + tax
    
    return {
        'order_id': f"ORD-{order_data['customer_id']}-{len(order_data['items'])}",
        'subtotal': round(subtotal, 2),
        'discount': round(discount, 2),
        'tax': round(tax, 2),
        'total': round(total, 2),
        'status': 'processed'
    }''',
            'reference_answer': """Break down this monolithic function into smaller, focused functions:

```python
def process_customer_order(order_data):
    if not _validate_order_data(order_data):
        return _get_validation_error(order_data)
    
    subtotal = _calculate_subtotal(order_data['items'])
    if isinstance(subtotal, dict):  # Error case
        return subtotal
    
    discount = _calculate_discount(subtotal, order_data.get('discount_code'))
    tax = _calculate_tax(subtotal - discount, order_data.get('shipping_state'))
    
    return _build_order_result(order_data, subtotal, discount, tax)

def _validate_order_data(order_data):
    return (order_data and 
            order_data.get('customer_id') and 
            order_data.get('items') and 
            len(order_data['items']) > 0)

def _calculate_subtotal(items):
    subtotal = 0
    for item in items:
        if item.get('quantity', 0) <= 0 or item.get('price', 0) <= 0:
            return {'error': f'Invalid item data for {item.get("name", "unknown")}'}
        subtotal += item['quantity'] * item['price']
    return subtotal
```

This refactoring improves readability by giving each function a single responsibility."""
        },
        {
            'query': 'How can I make these loops more Pythonic and concise?',
            'original_code': '''def process_sales_data(sales_records):
    # Filter active sales
    active_sales = []
    for sale in sales_records:
        if sale.get('status') == 'completed' and sale.get('amount', 0) > 0:
            active_sales.append(sale)
    
    # Calculate commission for each sale
    commissioned_sales = []
    for sale in active_sales:
        commission_rate = 0.05 if sale.get('amount', 0) < 1000 else 0.08
        commissioned_sale = {
            'sale_id': sale['id'],
            'amount': sale['amount'],
            'commission': sale['amount'] * commission_rate,
            'salesperson': sale['salesperson']
        }
        commissioned_sales.append(commissioned_sale)
    
    return commissioned_sales''',
            'reference_answer': """Use Python comprehensions for more concise code:

```python
def process_sales_data(sales_records):
    return [
        {
            'sale_id': sale['id'],
            'amount': sale['amount'],
            'commission': sale['amount'] * (0.05 if sale['amount'] < 1000 else 0.08),
            'salesperson': sale['salesperson']
        }
        for sale in sales_records
        if sale.get('status') == 'completed' and sale.get('amount', 0) > 0
    ]
```

This is more concise and leverages Python's powerful comprehension syntax."""
        },
        {
            'query': 'How can I optimize this slow search function?',
            'original_code': '''def find_matching_records(dataset1, dataset2, match_field):
    """Find records that match between two datasets"""
    matches = []
    
    for record1 in dataset1:
        for record2 in dataset2:
            if record1.get(match_field) == record2.get(match_field):
                if record1.get(match_field) is not None:
                    match_info = {
                        'record1': record1,
                        'record2': record2,
                        'match_value': record1[match_field]
                    }
                    
                    # Check if we already have this match
                    duplicate = False
                    for existing in matches:
                        if (existing['record1'] == record1 and 
                            existing['record2'] == record2):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        matches.append(match_info)
    
    return matches''',
            'reference_answer': """Optimize using hash-based lookups for O(n) performance:

```python
from collections import defaultdict

def find_matching_records(dataset1, dataset2, match_field):
    # Create hash map for dataset2 for O(1) lookups
    dataset2_by_field = defaultdict(list)
    for record in dataset2:
        field_value = record.get(match_field)
        if field_value is not None:
            dataset2_by_field[field_value].append(record)
    
    matches = []
    seen_pairs = set()
    
    for record1 in dataset1:
        field_value = record1.get(match_field)
        if field_value is None:
            continue
            
        # O(1) lookup instead of O(n) iteration
        for record2 in dataset2_by_field[field_value]:
            pair_key = (id(record1), id(record2))
            if pair_key not in seen_pairs:
                matches.append({
                    'record1': record1,
                    'record2': record2,
                    'match_value': field_value
                })
                seen_pairs.add(pair_key)
    
    return matches
```

This optimization reduces time complexity from O(n²) to O(n) using hash-based lookups."""
        }
    ]



def setup_system():
    """Initialize and setup the RAG system."""
    logger.info("Initializing Python Code Refactoring RAG System...")
    ensure_directories()

    try:
        llm_configs = get_default_llm_configs()
        if not llm_configs:
            logger.error("No valid LLM configurations found. Please check your API keys.")
            sys.exit(1)
        logger.info(f"Loaded {len(llm_configs)} LLM configurations")
    except Exception as e:
        logger.error(f"Error loading LLM configurations: {e}")
        sys.exit(1)

    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    )
    config = get_default_config()
    system.setup(config=config)
    return system


def process_data(system, force_reindex=False):
    """Process and index data sources."""
    results = {'dataset_chunks': 0, 'pdf_chunks': 0}

    dataset_path = DATASETS_DIR / "python_legacy_refactoring_dataset.json"
    if dataset_path.exists():
        logger.info(f"Processing dataset: {dataset_path}")
        chunks = system.process_dataset(str(dataset_path), force_reindex)
        results['dataset_chunks'] = chunks
        logger.info(f"Processed {chunks} chunks from dataset")
    else:
        logger.warning(f"Dataset file not found: {dataset_path}")

    pdf_files = list(EXPERT_KNOWLEDGE_DIR.glob("*.pdf"))
    if pdf_files:
        pdf_paths = [str(pdf) for pdf in pdf_files]
        logger.info(f"Found {len(pdf_files)} PDF files: {[pdf.name for pdf in pdf_files]}")
        chunks = system.process_pdfs(pdf_paths, force_reindex)
        results['pdf_chunks'] = chunks
        if chunks > 0:
            logger.info(f"Processed {chunks} chunks from PDFs")
        else:
            logger.info("No new PDF chunks processed (already up to date)")
    else:
        logger.warning("No PDF files found in expert_knowledge directory")

    return results


def run_evaluation(system):
    """Run system evaluation on available models."""
    logger.info("Running comprehensive system evaluation...")
    evaluator = RAGEvaluator()
    available_models = system.llm_provider.get_available_models()
    evaluation_results = {}

    print("\n" + "=" * 80)
    print(f"EVALUATING {len(available_models)} MODELS ON TEST CASES")
    print("=" * 80)

    for model_name in available_models:
        logger.info(f"Evaluating model: {model_name}")
        model_results = []
        print(f"\nTesting Model: {model_name}")
        print("-" * 50)
        test_cases = get_enhanced_test_cases()  

        for i, test_case in enumerate(test_cases):
            logger.info(f"Testing case {i+1}: {test_case['query'][:50]}...")
            try:
                suggestion = system.get_refactoring_suggestions(
                    query=test_case['query'],
                    model_name=model_name,
                    user_code=test_case.get('original_code')
                )
                context_chunks = system.retrieval_service.search_with_enhanced_query(
                    test_case['query'], 5
                )
                result = evaluator.evaluate_single_query(
                    query=test_case['query'],
                    context_chunks=context_chunks,
                    answer=suggestion,
                    reference_answer=test_case.get('reference_answer')
                )
                result['model_name'] = model_name
                model_results.append(result)

                if result['success'] and result.get('rag_metrics'):
                    rag = result['rag_metrics']
                    print(f"  Case {i+1}: Context:{rag.context_relevance:.3f} | "
                          f"Answer:{rag.answer_relevance:.3f} | Faithful:{rag.faithfulness:.3f}")
                else:
                    print(f"  Case {i+1}: Failed")
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on case {i+1}: {e}")
                error_result = {
                    'query': test_case['query'],
                    'model_name': model_name,
                    'success': False,
                    'error': str(e)
                }
                model_results.append(error_result)
                print(f"  Case {i+1}: Error - {str(e)[:50]}...")

        evaluation_results[model_name] = model_results

    return evaluation_results


def display_evaluation_summary(evaluation_results):
    """Display aggregated evaluation results per model."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    model_aggregates = {}
    for model_name, results in evaluation_results.items():
        successful = [r for r in results if r['success'] and r.get('rag_metrics')]
        if successful:
            avg_context = sum(r['rag_metrics'].context_relevance for r in successful) / len(successful)
            avg_answer = sum(r['rag_metrics'].answer_relevance for r in successful) / len(successful)
            avg_faith = sum(r['rag_metrics'].faithfulness for r in successful) / len(successful)
            avg_completeness = sum(r['rag_metrics'].response_completeness for r in successful) / len(successful)
            success_rate = len(successful) / len(results)
            model_aggregates[model_name] = {
                'context_relevance': avg_context,
                'answer_relevance': avg_answer,
                'faithfulness': avg_faith,
                'response_completeness': avg_completeness,
                'success_rate': success_rate
            }
        else:
            model_aggregates[model_name] = {
                'context_relevance': 0.0,
                'answer_relevance': 0.0,
                'faithfulness': 0.0,
                'response_completeness': 0.0,
                'success_rate': 0.0
            }

    print(f"{'Model':<30} {'Context':<8} {'Answer':<8} {'Faith':<8} {'Complete':<8} {'Success':<8}")
    print("-" * 78)
    for model, metrics in model_aggregates.items():
        print(f"{model:<30} {metrics['context_relevance']:<8.3f} {metrics['answer_relevance']:<8.3f} "
              f"{metrics['faithfulness']:<8.3f} {metrics['response_completeness']:<8.3f} {metrics['success_rate']:<8.2%}")

    return model_aggregates


def display_detailed_before_after_comparison(optimization_results):
    """Display detailed before/after metrics comparison."""
    baseline = optimization_results['baseline_metrics']
    optimized = optimization_results['optimized_metrics']
    improvements = optimization_results.get('improvements', {})

    print("\n" + "=" * 80)
    print("DETAILED BEFORE vs AFTER OPTIMIZATION COMPARISON")
    print("=" * 80)

    print("\nCORE PERFORMANCE METRICS")
    print("-" * 80)
    print(f"{'Metric':<25} {'Before':<12} {'After':<12} {'Absolute':<12} {'Relative':<12} {'Status'}")
    print("-" * 80)

    core_metrics = [
        ('context_relevance', 'Context Relevance'),
        ('answer_relevance', 'Answer Relevance'),
        ('faithfulness', 'Faithfulness'),
        ('response_completeness', 'Completeness'),
        ('overall_performance', 'Overall Performance')
    ]

    for key, name in core_metrics:
        before = baseline.get(key, 0.0)
        after = optimized.get(key, 0.0)
        abs_change = improvements.get(f'{key}_absolute', 0.0)
        rel_change = improvements.get(f'{key}_relative', 0.0)
        status = "IMPROVED" if abs_change > 0.01 else "DECLINED" if abs_change < -0.01 else "STABLE"
        print(f"{name:<25} {before:<12.4f} {after:<12.4f} {abs_change:+12.4f} {rel_change:+12.2%} {status}")

    print("\nRESPONSE QUALITY METRICS")
    print("-" * 80)
    print(f"{'Metric':<25} {'Before':<12} {'After':<12} {'Absolute':<12} {'Relative':<12} {'Status'}")
    print("-" * 80)

    quality_metrics = [
        ('bleu_score', 'BLEU Score'),
        ('rouge_l_score', 'ROUGE-L Score'),
        ('success_rate', 'Success Rate')
    ]

    for key, name in quality_metrics:
        before = baseline.get(key, 0.0)
        after = optimized.get(key, 0.0)
        abs_change = improvements.get(f'{key}_absolute', 0.0)
        rel_change = improvements.get(f'{key}_relative', 0.0)
        status = "IMPROVED" if abs_change > 0.001 else "DECLINED" if abs_change < -0.001 else "STABLE"
        print(f"{name:<25} {before:<12.4f} {after:<12.4f} {abs_change:+12.4f} {rel_change:+12.2%} {status}")

    overall_abs = improvements.get('overall_absolute_improvement', 0.0)
    overall_rel = improvements.get('overall_relative_improvement', 0.0)

    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    if overall_abs > 0.02:
        print("SIGNIFICANT IMPROVEMENT ACHIEVED!")
        print(f"Overall performance improved by {overall_abs:.4f} points ({overall_rel:.2%})")
        level = "Excellent"
    elif overall_abs > 0.01:
        print("Moderate improvement achieved")
        print(f"Overall performance improved by {overall_abs:.4f} points ({overall_rel:.2%})")
        level = "Good"
    elif overall_abs > 0.005:
        print("Small improvement detected")
        print(f"Overall performance improved by {overall_abs:.4f} points ({overall_rel:.2%})")
        level = "Minor"
    elif overall_abs > 0:
        print("Minimal improvement detected")
        print(f"Overall performance improved by {overall_abs:.4f} points ({overall_rel:.2%})")
        level = "Minimal"
    elif overall_abs == 0:
        print("No change in performance")
        print("Models have very similar capabilities")
        level = "None"
    else:
        print("Performance declined")
        print(f"Overall performance decreased by {abs(overall_abs):.4f} points ({abs(overall_rel):.2%})")
        level = "Decline"

    print("\nDETAILED ANALYSIS:")

    all_improvements = [(n, improvements.get(f'{k}_relative', 0.0)) for k, n in core_metrics + quality_metrics]
    positive = sorted([(n, v) for n, v in all_improvements if v > 0.02], key=lambda x: -x[1])
    negative = sorted([(n, v) for n, v in all_improvements if v < -0.02], key=lambda x: x[1])

    if positive:
        print("Strongest improvements:")
        for name, val in positive[:3]:
            print(f"  • {name}: +{val:.2%}")

    if negative:
        print("Areas of concern:")
        for name, val in negative[:3]:
            print(f"  • {name}: {val:.2%}")

    print("\nRECOMMENDATIONS:")
    if level in ["Excellent", "Good"]:
        print("Optimization successful. Use the selected model.")
    elif level in ["Minor", "Minimal"]:
        print("Small improvements. Consider using optimized model.")
    elif level == "None":
        print("No improvement. Focus on data quality.")
    else:
        print("Optimization did not improve performance.")

    if 'all_pareto_solutions' in optimization_results:
        solutions = optimization_results['all_pareto_solutions']
        if len(solutions) > 1:
            sorted_sols = sorted(solutions, key=lambda x: x.get('metrics', {}).get('overall_performance', 0), reverse=True)
            print(f"\nTop performing models:")
            for i, sol in enumerate(sorted_sols[:3], 1):
                print(f"  {i}. {sol['model']} (Performance: {sol.get('metrics', {}).get('overall_performance', 0):.4f})")

    print("\n" + "=" * 80)


def run_model_optimization(evaluation_results):
    """Run NSGA-II optimization and display comparison."""
    logger.info("Running NSGA-II multi-objective optimization...")
    try:
        optimization_results = run_nsga2_optimization(evaluation_results)
        print("\n" + "=" * 80)
        print("NSGA-II OPTIMIZATION RESULTS & BEFORE/AFTER COMPARISON")
        print("=" * 80)
        print(f"Best Model Selected: {optimization_results['best_model']}")
        print(f"Pareto Front Size: {optimization_results['pareto_front_size']}")
        print(f"Optimization Time: {optimization_results['optimization_time_seconds']:.2f}s")
        print(f"Algorithm: {optimization_results['algorithm']}")

        if 'baseline_metrics' in optimization_results and 'optimized_metrics' in optimization_results:
            display_detailed_before_after_comparison(optimization_results)
        else:
            print("Warning: Baseline comparison data not available")

        return optimization_results
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        return {'error': str(e)}


def interactive_mode(system, best_model=None):
    """Start interactive mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    if best_model:
        system.set_best_model(best_model)
    service = InteractiveService(system)
    service.run()


def main():
    """Main entry point."""
    print("Python Code Refactoring RAG System")
    print("=" * 60)

    try:
        system = setup_system()
        health = system.health_check()
        if not health['overall']:
            logger.error("System health check failed.")
            return

        print("System initialized successfully!")
        results = process_data(system, force_reindex=False)
        total_chunks = results['dataset_chunks'] + results['pdf_chunks']
        print(f"Data processing complete: {total_chunks} total chunks indexed")

        if total_chunks == 0:
            print("\nWARNING: No data loaded. System will have limited functionality.")
            print("To improve: generate dataset or add PDFs.")

        stats = system.get_system_stats()
        print(f"\nSystem ready with {stats['vector_store'].get('points_count', 0)} indexed chunks")

        while True:
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            print("1. Run full evaluation and optimization")
            print("2. Interactive mode (quick start)")
            print("3. System statistics")
            print("4. Health check")
            print("5. Exit")

            choice = input("\nEnter your choice (1-5): ").strip()
            if choice == '1':
                print("\n" + "=" * 70)
                print("RUNNING COMPREHENSIVE SYSTEM EVALUATION")
                print("=" * 70)
                eval_results = run_evaluation(system)
                model_metrics = display_evaluation_summary(eval_results)
                opt_results = run_model_optimization(eval_results)

                if 'error' not in opt_results:
                    best = opt_results['best_model']
                    system.set_best_model(best)
                    print(f"\nSystem optimized! Best model: {best}")
                    if input("\nEnter interactive mode? (y/n): ").lower() == 'y':
                        interactive_mode(system, best)
                else:
                    print(f"\nOptimization failed: {opt_results['error']}")
            elif choice == '2':
                interactive_mode(system)
            elif choice == '3':
                stats = system.get_system_stats()
                print("\nSystem Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif choice == '4':
                health = system.health_check()
                for comp, status in health.items():
                    print(f"  {comp}: {'HEALTHY' if status else 'UNHEALTHY'}")
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Try again.")
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()