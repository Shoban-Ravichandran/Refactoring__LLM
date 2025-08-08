"""
Interactive Demo Mode for Python Code Refactoring RAG System

This demo provides a streamlined experience for demonstrating the system capabilities
without requiring full evaluation and optimization setup.
"""

import logging
import sys
from pathlib import Path

# Setup basic logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_demo_environment():
    """Setup minimal environment for demo."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path="key.env")
        logger.info("Loaded environment variables")
    except ImportError:
        logger.warning("python-dotenv not available. Ensure environment variables are set manually.")

def check_demo_requirements():
    """Check if basic requirements are met for demo."""
    import os
    
    # Check for at least one API key
    api_keys = {
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
    }
    
    available_keys = [name for name, key in api_keys.items() if key]
    
    if not available_keys:
        logger.error("No API keys found. Please set at least one of: GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        return False
    
    logger.info(f"Found API keys for: {', '.join(available_keys)}")
    return True

def initialize_demo_system():
    """Initialize system with minimal setup for demo."""
    import os
    from config.model_configs import get_default_llm_configs
    from config.settings import get_default_config, ensure_directories
    from services.rag_service import RefactoringRAGSystem
    
    logger.info("Initializing demo system...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Get available LLM configurations
    try:
        llm_configs = get_default_llm_configs()
        if not llm_configs:
            logger.error("No valid LLM configurations found. Please check your API keys.")
            return None
        
        logger.info(f"Loaded {len(llm_configs)} LLM configurations")
        
    except Exception as e:
        logger.error(f"Error loading LLM configurations: {e}")
        return None
    
    # Get Qdrant configuration from environment
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if qdrant_url and qdrant_api_key:
        logger.info(f"Using Qdrant Cloud: {qdrant_url}")
    else:
        logger.info("Using local Qdrant (no cloud credentials found)")
    
    # Initialize RAG system with Qdrant credentials
    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    
    # Setup with demo configuration
    config = get_default_config()
    system.setup(config=config)
    
    return system

def load_demo_data(system):
    """Load sample data for demo if available."""
    from config.settings import DATASETS_DIR, EXPERT_KNOWLEDGE_DIR
    
    # Check for dataset
    dataset_file = DATASETS_DIR / "python_legacy_refactoring_dataset.json"
    pdf_files = list(EXPERT_KNOWLEDGE_DIR.glob("*.pdf"))
    
    total_chunks = 0
    
    # Load dataset if available
    if dataset_file.exists():
        logger.info(f"Loading dataset: {dataset_file}")
        try:
            chunks = system.process_dataset(str(dataset_file))
            total_chunks += chunks
            logger.info(f"Loaded {chunks} chunks from dataset")
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
    else:
        logger.info("No dataset found. Generate one with: python -m data.generators.legacy_code_generator")
    
    # Load PDFs if available
    if pdf_files:
        logger.info(f"Loading {len(pdf_files)} PDF files...")
        try:
            pdf_paths = [str(pdf) for pdf in pdf_files]
            chunks = system.process_pdfs(pdf_paths)
            total_chunks += chunks
            logger.info(f"Loaded {chunks} chunks from PDFs")
        except Exception as e:
            logger.warning(f"Failed to load PDFs: {e}")
    else:
        logger.info("No PDF files found in expert_knowledge directory")
    
    if total_chunks == 0:
        logger.warning("No data loaded. System will have limited functionality.")
        return False
    
    logger.info(f"Demo data loaded successfully: {total_chunks} total chunks")
    return True

def run_demo_queries(system):
    """Run some demonstration queries."""
    demo_queries = [
        {
            "query": "How can I simplify this complex nested function?",
            "code": '''def process_orders(orders, discount_codes, tax_rates):
    results = []
    for order in orders:
        if order['status'] == 'pending':
            total = 0
            for item in order['items']:
                if item['quantity'] > 0:
                    item_total = item['price'] * item['quantity']
                    if order.get('discount_code') in discount_codes:
                        if discount_codes[order['discount_code']]['type'] == 'percentage':
                            discount = item_total * (discount_codes[order['discount_code']]['value'] / 100)
                        else:
                            discount = min(discount_codes[order['discount_code']]['value'], item_total)
                        item_total -= discount
                    total += item_total
            
            if order['shipping_country'] in tax_rates:
                tax = total * tax_rates[order['shipping_country']]
                total += tax
            
            results.append({
                'order_id': order['id'],
                'customer': order['customer'],
                'total': round(total, 2)
            })
    
    return results''',
            "description": "Complex order processing function with nested logic"
        },
        {
            "query": "How can I make this code more readable and maintainable?",
            "code": '''def calc_user_score(u_data):
    s = 0
    for a in u_data['actions']:
        if a['t'] == 'click':
            s += 1
        elif a['t'] == 'view':
            s += 0.5
        elif a['t'] == 'purchase':
            s += 10
    
    if u_data['premium']:
        s *= 1.5
    
    return s''',
            "description": "Function with poor naming and unclear logic"
        },
        {
            "query": "How can I optimize this slow search function?",
            "code": '''def find_matching_users(users, criteria):
    matches = []
    for user in users:
        for criterion in criteria:
            if criterion['field'] == 'age':
                if user['age'] >= criterion['min'] and user['age'] <= criterion['max']:
                    if user not in matches:
                        matches.append(user)
            elif criterion['field'] == 'location':
                if user['city'] == criterion['value']:
                    if user not in matches:
                        matches.append(user)
            elif criterion['field'] == 'interests':
                for interest in user['interests']:
                    if interest in criterion['values']:
                        if user not in matches:
                            matches.append(user)
                        break
    return matches''',
            "description": "Inefficient search with O(n²) complexity"
        }
    ]
    
    print("\n" + "="*70)
    print("DEMO QUERIES - See the system in action!")
    print("="*70)
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n--- Demo {i}: {demo['description']} ---")
        print(f"Query: {demo['query']}")
        
        if demo.get('code'):
            print("\nOriginal Code:")
            print("```python")
            print(demo['code'])
            print("```")
        
        print("\nProcessing...")
        
        try:
            # Get suggestion from the best available model
            suggestions = system.get_refactoring_suggestions(
                demo['query'], 
                user_code=demo.get('code')
            )
            
            if isinstance(suggestions, dict):
                # Multiple models returned
                model_name = list(suggestions.keys())[0]
                suggestion = suggestions[model_name]
                print(f"\nSuggestion from {model_name}:")
            else:
                # Single model returned
                print("\nSuggestion:")
                suggestion = suggestions
            
            print(suggestion)
            
        except Exception as e:
            print(f"Error getting suggestion: {e}")
        
        # Ask if user wants to continue
        if i < len(demo_queries):
            response = input("\nPress Enter to continue to next demo, or 'q' to quit: ")
            if response.lower() == 'q':
                break

def run_interactive_demo(system):
    """Run interactive demo mode."""
    from services.interactive_service import InteractiveService
    
    print("\n" + "="*70)
    print("INTERACTIVE DEMO MODE")
    print("="*70)
    print("Enter your own refactoring questions or paste code for suggestions.")
    print("Type 'help' for commands, 'demo' for sample queries, or 'quit' to exit.")
    
    interactive_service = InteractiveService(system)
    interactive_service.run()

def main():
    """Main demo entry point."""
    print("Python Code Refactoring RAG System - Demo Mode")
    print("="*60)
    
    # Setup environment
    setup_demo_environment()
    
    # Check requirements
    if not check_demo_requirements():
        sys.exit(1)
    
    # Initialize system
    system = initialize_demo_system()
    if not system:
        logger.error("Failed to initialize system")
        sys.exit(1)
    
    # Load demo data
    data_loaded = load_demo_data(system)
    
    if not data_loaded:
        print("\nWARNING: No data loaded. The system will have limited functionality.")
        print("To get the full experience:")
        print("1. Generate a dataset: python -m data.generators.legacy_code_generator")
        print("2. Add PDF files to inputs/expert_knowledge/ directory")
        print("3. Restart the demo")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Check system health
    health = system.health_check()
    if not health.get('overall', False):
        logger.error("System health check failed")
        for component, status in health.items():
            if not status:
                logger.error(f"  {component}: FAILED")
        sys.exit(1)
    
    logger.info("System health check passed")
    
    # Show menu
    while True:
        print("\n" + "="*50)
        print("DEMO OPTIONS")
        print("="*50)
        print("1. Run sample queries (automated demo)")
        print("2. Interactive mode (ask your own questions)")
        print("3. System statistics")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_demo_queries(system)
        elif choice == '2':
            run_interactive_demo(system)
        elif choice == '3':
            stats = system.get_system_stats()
            print("\nSystem Statistics:")
            print(f"  Status: {stats.get('status', 'unknown')}")
            print(f"  Available models: {stats.get('llm_models', [])}")
            print(f"  Vector store points: {stats.get('vector_store', {}).get('points_count', 0)}")
            if 'pdf_processing' in stats:
                pdf_stats = stats['pdf_processing']
                print(f"  PDF files processed: {len(pdf_stats.get('pdf_files', {}))}")
                print(f"  Total PDF chunks: {pdf_stats.get('total_chunks', 0)}")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()