"""Interactive service for demo and user interaction."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("Rich not available. Using basic console output.")


class InteractiveService:
    """Interactive service for user interaction and demos."""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.console = Console() if RICH_AVAILABLE else None
        self.session_count = 0
    
    def run(self):
        """Run interactive session."""
        if RICH_AVAILABLE:
            self._run_rich_interactive()
        else:
            self._run_basic_interactive()
    
    def _run_rich_interactive(self):
        """Run interactive session with Rich formatting."""
        self.console.print(Panel.fit(
            "[bold blue]Interactive Refactoring Assistant[/bold blue]\n"
            "[yellow]Ask questions about code refactoring or paste code for suggestions[/yellow]",
            title="Welcome"
        ))
        
        self._show_help()
        
        while True:
            try:
                self.session_count += 1
                self.console.print(f"\n[bold magenta]Session {self.session_count}[/bold magenta]")
                
                # Get user input
                query = self._get_rich_input()
                
                if not query:
                    continue
                
                # Handle commands
                if self._handle_command(query):
                    continue
                
                # Process query
                self._process_query_rich(query)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def _run_basic_interactive(self):
        """Run interactive session with basic formatting."""
        print("\n" + "="*60)
        print("Interactive Refactoring Assistant")
        print("="*60)
        print("Ask questions about code refactoring or paste code for suggestions")
        print("Commands: help, demo, stats, quit")
        
        while True:
            try:
                self.session_count += 1
                print(f"\n--- Session {self.session_count} ---")
                
                query = input("Enter your query (or 'help' for commands): ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if self._handle_command(query):
                    continue
                
                # Process query
                self._process_query_basic(query)
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _get_rich_input(self) -> str:
        """Get user input with Rich interface. Requires '###END###' to finish."""
        self.console.print("\n[bold yellow]Enter your query or code:[/bold yellow]")
        self.console.print("[dim]• End multi-line input with '###END###'[/dim]")
        lines = []
        while True:
            try:
                line = Prompt.ask(f"[dim]{len(lines)+1:2d}|[/dim]", default="")
                if line.strip() == '###END###':
                    break
                lines.append(line)
            except EOFError:
                break
        return '\n'.join(lines).strip()
    
    def _handle_command(self, query: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = query.lower().strip()
        
        if command in ['quit', 'exit', 'q']:
            if RICH_AVAILABLE:
                self.console.print("[green]Goodbye![/green]")
            else:
                print("Goodbye!")
            exit(0)
        
        elif command == 'help':
            self._show_help()
            return True
        
        elif command == 'demo':
            self._show_demo_queries()
            return True
        
        elif command == 'stats':
            self._show_stats()
            return True
        
        elif command == 'clear':
            if RICH_AVAILABLE:
                self.console.clear()
            else:
                print("\n" * 50)
            return True
        
        return False
    
    def _show_help(self):
        """Show help information."""
        help_text = """[bold green]Available Commands:[/bold green]
• [cyan]help[/cyan] - Show this help message
• [cyan]demo[/cyan] - Show sample queries
• [cyan]stats[/cyan] - Show system statistics  
• [cyan]clear[/cyan] - Clear screen
• [cyan]quit[/cyan] - Exit

[bold green]Usage Tips:[/bold green]
• Ask natural language questions about refactoring
• Paste code and ask for improvement suggestions
• Use specific terms like "complexity", "readability", "performance"
• For multi-line code, end with ###END###

[bold green]Example Queries:[/bold green]
• "How can I reduce complexity in this function?"
• "Make this code more readable"
• "Optimize this loop for better performance"
• "Extract this into smaller methods"
"""
        
        if RICH_AVAILABLE:
            self.console.print(Panel(help_text, title="Help"))
        else:
            print("\nHelp:")
            print(help_text.replace('[bold green]', '').replace('[/bold green]', '').replace('[cyan]', '').replace('[/cyan]', ''))
    
    def _show_demo_queries(self):
        """Show sample demo queries."""
        demo_queries = [
            "How can I simplify this complex nested function?",
            "What's the best way to handle errors in this code?",
            "How can I make this code more Pythonic?",
            "How do I extract repeated logic into reusable functions?",
            "What design patterns can improve this code structure?"
        ]
        
        if RICH_AVAILABLE:
            demo_text = "\n".join([f"• {query}" for query in demo_queries])
            self.console.print(Panel(demo_text, title="Sample Queries"))
        else:
            print("\nSample Queries:")
            for query in demo_queries:
                print(f"  • {query}")
    
    def _show_stats(self):
        """Show system statistics."""
        try:
            stats = self.rag_system.get_system_stats()
            
            if RICH_AVAILABLE:
                stats_text = f"""[bold]Status:[/bold] {stats.get('status', 'unknown')}
[bold]Available Models:[/bold] {', '.join(stats.get('llm_models', []))}
[bold]Vector Store Points:[/bold] {stats.get('vector_store', {}).get('points_count', 0)}
[bold]Session Count:[/bold] {self.session_count}"""
                
                if 'pdf_processing' in stats:
                    pdf_stats = stats['pdf_processing']
                    stats_text += f"\n[bold]PDF Files:[/bold] {len(pdf_stats.get('pdf_files', {}))}"
                    stats_text += f"\n[bold]PDF Chunks:[/bold] {pdf_stats.get('total_chunks', 0)}"
                
                self.console.print(Panel(stats_text, title="System Statistics"))
            else:
                print("\nSystem Statistics:")
                print(f"  Status: {stats.get('status', 'unknown')}")
                print(f"  Available Models: {', '.join(stats.get('llm_models', []))}")
                print(f"  Vector Store Points: {stats.get('vector_store', {}).get('points_count', 0)}")
                print(f"  Session Count: {self.session_count}")
                
        except Exception as e:
            error_msg = f"Error getting stats: {e}"
            if RICH_AVAILABLE:
                self.console.print(f"[red]{error_msg}[/red]")
            else:
                print(error_msg)
    
    def _process_query_rich(self, query: str):
        """Process query with Rich formatting."""
        # Detect if query contains code
        has_code = self._detect_code(query)
        user_code = query if has_code else None
        
        if has_code:
            self._show_code_preview_rich(query)
        
        self.console.print(f"\n[bold blue]Processing your query...[/bold blue]")
        
        try:
            # Get suggestions
            suggestions = self.rag_system.get_refactoring_suggestions(
                query, user_code=user_code
            )
            
            if isinstance(suggestions, dict):
                # Multiple models
                for model_name, suggestion in suggestions.items():
                    self._display_suggestion_rich(model_name, suggestion)
            else:
                # Single suggestion
                self._display_suggestion_rich("Assistant", suggestions)
                
        except Exception as e:
            self.console.print(f"[red]Error processing query: {e}[/red]")
    
    def _process_query_basic(self, query: str):
        """Process query with basic formatting."""
        has_code = self._detect_code(query)
        user_code = query if has_code else None
        
        if has_code:
            print("\nCode detected in input.")
        
        print("\nProcessing your query...")
        
        try:
            suggestions = self.rag_system.get_refactoring_suggestions(
                query, user_code=user_code
            )
            
            if isinstance(suggestions, dict):
                for model_name, suggestion in suggestions.items():
                    print(f"\n--- {model_name} ---")
                    print(suggestion)
            else:
                print(f"\nSuggestion:")
                print(suggestions)
                
        except Exception as e:
            print(f"Error processing query: {e}")
    
    def _show_code_preview_rich(self, code: str):
        """Show code preview with syntax highlighting."""
        if len(code) > 500:
            preview = code[:500] + "\n... (truncated)"
        else:
            preview = code
        
        try:
            syntax = Syntax(preview, "python", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="Code Preview"))
        except:
            # Fallback if syntax highlighting fails
            self.console.print(Panel(preview, title="Code Preview"))
    
    def _display_suggestion_rich(self, model_name: str, suggestion: str):
        """Display suggestion with Rich formatting."""
        title = f"Suggestion from {model_name}"
        self.console.print(Panel(suggestion, title=title, border_style="green"))
    
    def _detect_code(self, text: str) -> bool:
        """Detect if text contains code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if __name__',
            'for ', 'while ', 'try:', 'except:', 'return ',
            '    ', '\t'  # Indentation
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators) or text.count('\n') > 3