"""Enhanced interactive display manager that handles full responses without truncation."""

import os
import sys
from typing import Dict, Any
import textwrap

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ResponseDisplayManager:
    """Manages display of model responses with proper handling of long content."""
    
    def __init__(self, use_rich: bool = None):
        self.use_rich = use_rich if use_rich is not None else RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()
    
    def display_response(self, model_name: str, response: str, 
                        is_best: bool = False, show_full: bool = True):
        """Display a single model response with proper formatting."""
        if self.use_rich:
            self._display_response_rich(model_name, response, is_best, show_full)
        else:
            self._display_response_basic(model_name, response, is_best, show_full)
    
    def _display_response_rich(self, model_name: str, response: str, 
                              is_best: bool, show_full: bool):
        """Display response using Rich with full content handling."""
        # Determine title and styling
        if is_best:
            title = f"🏆 BEST MODEL: {model_name}"
            border_style = "green"
        else:
            title = f"📋 {model_name}"
            border_style = "blue"
        
        # Handle long responses
        if len(response) > 3000 and not show_full:
            # Show preview with option to see full
            preview = response[:3000] + "\n\n[... Response continues ...]"
            panel = Panel(preview, title=f"{title} (Preview)", border_style=border_style)
            self.console.print(panel)
            
            if Confirm.ask(f"Show full response from {model_name}?", default=False):
                self._display_full_response_rich(model_name, response, is_best)
        else:
            # Show full response
            self._display_full_response_rich(model_name, response, is_best)
    
    def _display_full_response_rich(self, model_name: str, response: str, is_best: bool):
        """Display full response with pagination if needed."""
        if is_best:
            title = f"🏆 BEST MODEL: {model_name}"
            border_style = "green"
        else:
            title = f"📋 {model_name}"
            border_style = "blue"
        
        # For very long responses, implement simple pagination
        if len(response) > 5000:
            self._paginate_response_rich(title, response, border_style)
        else:
            panel = Panel(response, title=title, border_style=border_style)
            self.console.print(panel)
    
    def _paginate_response_rich(self, title: str, response: str, border_style: str):
        """Paginate very long responses."""
        lines = response.split('\n')
        page_size = 50  # lines per page
        total_pages = (len(lines) + page_size - 1) // page_size
        
        for page in range(total_pages):
            start_idx = page * page_size
            end_idx = min((page + 1) * page_size, len(lines))
            page_content = '\n'.join(lines[start_idx:end_idx])
            
            page_title = f"{title} (Page {page + 1}/{total_pages})"
            panel = Panel(page_content, title=page_title, border_style=border_style)
            self.console.print(panel)
            
            if page < total_pages - 1:  # Not the last page
                if not Confirm.ask("Continue to next page?", default=True):
                    self.console.print("[yellow]Response truncated by user.[/yellow]")
                    break
    
    def _display_response_basic(self, model_name: str, response: str, 
                               is_best: bool, show_full: bool):
        """Display response in basic terminal format."""
        marker = "🏆" if is_best else "📋"
        best_indicator = " (BEST MODEL)" if is_best else ""
        
        print(f"\n{marker} {model_name}{best_indicator}")
        print("=" * (len(model_name) + 10))
        
        if len(response) > 3000 and not show_full:
            print(response[:3000])
            print("\n[... Response continues ...]")
            choice = input(f"\nShow full response from {model_name}? (y/n): ").lower()
            if choice in ['y', 'yes']:
                self._display_full_response_basic(response)
        else:
            self._display_full_response_basic(response)
    
    def _display_full_response_basic(self, response: str):
        """Display full response with basic pagination."""
        if len(response) > 5000:
            lines = response.split('\n')
            page_size = 50
            total_pages = (len(lines) + page_size - 1) // page_size
            
            for page in range(total_pages):
                start_idx = page * page_size
                end_idx = min((page + 1) * page_size, len(lines))
                page_content = '\n'.join(lines[start_idx:end_idx])
                
                print(f"\n--- Page {page + 1}/{total_pages} ---")
                print(page_content)
                
                if page < total_pages - 1:
                    choice = input("\nContinue to next page? (y/n): ").lower()
                    if choice not in ['y', 'yes']:
                        print("Response truncated by user.")
                        break
        else:
            print(response)
    
    def display_multiple_responses(self, responses: Dict[str, str], 
                                 best_model: str = None,
                                 mode: str = "comprehensive"):
        """Display multiple model responses with various modes."""
        if mode == "best_only" and best_model and best_model in responses:
            self.display_response(best_model, responses[best_model], is_best=True)
            return
        
        if mode == "all":
            for model_name, response in responses.items():
                is_best = model_name == best_model
                self.display_response(model_name, response, is_best=is_best)
            return
        
        # Comprehensive mode (default)
        if best_model and best_model in responses:
            # Show best model first
            self.display_response(best_model, responses[best_model], is_best=True)
            
            # Ask if user wants to see others
            if len(responses) > 1:
                if self.use_rich:
                    show_others = Confirm.ask("Show responses from other models?", default=False)
                else:
                    choice = input("\nShow responses from other models? (y/n): ").lower()
                    show_others = choice in ['y', 'yes']
                
                if show_others:
                    if self.use_rich:
                        self.console.print("\n[bold cyan]OTHER MODELS' RESPONSES:[/bold cyan]")
                    else:
                        print(f"\n{'='*50}")
                        print("OTHER MODELS' RESPONSES")
                        print(f"{'='*50}")
                    
                    for model_name, response in responses.items():
                        if model_name != best_model:
                            self.display_response(model_name, response, is_best=False)
        else:
            # No best model specified, show all
            for model_name, response in responses.items():
                is_best = model_name == best_model
                self.display_response(model_name, response, is_best=is_best)
    
    def display_code_preview(self, code: str, language: str = "python"):
        """Display code with syntax highlighting if available."""
        if self.use_rich and len(code) < 2000:  # Only highlight shorter code
            try:
                syntax = Syntax(code, language, theme="monokai", line_numbers=True)
                panel = Panel(syntax, title="Code Preview", border_style="yellow")
                self.console.print(panel)
                return
            except Exception:
                pass  # Fall back to basic display
        
        # Basic display
        print("\nCode Preview:")
        print("-" * 40)
        lines = code.split('\n')
        for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
            print(f"{i:2d}| {line}")
        if len(lines) > 20:
            print(f"... ({len(lines) - 20} more lines)")
    
    def display_query_info(self, query: str, has_code: bool = False):
        """Display query information and analysis."""
        if self.use_rich:
            self.console.print(f"\n[bold blue]Processing Query...[/bold blue]")
            self.console.print(f"[dim]Length: {len(query)} characters[/dim]")
            if has_code:
                self.console.print(f"[dim]🔍 Code detected in input[/dim]")
        else:
            print(f"\nProcessing Query...")
            print(f"Length: {len(query)} characters")
            if has_code:
                print("🔍 Code detected in input")
    
    def display_system_status(self, stats: Dict[str, Any]):
        """Display system status and statistics."""
        if self.use_rich:
            status_text = Text()
            status_text.append("System Status: ", style="bold")
            status_text.append(stats.get('status', 'unknown').upper(), 
                             style="green" if stats.get('status') == 'ready' else "red")
            
            if 'best_model' in stats and stats['best_model']:
                status_text.append(f"\nBest Model: {stats['best_model']}", style="yellow")
            
            if 'vector_store' in stats:
                vs_stats = stats['vector_store']
                if 'points_count' in vs_stats:
                    status_text.append(f"\nIndexed Chunks: {vs_stats['points_count']}", style="cyan")
            
            panel = Panel(status_text, title="System Information", border_style="blue")
            self.console.print(panel)
        else:
            print(f"\nSystem Status: {stats.get('status', 'unknown').upper()}")
            if 'best_model' in stats and stats['best_model']:
                print(f"Best Model: {stats['best_model']}")
            if 'vector_store' in stats and 'points_count' in stats['vector_store']:
                print(f"Indexed Chunks: {stats['vector_store']['points_count']}")
    
    def display_help(self):
        """Display comprehensive help information."""
        help_content = """
Available Commands:
• quit/exit - Exit interactive mode
• stats - Show system statistics  
• health - Perform system health check
• best - Show only best model responses
• all - Show all models' responses
• pdf-status - Show PDF processing status and statistics
• reprocess-pdf - Instructions for reprocessing PDFs
• help - Display this help menu

Input Options:
• Paste Python code or ask natural language questions
• Use ###END### or CTRL+D to submit multi-line input
• Double ENTER also works for short queries

Response Modes:
• Comprehensive (default) - Shows best model first, option to see others
• Best only - Shows only the optimized best model
• All models - Shows responses from all available models

PDF Management:
• The system intelligently skips PDFs that have already been processed
• Use 'pdf-status' to see which PDFs are indexed and when
• To force reprocess PDFs, restart with: python main.py --force-reindex-pdfs
• To check PDF status from command line: python main.py --pdf-status

Tips:
• Provide code context for better suggestions
• Be specific about what you want to improve
• Ask about specific refactoring patterns
• PDFs are automatically skipped if already processed (saves time!)
"""
        
        if self.use_rich:
            panel = Panel(help_content.strip(), title="Help", border_style="cyan")
            self.console.print(panel)
        else:
            print(f"\n{'='*60}")
            print("HELP")
            print(f"{'='*60}")
            print(help_content.strip())
    
    def display_error(self, error_msg: str):
        """Display error message with appropriate formatting."""
        if self.use_rich:
            self.console.print(f"[bold red]Error:[/bold red] {error_msg}")
        else:
            print(f"Error: {error_msg}")
    
    def display_warning(self, warning_msg: str):
        """Display warning message with appropriate formatting."""
        if self.use_rich:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] {warning_msg}")
        else:
            print(f"Warning: {warning_msg}")
    
    def display_success(self, success_msg: str):
        """Display success message with appropriate formatting."""
        if self.use_rich:
            self.console.print(f"[bold green]Success:[/bold green] {success_msg}")
        else:
            print(f"Success: {success_msg}")


class InteractiveSession:
    """Manages an interactive session with enhanced display capabilities."""
    
    def __init__(self, system, best_model: str = None):
        self.system = system
        self.best_model = best_model
        self.display_manager = ResponseDisplayManager()
        self.session_count = 0
        self.response_mode = "comprehensive"  # comprehensive, best_only, all
        self.should_exit = False  # Flag to control session exit
    
    def run(self):
        """Run the interactive session."""
        self._display_welcome()
        
        while not self.should_exit:
            try:
                self.session_count += 1
                self._display_session_header()
                
                query = self._get_user_input()
                if not query:
                    continue
                
                # Handle commands - check if we should exit
                if self._handle_command(query):
                    if self.should_exit:
                        break
                    continue
                
                # Process query
                self._process_query(query)
                
            except KeyboardInterrupt:
                self.display_manager.display_warning("Session interrupted by user")
                break
            except Exception as e:
                self.display_manager.display_error(f"Unexpected error: {e}")
        
        # Final goodbye message if not already shown
        if not self.should_exit:
            self.display_manager.display_success("Session ended. Goodbye!")
    
    def _display_welcome(self):
        """Display welcome message and instructions."""
        if self.display_manager.use_rich:
            welcome_text = f"""[bold blue]Python Code Refactoring RAG System[/bold blue]
[yellow]Enhanced Interactive Mode[/yellow]
[green]Optimized Best Model: {self.best_model or 'Not set'}[/green]"""
            
            panel = Panel.fit(welcome_text, title="Welcome")
            self.display_manager.console.print(panel)
        else:
            print(f"\n{'='*70}")
            print("Python Code Refactoring RAG System - Enhanced Interactive Mode")
            print(f"{'='*70}")
            print(f"Optimized Best Model: {self.best_model or 'Not set'}")
        
        self.display_manager.display_help()
    
    def _display_session_header(self):
        """Display session header."""
        if self.display_manager.use_rich:
            self.display_manager.console.print(f"\n[bold magenta]Session {self.session_count}[/bold magenta] "
                                             f"[dim](Mode: {self.response_mode})[/dim]")
        else:
            print(f"\n{'='*20} Session {self.session_count} {'='*20}")
            print(f"Mode: {self.response_mode}")
    
    def _get_user_input(self) -> str:
        """Get user input with multi-line support."""
        if self.display_manager.use_rich:
            return self._get_input_rich()
        else:
            return self._get_input_basic()
    
    def _get_input_rich(self) -> str:
        """Get input using Rich interface."""
        lines = []
        line_num = 1
        empty_count = 0
        
        self.display_manager.console.print("\n[bold yellow]Enter your query or code:[/bold yellow]")
        
        while True:
            try:
                prompt_text = f"[dim]{line_num:2d}|[/dim] "
                line = Prompt.ask(prompt_text, default="")
                
                if line.strip() == '###END###':
                    break
                
                if not line.strip():
                    empty_count += 1
                    if empty_count >= 2:
                        self.display_manager.console.print("[dim]Double empty line detected. Submitting...[/dim]")
                        break
                else:
                    empty_count = 0
                
                lines.append(line)
                line_num += 1
                
            except EOFError:
                self.display_manager.console.print("\n[dim]EOF detected. Submitting...[/dim]")
                break
        
        return '\n'.join(lines).strip()
    
    def _get_input_basic(self) -> str:
        """Get input using basic interface."""
        print("\nEnter your query or code (double ENTER to submit):")
        lines = []
        empty_count = 0
        
        try:
            while True:
                line = input(f"{len(lines)+1:2d}| ")
                
                if line.strip() == '###END###':
                    break
                
                if not line.strip():
                    empty_count += 1
                    if empty_count >= 2:
                        print("Double empty line detected. Submitting...")
                        break
                else:
                    empty_count = 0
                
                lines.append(line)
        
        except EOFError:
            print("\nEOF detected. Submitting...")
        
        return '\n'.join(lines).strip()
    
    def _handle_command(self, query: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = query.lower().strip()
        
        if command in ['quit', 'exit']:
            self.display_manager.display_success("Goodbye!")
            self.should_exit = True  # Set exit flag
            return True
        
        elif command == 'stats':
            stats = self.system.get_system_stats()
            self.display_manager.display_system_status(stats)
            return True
        
        elif command == 'health':
            health = self.system.health_check()
            self._display_health_check(health)
            return True
        
        elif command == 'best':
            self.response_mode = "best_only"
            self.display_manager.display_success("Mode set to: Best model only")
            return True
        
        elif command == 'all':
            self.response_mode = "all"
            self.display_manager.display_success("Mode set to: All models")
            return True
        
        elif command == 'pdf-status':
            self._display_pdf_status()
            return True
        
        elif command.startswith('reprocess-pdf'):
            # Handle PDF reprocessing command
            self._handle_pdf_reprocessing()
            return True
        
        elif command == 'help':
            self.display_manager.display_help()
            return True
        
        return False
    
    def _display_pdf_status(self):
        """Display PDF processing status."""
        try:
            pdf_summary = self.system.get_pdf_processing_summary()
            
            if self.display_manager.use_rich:
                from rich.table import Table
                
                table = Table(title="PDF Processing Status")
                table.add_column("PDF File", style="cyan")
                table.add_column("Total Chunks", justify="right")
                table.add_column("Text Chunks", justify="right")
                table.add_column("Code Chunks", justify="right")
                table.add_column("Processed Date", style="dim")
                
                if pdf_summary['pdf_files']:
                    for pdf_file, stats in pdf_summary['pdf_files'].items():
                        processed_date = pdf_summary['processing_dates'].get(pdf_file, 'Unknown')
                        table.add_row(
                            pdf_file,
                            str(stats['total_chunks']),
                            str(stats['text_chunks']),
                            str(stats['code_chunks']),
                            processed_date
                        )
                    
                    self.display_manager.console.print(table)
                    self.display_manager.console.print(f"\n[bold]Total PDF chunks: {pdf_summary['total_chunks']}[/bold]")
                else:
                    self.display_manager.console.print("[yellow]No PDF files have been processed yet.[/yellow]")
            else:
                print("\nPDF Processing Status:")
                print("-" * 60)
                
                if pdf_summary['pdf_files']:
                    print(f"Total PDF chunks: {pdf_summary['total_chunks']}")
                    print(f"Processed PDF files: {len(pdf_summary['pdf_files'])}\n")
                    
                    for pdf_file, stats in pdf_summary['pdf_files'].items():
                        processed_date = pdf_summary['processing_dates'].get(pdf_file, 'Unknown')
                        print(f"📄 {pdf_file}")
                        print(f"   Total chunks: {stats['total_chunks']}")
                        print(f"   Text chunks: {stats['text_chunks']}")
                        print(f"   Code chunks: {stats['code_chunks']}")
                        print(f"   Processed: {processed_date}")
                        print()
                else:
                    print("No PDF files have been processed yet.")
                    
        except Exception as e:
            self.display_manager.display_error(f"Failed to get PDF status: {e}")
    
    def _handle_pdf_reprocessing(self):
        """Handle PDF reprocessing request."""
        try:
            if self.display_manager.use_rich:
                from rich.prompt import Confirm
                should_reprocess = Confirm.ask("⚠️  This will reprocess all PDFs and regenerate embeddings. Continue?")
            else:
                choice = input("⚠️  This will reprocess all PDFs and regenerate embeddings. Continue? (y/n): ").lower()
                should_reprocess = choice in ['y', 'yes']
            
            if should_reprocess:
                self.display_manager.display_warning("Reprocessing PDFs... This may take several minutes.")
                
                # Get the PDF paths from the system configuration (you may need to store these)
                # For now, we'll just show a message about how to do it
                if self.display_manager.use_rich:
                    self.display_manager.console.print("""
[yellow]To reprocess PDFs, please restart the system with:[/yellow]
[cyan]python main.py --force-reindex-pdfs[/cyan]

[dim]Or to check current PDF status:[/dim]
[cyan]python main.py --pdf-status[/cyan]
""")
                else:
                    print("""
To reprocess PDFs, please restart the system with:
  python main.py --force-reindex-pdfs

Or to check current PDF status:
  python main.py --pdf-status
""")
            else:
                self.display_manager.display_success("PDF reprocessing cancelled.")
                
        except Exception as e:
            self.display_manager.display_error(f"Error handling PDF reprocessing: {e}")
    
    def _display_health_check(self, health: Dict[str, bool]):
        """Display health check results."""
        if self.display_manager.use_rich:
            health_text = Text()
            for component, status in health.items():
                status_text = "HEALTHY" if status else "UNHEALTHY"
                color = "green" if status else "red"
                health_text.append(f"{component}: ", style="white")
                health_text.append(f"{status_text}\n", style=color)
            
            panel = Panel(health_text, title="Health Check Results", border_style="blue")
            self.display_manager.console.print(panel)
        else:
            print("\nHealth Check Results:")
            print("-" * 30)
            for component, status in health.items():
                status_text = "HEALTHY" if status else "UNHEALTHY"
                print(f"{component}: {status_text}")
    
    def _process_query(self, query: str):
        """Process a user query and display results."""
        # Analyze query
        has_code = any(kw in query.lower() for kw in ['def ', 'class ', 'import ', 'for ', 'if ', 'while '])
        self.display_manager.display_query_info(query, has_code)
        
        # Show code preview if detected
        if has_code:
            self.display_manager.display_code_preview(query)
        
        user_code = query if has_code else None
        
        try:
            # Get responses based on mode
            if self.response_mode == "best_only" and self.best_model:
                response = self.system.get_best_model_suggestion(query, user_code=user_code)
                self.display_manager.display_response(self.best_model, response, is_best=True)
            
            elif self.response_mode == "all":
                responses = self.system.get_all_model_suggestions(query, user_code=user_code)
                self.display_manager.display_multiple_responses(responses, self.best_model, mode="all")
            
            else:  # comprehensive mode
                responses = self.system.get_all_model_suggestions(query, user_code=user_code)
                self.display_manager.display_multiple_responses(responses, self.best_model, mode="comprehensive")
            
            self.display_manager.display_success("Query processed successfully")
            
        except Exception as e:
            self.display_manager.display_error(f"Failed to process query: {e}")


def create_interactive_session(system, best_model: str = None) -> InteractiveSession:
    """Factory function to create an interactive session."""
    return InteractiveSession(system, best_model)