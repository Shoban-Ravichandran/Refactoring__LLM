"""LLM providers and multi-model generation system."""

import logging
import time
from typing import List, Dict, Any

from config.model_configs import LLMConfig
from config.settings import LLMProvider, RAGConfig

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    logger.warning("Groq client not available")
    GROQ_AVAILABLE = False


def create_grounded_prompt(query: str, context_chunks: List[Dict], user_code: str = None) -> str:
    """Create a grounded prompt that encourages faithfulness to context."""
    
    # Organize context by refactoring type
    organized_context = {}
    for chunk in context_chunks[:3]:  # Limit to top 3
        ref_type = chunk['metadata'].get('refactoring_type', 'general')
        if ref_type not in organized_context:
            organized_context[ref_type] = []
        organized_context[ref_type].append(chunk)
    
    context_sections = []
    for ref_type, chunks in organized_context.items():
        section = f"### {ref_type.replace('_', ' ').title()} Examples:\n"
        for i, chunk in enumerate(chunks):
            metadata = chunk['metadata']
            code = chunk.get('code', chunk['text'])
            
            # Add clear example structure
            section += f"\n**Example {i+1}:**\n"
            section += f"- Type: {metadata.get('type', 'code')}\n"
            section += f"- Complexity: {metadata.get('complexity', {}).get('cyclomatic_complexity', 'N/A')}\n"
            section += f"- Status: {metadata.get('version', 'unknown')}\n"
            
            if len(code) > 200:
                code = code[:200] + "\n# ... (truncated)"
            section += f"```python\n{code}\n```\n"
        
        context_sections.append(section)
    
    context = '\n'.join(context_sections)
    
    # Enhanced prompt with explicit grounding instructions
    prompt = f"""You are a Python refactoring expert. Your task is to provide specific, actionable advice based ONLY on the examples provided below.

{context}

**User Query:** {query}

**CRITICAL INSTRUCTIONS:**
1. Base your response ONLY on the patterns and techniques shown in the examples above
2. Reference specific examples by number when making suggestions
3. If the examples don't contain relevant information for the query, explicitly state this
4. Provide concrete before/after code snippets that follow the patterns from the examples
5. Explain WHY each refactoring improves the code (based on what the examples demonstrate)

**Response Structure:**
1. **Analysis of Examples**: What refactoring patterns do the examples demonstrate?
2. **Application to Query**: How do these patterns apply to your specific question?
3. **Concrete Recommendations**: Step-by-step refactoring suggestions with code
4. **Expected Benefits**: What improvements should you expect (based on the examples)?

**Your Response:**"""

    # Add user code context if provided
    if user_code:
        prompt += f"\n\n**User's Code Context:**\n```python\n{user_code[:300]}\n```\n"
        prompt += "Please tailor your recommendations to this specific code."
    
    return prompt


class MultiLLMProvider:
    """Multi-LLM provider for generating refactoring suggestions."""
    
    def __init__(self, llm_configs: List[LLMConfig]):
        self.llm_configs = {config.model_name: config for config in llm_configs}
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients with error handling."""
        for model_name, config in self.llm_configs.items():
            try:
                if config.provider == LLMProvider.GROQ and GROQ_AVAILABLE:
                    client = Groq(api_key=config.api_key)
                    # Test the client
                    try:
                        test_response = client.chat.completions.create(
                            model=config.model_name,
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1,
                            temperature=0.1
                        )
                        self.clients[model_name] = client
                        logger.info(f"Successfully initialized Groq client for {model_name}")
                    except Exception as model_error:
                        logger.error(f"Model {model_name} is not available: {model_error}")
                        continue
                else:
                    logger.warning(f"Provider {config.provider} not supported or not available")
                    
            except Exception as e:
                logger.error(f"Failed to initialize client for {model_name}: {e}")
        
        if not self.clients:
            raise ValueError("No valid LLM clients were initialized.")

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.clients.keys())

    def generate_suggestion(self, query: str, context_chunks: List[Dict],
                          config: RAGConfig, model_name: str,
                          user_code: str = None) -> str:
        """Generate refactoring suggestion using specified model."""
        if model_name not in self.clients:
            raise ValueError(f"Model {model_name} not available")
        
        # Add delay to prevent rate limiting
        time.sleep(1)
        
        # Filter relevant chunks
        relevant_chunks = [
            chunk for chunk in context_chunks
            if chunk['score'] >= config.similarity_threshold
        ]
        
        if not relevant_chunks:
            return "I couldn't find relevant code examples in my knowledge base for this specific refactoring request."

        # Use the grounded prompt
        prompt = create_grounded_prompt(query, relevant_chunks, user_code)
        
        # Generate response
        llm_config = self.llm_configs[model_name]
        client = self.clients[model_name]
        
        if llm_config.provider == LLMProvider.GROQ:
            try:
                response = client.chat.completions.create(
                    model=llm_config.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Python refactoring expert. Always base your responses on the provided examples and reference them specifically."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.1,
                    timeout=30
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    logger.warning(f"Rate limit for {model_name}, waiting...")
                    time.sleep(15)
                    try:
                        response = client.chat.completions.create(
                            model=llm_config.model_name,
                            messages=[
                                {"role": "system", "content": "You are a Python refactoring expert."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=800,
                            temperature=0.1
                        )
                        return response.choices[0].message.content
                    except Exception as retry_e:
                        return f"Service temporarily unavailable: {retry_e}"
                else:
                    logger.error(f"Error with {model_name}: {e}")
                    return f"Error generating suggestion: {e}"
        
        return "Model provider not supported"

    def generate_suggestions_all_models(self, query: str, context_chunks: List[Dict],
                                      config: RAGConfig, user_code: str = None) -> Dict[str, str]:
        """Generate suggestions from all available models."""
        results = {}
        
        for model_name in self.clients.keys():
            try:
                suggestion = self.generate_suggestion(
                    query, context_chunks, config, model_name, user_code
                )
                results[model_name] = suggestion
            except Exception as e:
                logger.error(f"Error generating suggestion with {model_name}: {e}")
                results[model_name] = f"Error generating suggestion: {str(e)}"
        
        return results

    def test_model(self, model_name: str) -> Dict[str, Any]:
        """Test a specific model with a simple query."""
        if model_name not in self.clients:
            return {'success': False, 'error': f'Model {model_name} not available'}
        
        try:
            test_query = "How can I improve code readability?"
            mock_context = [{
                'text': 'def process_data(d): return [x for x in d if x > 0]',
                'metadata': {'type': 'function', 'version': 'refactored'},
                'score': 0.8
            }]
            
            suggestion = self.generate_suggestion(
                test_query, mock_context, RAGConfig(), model_name
            )
            
            return {
                'success': True,
                'model_name': model_name,
                'suggestion_length': len(suggestion),
                'suggestion_preview': suggestion[:100]
            }
            
        except Exception as e:
            return {
                'success': False,
                'model_name': model_name,
                'error': str(e)
            }

    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured models."""
        info = {}
        
        for model_name, config in self.llm_configs.items():
            info[model_name] = {
                'provider': config.provider.value,
                'max_tokens': config.max_tokens,
                'temperature': config.temperature,
                'timeout': config.timeout,
                'available': model_name in self.clients
            }
        
        return info