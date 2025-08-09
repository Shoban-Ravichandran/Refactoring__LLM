"""RAG system evaluation with comprehensive metrics including CodeBLEU."""

import logging
import re
import time
import ast
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np

from config.settings import EvaluationConfig

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK library not found. Install 'nltk' for CodeBLEU metrics.")
    NLTK_AVAILABLE = False


@dataclass
class RAGEvaluationMetrics:
    """Metrics for evaluating RAG system performance."""
    answer_relevance: float
    faithfulness: float
    response_completeness: float
    retrieval_quality: float
    latency_ms: float
    codebleu_score: float = 0.0
    token_usage: Dict[str, int] = None  # {'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z}

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}


class CodeBLEUEvaluator:
    """CodeBLEU evaluator with AST-aware scoring for code snippets."""
    
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1 if NLTK_AVAILABLE else None
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text."""
        # Extract code blocks with triple backticks
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
        
        # Extract inline code with single backticks
        inline_code = re.findall(r'`([^`\n]+)`', text)
        
        # Combine all code snippets
        all_code = code_blocks + [code for code in inline_code if any(keyword in code for keyword in ['def ', 'class ', 'import ', 'for ', 'if ', 'while '])]
        
        return [code.strip() for code in all_code if code.strip()]
    
    def parse_ast_safely(self, code: str) -> Optional[ast.AST]:
        """Safely parse code into AST."""
        try:
            # Clean up common formatting issues
            code = code.strip()
            if not code:
                return None
            
            # Handle incomplete code snippets
            if not any(code.startswith(keyword) for keyword in ['def ', 'class ', 'import ', 'from ']):
                # Wrap in a function if it's just statements
                code = f"def temp_function():\n    {code.replace(chr(10), chr(10) + '    ')}"
            
            return ast.parse(code)
        except SyntaxError:
            # Try to fix common issues
            try:
                # Add missing indentation
                lines = code.split('\n')
                if lines and not lines[0].startswith((' ', '\t')):
                    indented_code = '\n'.join('    ' + line if line.strip() else line for line in lines)
                    code = f"def temp_function():\n{indented_code}"
                    return ast.parse(code)
            except:
                pass
            return None
        except Exception:
            return None
    
    def extract_ast_features(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract AST features for comparison."""
        features = {
            'node_types': set(),
            'function_names': set(),
            'variable_names': set(),
            'attribute_names': set(),
            'literals': set()
        }
        
        for node in ast.walk(tree):
            # Node types
            features['node_types'].add(type(node).__name__)
            
            # Function and method names
            if isinstance(node, ast.FunctionDef):
                features['function_names'].add(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                features['function_names'].add(node.func.id)
            
            # Variable names
            if isinstance(node, ast.Name):
                features['variable_names'].add(node.id)
            
            # Attribute names
            if isinstance(node, ast.Attribute):
                features['attribute_names'].add(node.attr)
            
            # Literals
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (str, int, float)):
                    features['literals'].add(str(node.value))
        
        return features
    
    def calculate_ast_similarity(self, ref_features: Dict[str, Set[str]], 
                                cand_features: Dict[str, Set[str]]) -> float:
        """Calculate AST-based similarity between reference and candidate features."""
        similarities = []
        
        for feature_type in ref_features:
            ref_set = ref_features[feature_type]
            cand_set = cand_features[feature_type]
            
            if not ref_set and not cand_set:
                similarity = 1.0
            elif not ref_set or not cand_set:
                similarity = 0.0
            else:
                intersection = len(ref_set.intersection(cand_set))
                union = len(ref_set.union(cand_set))
                similarity = intersection / union if union > 0 else 0.0
            
            # Weight different features
            weights = {
                'node_types': 0.3,
                'function_names': 0.25,
                'variable_names': 0.2,
                'attribute_names': 0.15,
                'literals': 0.1
            }
            
            similarities.append(similarity * weights.get(feature_type, 0.2))
        
        return sum(similarities)
    
    def calculate_token_bleu(self, reference: str, candidate: str) -> float:
        """Calculate token-level BLEU score."""
        if not NLTK_AVAILABLE:
            return 0.0
        
        try:
            # Tokenize by preserving code structure
            ref_tokens = self.tokenize_code_aware(reference)
            cand_tokens = self.tokenize_code_aware(candidate)
            
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            # Use modified weights for code context
            weights = (0.4, 0.3, 0.2, 0.1)
            score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, 
                                smoothing_function=self.smoothing_function)
            return score
        except Exception as e:
            logger.warning(f"Error calculating token BLEU: {e}")
            return 0.0
    
    def tokenize_code_aware(self, text: str) -> List[str]:
        """Tokenize text while preserving code structure."""
        # First, protect code blocks
        code_blocks = self.extract_code_blocks(text)
        
        # Replace code blocks with placeholders
        protected_text = text
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            protected_text = protected_text.replace(f"```python\n{block}\n```", placeholder)
            protected_text = protected_text.replace(f"```\n{block}\n```", placeholder)
            protected_text = protected_text.replace(f"`{block}`", placeholder)
        
        # Tokenize the protected text
        import string
        # Remove punctuation except underscores and dots (for code)
        translator = str.maketrans('', '', string.punctuation.replace('_', '').replace('.', ''))
        clean_text = protected_text.translate(translator)
        
        tokens = clean_text.lower().split()
        
        # Add back code tokens
        for i, block in enumerate(code_blocks):
            placeholder = f"__code_block_{i}__"
            if placeholder in tokens:
                # Tokenize the code block separately
                code_tokens = self.tokenize_code_block(block)
                # Replace placeholder with code tokens
                idx = tokens.index(placeholder)
                tokens = tokens[:idx] + code_tokens + tokens[idx+1:]
        
        return tokens
    
    def tokenize_code_block(self, code: str) -> List[str]:
        """Tokenize a code block preserving semantic elements."""
        # Split on common code delimiters while preserving semantic meaning
        import re
        
        # Split on whitespace and common delimiters
        tokens = re.findall(r'\w+|[(){}[\].,;:=+\-*/<>!&|]', code)
        
        # Convert to lowercase for comparison, but preserve case-sensitive keywords
        processed_tokens = []
        for token in tokens:
            if token.lower() in ['def', 'class', 'if', 'else', 'for', 'while', 'try', 'except', 'import', 'from', 'return']:
                processed_tokens.append(token.lower())
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def calculate_codebleu(self, reference: str, candidate: str) -> float:
        """Calculate CodeBLEU score combining token-level and AST-level similarities."""
        # Extract code blocks from both texts
        ref_code_blocks = self.extract_code_blocks(reference)
        cand_code_blocks = self.extract_code_blocks(candidate)
        
        # If no code blocks in either, fall back to token BLEU
        if not ref_code_blocks and not cand_code_blocks:
            return self.calculate_token_bleu(reference, candidate)
        
        # If only one has code, penalize but still calculate token similarity
        if not ref_code_blocks or not cand_code_blocks:
            token_score = self.calculate_token_bleu(reference, candidate)
            return token_score * 0.5  # Penalty for missing code
        
        # Calculate token-level BLEU
        token_bleu = self.calculate_token_bleu(reference, candidate)
        
        # Calculate AST-level similarity
        ast_similarities = []
        
        # Compare each reference code block with best matching candidate block
        for ref_block in ref_code_blocks:
            ref_ast = self.parse_ast_safely(ref_block)
            if ref_ast is None:
                continue
            
            ref_features = self.extract_ast_features(ref_ast)
            
            best_ast_sim = 0.0
            for cand_block in cand_code_blocks:
                cand_ast = self.parse_ast_safely(cand_block)
                if cand_ast is None:
                    continue
                
                cand_features = self.extract_ast_features(cand_ast)
                ast_sim = self.calculate_ast_similarity(ref_features, cand_features)
                best_ast_sim = max(best_ast_sim, ast_sim)
            
            ast_similarities.append(best_ast_sim)
        
        # Average AST similarity
        avg_ast_similarity = np.mean(ast_similarities) if ast_similarities else 0.0
        
        # Combine token BLEU and AST similarity
        # Give more weight to AST similarity for code-heavy content
        code_ratio = len(''.join(ref_code_blocks)) / max(len(reference), 1)
        ast_weight = min(0.6, 0.3 + code_ratio * 0.3)  # 30-60% weight for AST
        token_weight = 1.0 - ast_weight
        
        codebleu_score = token_weight * token_bleu + ast_weight * avg_ast_similarity
        
        return min(1.0, codebleu_score)


class ContextRelevanceEvaluator:
    """Enhanced evaluator with improved context relevance scoring and diagnostics."""
    
    def __init__(self):
        # Core refactoring concepts
        self.refactoring_concepts = {
            'complexity_reduction': {
                'keywords': ['complex', 'nested', 'simplify', 'extract method', 'break down',
                           'cyclomatic', 'complicated', 'reduce complexity', 'too complex'],
                'weight': 1.0
            },
            'readability': {
                'keywords': ['readable', 'clear', 'naming', 'variable names', 'clean',
                           'understand', 'clarity', 'readability', 'understandable'],
                'weight': 1.0
            },
            'structure': {
                'keywords': ['refactor', 'restructure', 'organize', 'extract', 'split',
                           'separate', 'structure', 'method extraction'],
                'weight': 1.0
            },
            'performance': {
                'keywords': ['performance', 'optimize', 'efficient', 'speed', 'faster'],
                'weight': 0.8
            }
        }
        
        # Code-specific terms that boost relevance
        self.code_indicators = [
            'function', 'method', 'class', 'def', 'python', 'code', 'loop', 'if', 'for'
        ]
        
        # Initialize CodeBLEU evaluator
        self.codebleu_evaluator = CodeBLEUEvaluator()

    def evaluate_context_relevance(self, query: str, context_chunks: List[Dict]) -> float:
        """Enhanced context relevance calculation with debugging."""
        if not context_chunks:
            logger.warning("No context chunks provided for relevance evaluation")
            return 0.0
            
        query_lower = query.lower().strip()
        if len(query_lower) < 3:
            return 0.1
            
        # Extract meaningful query words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                     'for', 'of', 'with', 'by', 'this', 'that', 'can', 'i', 'how', 
                     'what', 'is', 'are', 'my'}
        query_words = set(
            word.strip('.,!?()[]{}')
            for word in query_lower.split()
            if len(word) > 2 and word not in stop_words
        )
        
        total_relevance = 0.0
        chunk_scores = []
        
        for i, chunk in enumerate(context_chunks):
            chunk_relevance = 0.0
            content = chunk.get('text', chunk.get('code', '')).lower()
            metadata = chunk.get('metadata', {})
            
            # 1. Word Overlap (50% weight)
            if query_words:
                content_words = set(word.strip('.,!?()[]{}') for word in content.split() if len(word) > 2)
                direct_overlap = len(query_words.intersection(content_words)) / len(query_words)
                
                # Also check for partial word matches
                partial_matches = 0
                for qword in query_words:
                    if len(qword) > 4:  # Only for longer words
                        for cword in content_words:
                            if qword in cword or cword in qword:
                                partial_matches += 1
                                break
                partial_score = partial_matches / len(query_words)
                word_score = max(direct_overlap, partial_score * 0.6)
                chunk_relevance += 0.5 * word_score
            
            # 2. Concept Matching (30% weight)
            concept_score = 0.0
            query_concepts = []
            
            # Find query concepts
            for concept, data in self.refactoring_concepts.items():
                if any(keyword in query_lower for keyword in data['keywords']):
                    query_concepts.append(concept)
            
            # Score concept matches in content
            for concept in query_concepts:
                keywords = self.refactoring_concepts[concept]['keywords']
                content_matches = sum(1 for kw in keywords if kw in content)
                if content_matches > 0:
                    concept_score += 0.4  # More generous scoring per concept
            
            chunk_relevance += 0.3 * min(concept_score, 1.0)
            
            # 3. Metadata Boost (20% weight)
            metadata_score = 0.0
            
            # Type matching
            if 'function' in query_lower and metadata.get('type') == 'function':
                metadata_score += 0.5
            elif 'class' in query_lower and metadata.get('type') == 'class':
                metadata_score += 0.5
            elif metadata.get('type') in ['function', 'class', 'method']:
                metadata_score += 0.3  # Any code structure is relevant
            
            # Version matching - VERY IMPORTANT
            if any(word in query_lower for word in ['refactor', 'improve', 'better', 'fix']):
                if metadata.get('version') == 'refactored':
                    metadata_score += 0.7  # Big boost for refactored examples
                elif metadata.get('version') == 'original':
                    metadata_score += 0.4  # Some boost for "before" examples
            
            # Complexity relevance
            if any(word in query_lower for word in ['complex', 'nested', 'simplify']):
                complexity = metadata.get('complexity', {}).get('cyclomatic_complexity', 0)
                if complexity > 5:
                    metadata_score += 0.5
                elif complexity > 0:
                    metadata_score += 0.3
            
            chunk_relevance += 0.2 * min(metadata_score, 1.0)
            
            chunk_scores.append({
                'chunk_index': i,
                'relevance': chunk_relevance,
                'metadata': metadata,
                'content_preview': content[:100]
            })
            total_relevance += chunk_relevance
        
        # Calculate average and apply scaling
        avg_relevance = total_relevance / len(context_chunks)
        
        # Enhanced scaling - more generous and realistic
        if avg_relevance > 0.6:
            final_relevance = min(avg_relevance * 1.1, 1.0)
        elif avg_relevance > 0.4:
            final_relevance = avg_relevance * 1.3
        elif avg_relevance > 0.2:
            final_relevance = avg_relevance * 1.6
        else:
            final_relevance = max(avg_relevance * 2.0, 0.1)
        
        final_relevance = min(final_relevance, 1.0)
        
        logger.debug(f"Context relevance for query '{query[:50]}...': {final_relevance:.3f}")
        
        return final_relevance

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Enhanced answer relevance evaluation."""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Enhanced word matching
        query_words = [word for word in query_lower.split() if len(word) > 3]
        direct_matches = sum(1 for word in query_words if word in answer_lower)
        direct_score = direct_matches / max(1, len(query_words))
        
        # Enhanced concept matching
        concept_score = 0
        for category, data in self.refactoring_concepts.items():
            if any(keyword in query_lower for keyword in data['keywords']):
                category_coverage = sum(1 for keyword in data['keywords'] if keyword in answer_lower)
                concept_score += category_coverage / len(data['keywords'])
        
        concept_score = min(1.0, concept_score)
        
        # Check for code examples in answer if query is about code
        code_bonus = 0
        if any(word in query_lower for word in ['code', 'function', 'method', 'refactor']):
            if '```' in answer or 'def ' in answer or 'class ' in answer:
                code_bonus = 0.1
        
        return min(1.0, (direct_score + concept_score) / 2 + code_bonus)

    def evaluate_faithfulness(self, context_chunks: List[Dict], answer: str) -> float:
        """Enhanced faithfulness evaluation."""
        if not context_chunks:
            return 0.0
            
        context_text = ' '.join([chunk.get('text', chunk.get('code', '')) for chunk in context_chunks])
        context_lower = context_text.lower()
        answer_lower = answer.lower()
        
        # Split answer into meaningful statements
        statements = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
        if not statements:
            return 0.0
        
        supported_statements = 0
        for statement in statements:
            statement_words = [word for word in statement.split() if len(word) > 4]
            if statement_words:
                # Check both word overlap and semantic concepts
                word_support_ratio = sum(1 for word in statement_words if word in context_lower) / len(statement_words)
                
                # Check if statement references specific examples
                references_examples = any(ref in statement.lower() for ref in ['example', 'shown', 'demonstrates', 'above'])
                
                if word_support_ratio > 0.3 or references_examples:
                    supported_statements += 1
        
        return supported_statements / max(1, len(statements))

    def evaluate_response_completeness(self, query: str, answer: str) -> float:
        """Enhanced completeness evaluation."""
        completeness_indicators = [
            'what to change', 'why', 'benefits', 'risks', 'before', 'after',
            'example', 'code', 'implementation', 'suggestion', 'refactor',
            'how to', 'steps', 'approach', 'technique'
        ]
        
        answer_lower = answer.lower()
        present_indicators = sum(1 for indicator in completeness_indicators if indicator in answer_lower)
        
        # Bonus for code examples
        has_code_examples = bool(re.search(r'```|`[^`]+`', answer))
        if has_code_examples:
            present_indicators += 2
        
        # Bonus for structured response
        has_structure = bool(re.search(r'\*\*.*\*\*|#|1\.|2\.|•|-\s', answer))
        if has_structure:
            present_indicators += 1
        
        # Bonus for explanations
        has_explanations = any(word in answer_lower for word in ['because', 'since', 'reason', 'improve'])
        if has_explanations:
            present_indicators += 1
        
        return min(1.0, present_indicators / len(completeness_indicators))

    def evaluate_retrieval_quality(self, retrieved_chunks: List[Dict]) -> float:
        """Enhanced retrieval quality evaluation."""
        if not retrieved_chunks:
            return 0.0
        scores = [chunk.get('score', 0) for chunk in retrieved_chunks]
        return np.mean(scores) if scores else 0.0

    def comprehensive_evaluation(self, query: str, context_chunks: List[Dict],
                           answer: str, latency_ms: float = 0,
                           reference_answer: str = None,
                           token_usage: Dict[str, int] = None) -> RAGEvaluationMetrics:
        """Complete evaluation method with enhanced metrics including CodeBLEU."""
        answer_relevance = self.evaluate_answer_relevance(query, answer)
        faithfulness = self.evaluate_faithfulness(context_chunks, answer)
        completeness = self.evaluate_response_completeness(query, answer)
        retrieval_quality = self.evaluate_retrieval_quality(context_chunks)

        # Calculate CodeBLEU if reference answer is provided
        codebleu_score = 0.0
        if reference_answer:
            codebleu_score = self.codebleu_evaluator.calculate_codebleu(reference_answer, answer)

        return RAGEvaluationMetrics(
            answer_relevance=answer_relevance,
            faithfulness=faithfulness,
            response_completeness=completeness,
            retrieval_quality=retrieval_quality,
            latency_ms=latency_ms,
            codebleu_score=codebleu_score,
            token_usage=token_usage or {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        )


class RAGEvaluator:
    """Main RAG system evaluator."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.context_evaluator = ContextRelevanceEvaluator()
    
    def evaluate_single_query(self, query: str, context_chunks: List[Dict],
                         answer: str, latency_ms: float = 0,
                         reference_answer: str = None,
                         token_usage: Dict[str, int] = None) -> Dict[str, Any]:
        """Evaluate a single query-answer pair."""
        start_time = time.time()
        
        try:
            rag_metrics = self.context_evaluator.comprehensive_evaluation(
                query, context_chunks, answer, latency_ms, reference_answer, token_usage
            )
            
            evaluation_time = (time.time() - start_time) * 1000
            
            return {
                'query': query,
                'reference_answer': reference_answer,
                'answer': answer,
                'rag_metrics': rag_metrics,
                'evaluation_time_ms': evaluation_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in single query evaluation: {e}")
            return {
                'query': query,
                'reference_answer': reference_answer,
                'answer': answer,
                'rag_metrics': None,
                'evaluation_time_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_multiple_queries(self, test_cases: List[Dict]) -> List[Dict]:
        """Evaluate multiple test cases."""
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            result = self.evaluate_single_query(
                query=test_case['query'],
                context_chunks=test_case.get('context_chunks', []),
                answer=test_case['answer'],
                latency_ms=test_case.get('latency_ms', 0),
                reference_answer=test_case.get('reference_answer')
            )
            
            results.append(result)
        
        return results
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics across multiple evaluations."""
        successful_results = [r for r in results if r['success'] and r['rag_metrics']]
        
        if not successful_results:
            return {}
        
        metrics = {
            'answer_relevance': np.mean([r['rag_metrics'].answer_relevance for r in successful_results]),
            'faithfulness': np.mean([r['rag_metrics'].faithfulness for r in successful_results]),
            'response_completeness': np.mean([r['rag_metrics'].response_completeness for r in successful_results]),
            'retrieval_quality': np.mean([r['rag_metrics'].retrieval_quality for r in successful_results]),
            'avg_latency_ms': np.mean([r['rag_metrics'].latency_ms for r in successful_results]),
            'codebleu_score': np.mean([r['rag_metrics'].codebleu_score for r in successful_results]),
            'success_rate': len(successful_results) / len(results)
        }
        
        return metrics