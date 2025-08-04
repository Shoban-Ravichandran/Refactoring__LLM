"""RAG system evaluation with comprehensive metrics."""

import logging
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from config.settings import EvaluationConfig

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ROUGE_AVAILABLE = True
    BLEU_AVAILABLE = True
except ImportError:
    logger.warning("BLEU/ROUGE libraries not found. Install 'rouge-score' and 'nltk' for these metrics.")
    ROUGE_AVAILABLE = False
    BLEU_AVAILABLE = False


@dataclass
class RAGEvaluationMetrics:
    """Metrics for evaluating RAG system performance."""
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    response_completeness: float
    retrieval_quality: float
    latency_ms: float
    bleu_score: float = 0.0
    rouge_l_score: float = 0.0


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

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Enhanced BLEU calculation with better preprocessing."""
        if not BLEU_AVAILABLE:
            return 0.0
        try:
            # Enhanced preprocessing
            ref_clean = self._preprocess_for_bleu(reference)
            cand_clean = self._preprocess_for_bleu(candidate)
            
            ref_tokens = ref_clean.lower().split()
            cand_tokens = cand_clean.lower().split()

            if not ref_tokens or not cand_tokens:
                return 0.0

            # Use smoothing to handle short sentences
            smoothing = SmoothingFunction().method1
            # Enhanced weights for code refactoring context
            score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.4, 0.3, 0.2, 0.1), smoothing_function=smoothing)
            return score
        except Exception as e:
            logger.warning(f"Error calculating BLEU: {e}")
            return 0.0

    def _preprocess_for_bleu(self, text: str) -> str:
        """Preprocess text for better BLEU calculation."""
        # Remove code blocks for fairer comparison
        text = re.sub(r'```.*?```', ' CODE_BLOCK ', text, flags=re.DOTALL)
        # Normalize common refactoring terms
        replacements = {
            'refactoring': 'refactor',
            'optimization': 'optimize',
            'simplification': 'simplify'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Enhanced ROUGE calculation."""
        if not ROUGE_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        try:
            # Preprocess both texts
            ref_clean = self._preprocess_for_bleu(reference)
            cand_clean = self._preprocess_for_bleu(candidate)
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(ref_clean, cand_clean)
            return {k: v.fmeasure for k, v in scores.items()}
        except Exception as e:
            logger.warning(f"Error calculating ROUGE: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def comprehensive_evaluation(self, query: str, context_chunks: List[Dict],
                               answer: str, latency_ms: float = 0,
                               reference_answer: str = None) -> RAGEvaluationMetrics:
        """Complete evaluation method with enhanced metrics."""
        context_relevance = self.evaluate_context_relevance(query, context_chunks)
        answer_relevance = self.evaluate_answer_relevance(query, answer)
        faithfulness = self.evaluate_faithfulness(context_chunks, answer)
        completeness = self.evaluate_response_completeness(query, answer)
        retrieval_quality = self.evaluate_retrieval_quality(context_chunks)

        # Calculate BLEU and ROUGE
        bleu_score = 0.0
        rouge_l_score = 0.0
        if reference_answer:
            bleu_score = self.calculate_bleu(reference_answer, answer)
            rouge_scores = self.calculate_rouge(reference_answer, answer)
            rouge_l_score = rouge_scores.get('rougeL', 0.0)

        return RAGEvaluationMetrics(
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            faithfulness=faithfulness,
            response_completeness=completeness,
            retrieval_quality=retrieval_quality,
            latency_ms=latency_ms,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score
        )


class RAGEvaluator:
    """Main RAG system evaluator."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.context_evaluator = ContextRelevanceEvaluator()
    
    def evaluate_single_query(self, query: str, context_chunks: List[Dict],
                             answer: str, latency_ms: float = 0,
                             reference_answer: str = None) -> Dict[str, Any]:
        """Evaluate a single query-answer pair."""
        start_time = time.time()
        
        try:
            rag_metrics = self.context_evaluator.comprehensive_evaluation(
                query, context_chunks, answer, latency_ms, reference_answer
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
            'context_relevance': np.mean([r['rag_metrics'].context_relevance for r in successful_results]),
            'answer_relevance': np.mean([r['rag_metrics'].answer_relevance for r in successful_results]),
            'faithfulness': np.mean([r['rag_metrics'].faithfulness for r in successful_results]),
            'response_completeness': np.mean([r['rag_metrics'].response_completeness for r in successful_results]),
            'retrieval_quality': np.mean([r['rag_metrics'].retrieval_quality for r in successful_results]),
            'avg_latency_ms': np.mean([r['rag_metrics'].latency_ms for r in successful_results]),
            'bleu_score': np.mean([r['rag_metrics'].bleu_score for r in successful_results]),
            'rouge_l_score': np.mean([r['rag_metrics'].rouge_l_score for r in successful_results]),
            'success_rate': len(successful_results) / len(results)
        }
        
        return metrics