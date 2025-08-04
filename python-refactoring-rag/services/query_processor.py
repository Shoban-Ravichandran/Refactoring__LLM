"""Query processing and enhancement for better retrieval."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Enhanced query processor with better semantic understanding."""
    
    def __init__(self):
        self.refactoring_patterns = {
            'complexity_reduction': {
                'keywords': ['complex', 'nested', 'cyclomatic', 'complicated', 'simplify'],
                'synonyms': ['intricate', 'convoluted', 'tangled', 'reduce complexity'],
                'code_patterns': ['if.*if', 'for.*for', 'while.*while', 'nested loops']
            },
            'readability': {
                'keywords': ['readable', 'clear', 'naming', 'understand', 'clean'],
                'synonyms': ['comprehensible', 'legible', 'understandable', 'intuitive'],
                'code_patterns': ['variable naming', 'function naming', 'comments']
            },
            'structure': {
                'keywords': ['refactor', 'extract', 'organize', 'split', 'separate'],
                'synonyms': ['restructure', 'reorganize', 'decompose', 'modularize'],
                'code_patterns': ['extract method', 'extract class', 'move method']
            },
            'performance': {
                'keywords': ['optimize', 'performance', 'efficient', 'speed', 'faster'],
                'synonyms': ['improve', 'accelerate', 'streamline', 'enhance'],
                'code_patterns': ['list comprehension', 'generator', 'caching']
            }
        }

    def enhance_query_with_context(self, query: str, code_context: str = None) -> str:
        """Enhanced query processing with semantic expansion."""
        enhanced_parts = ['Python code refactoring']
        
        # Add original query
        enhanced_parts.append(query)
        
        # Detect and expand refactoring intents
        query_lower = query.lower()
        detected_intents = []
        
        for intent, data in self.refactoring_patterns.items():
            if any(kw in query_lower for kw in data['keywords']):
                detected_intents.append(intent)
                # Add synonyms for better matching
                enhanced_parts.extend(data['synonyms'][:2])  # Add top 2 synonyms
                enhanced_parts.append(intent.replace('_', ' '))
        
        # Add code-specific context if provided
        if code_context:
            code_features = self._extract_code_features(code_context)
            enhanced_parts.extend(code_features)
        
        # Add refactoring context if it's clearly a refactoring query
        refactoring_indicators = ['how to', 'how can', 'refactor', 'improve', 'better', 'fix', 'optimize']
        if any(indicator in query_lower for indicator in refactoring_indicators):
            if 'refactor' not in ' '.join(enhanced_parts).lower():
                enhanced_parts.append('refactoring techniques')
        
        enhanced_query = ' '.join(enhanced_parts)
        logger.debug(f"Query enhanced: '{query}' -> '{enhanced_query}'")
        return enhanced_query

    def _extract_code_features(self, code: str) -> List[str]:
        """Extract features from code to improve retrieval."""
        features = []
        
        if 'def ' in code and code.count('def ') > 1:
            features.append('multiple functions')
        if 'class ' in code:
            features.append('object oriented')
        if 'for ' in code and 'if ' in code:
            features.append('conditional loops')
        if code.count('if ') > 3:
            features.append('high branching complexity')
        if 'try:' in code:
            features.append('exception handling')
        if 'lambda' in code:
            features.append('functional programming')
            
        return features

    def extract_intent(self, query: str) -> List[str]:
        """Extract refactoring intents from query."""
        query_lower = query.lower()
        intents = []
        
        for intent, data in self.refactoring_patterns.items():
            if any(keyword in query_lower for keyword in data['keywords']):
                intents.append(intent)
        
        return intents

    def expand_query_terms(self, query: str) -> str:
        """Expand query with related terms."""
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Add synonyms for detected patterns
        for pattern, data in self.refactoring_patterns.items():
            if any(keyword in query_lower for keyword in data['keywords']):
                expanded_terms.extend(data['synonyms'][:1])  # Add one synonym
        
        return ' '.join(expanded_terms)

    def preprocess_query(self, query: str) -> str:
        """Basic preprocessing of queries."""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Expand common abbreviations
        abbreviations = {
            'func': 'function',
            'var': 'variable',
            'obj': 'object',
            'len': 'length',
            'str': 'string',
            'int': 'integer'
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_word = abbreviations.get(word.lower(), word)
            expanded_words.append(expanded_word)
        
        return ' '.join(expanded_words)

    def identify_code_elements(self, query: str) -> List[str]:
        """Identify specific code elements mentioned in query."""
        elements = []
        
        code_keywords = {
            'function': ['function', 'func', 'def', 'method'],
            'class': ['class', 'object', 'instance'],
            'loop': ['loop', 'for', 'while', 'iterate'],
            'condition': ['if', 'condition', 'conditional', 'branch'],
            'variable': ['variable', 'var', 'parameter', 'argument']
        }
        
        query_lower = query.lower()
        
        for element, keywords in code_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                elements.append(element)
        
        return elements