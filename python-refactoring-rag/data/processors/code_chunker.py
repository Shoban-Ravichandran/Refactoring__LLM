"""Code chunking strategies for processing Python code."""

import ast
import logging
from typing import List, Dict, Any
from config.settings import CodeChunkConfig

logger = logging.getLogger(__name__)


class CodeChunker:
    """Chunker designed to maximize useful chunks from code dataset."""
    
    def __init__(self, config: CodeChunkConfig):
        self.config = config
        self.chunk_counter = 0

    def process_code_aggressively(self, code: str, metadata: Dict) -> List[Dict]:
        """Create multiple overlapping chunks to maximize coverage."""
        chunks = []
        
        # Strategy 1: Try to parse as Python AST
        try:
            tree = ast.parse(code)
            # Extract functions and classes
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(code, node) or self._extract_node_code(code, node)
                    if func_code and len(func_code) > 30:
                        functions.append({
                            'name': node.name,
                            'code': func_code,
                            'type': 'function'
                        })
                elif isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(code, node) or self._extract_node_code(code, node)
                    if class_code and len(class_code) > 50:
                        classes.append({
                            'name': node.name,
                            'code': class_code,
                            'type': 'class'
                        })
            
            # Create chunks from functions
            for func in functions:
                chunk = self._create_enhanced_chunk(
                    func['code'],
                    {**metadata, 'type': 'function', 'name': func['name']}
                )
                chunks.append(chunk)
            
            # Create chunks from classes
            for cls in classes:
                chunk = self._create_enhanced_chunk(
                    cls['code'],
                    {**metadata, 'type': 'class', 'name': cls['name']}
                )
                chunks.append(chunk)
            
            # If we don't have enough chunks, create line-based chunks
            if len(chunks) < 2:
                line_chunks = self._create_line_based_chunks(code, metadata)
                chunks.extend(line_chunks)
                
        except SyntaxError:
            # Fallback: create line-based chunks for unparseable code
            chunks = self._create_line_based_chunks(code, metadata)
        
        return chunks[:4]  # Limit to 4 chunks per code example

    def _create_line_based_chunks(self, code: str, metadata: Dict) -> List[Dict]:
        """Create chunks by splitting code into logical sections."""
        lines = code.split('\n')
        chunks = []
        chunk_size = 25  # Smaller chunks for better granularity
        overlap = 5
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            if len(chunk_lines) < 5:  # Skip very small chunks
                continue
                
            chunk_code = '\n'.join(chunk_lines)
            if chunk_code.strip():
                chunk = self._create_enhanced_chunk(
                    chunk_code,
                    {**metadata, 'type': 'code_block', 'chunk_index': i}
                )
                chunks.append(chunk)
        
        return chunks

    def _create_enhanced_chunk(self, code: str, metadata: Dict) -> Dict:
        """Create a chunk with enhanced metadata."""
        enhanced_metadata = metadata.copy()
        
        # Add code characteristics
        enhanced_metadata.update({
            'has_functions': 'def ' in code,
            'has_classes': 'class ' in code,
            'has_loops': any(kw in code for kw in ['for ', 'while ']),
            'has_conditions': 'if ' in code,
            'line_count': len(code.split('\n')),
            'char_count': len(code),
            'chunk_id': f"chunk_{self.chunk_counter}_{hash(code) % 100000}"
        })
        
        # Detect refactoring intent from description
        description = metadata.get('description', '').lower()
        refactoring_type = metadata.get('refactoring_type', '').lower()
        intent_tags = []
        
        if any(word in description + refactoring_type for word in ['complex', 'nested', 'extract']):
            intent_tags.append('complexity_reduction')
        if any(word in description + refactoring_type for word in ['readable', 'clear', 'naming']):
            intent_tags.append('readability_improvement')
        if any(word in description + refactoring_type for word in ['performance', 'optimize']):
            intent_tags.append('performance_optimization')
        
        enhanced_metadata['refactoring_intents'] = intent_tags
        self.chunk_counter += 1
        
        return {
            'text': code,
            'code': code,
            'metadata': enhanced_metadata,
            'chunk_id': enhanced_metadata['chunk_id']
        }

    def _extract_node_code(self, full_code: str, node) -> str:
        """Fallback method to extract code from AST node."""
        try:
            import astunparse
            return astunparse.unparse(node)
        except ImportError:
            # Ultimate fallback: extract by line numbers
            lines = full_code.split('\n')
            start_line = getattr(node, 'lineno', 1) - 1
            end_line = getattr(node, 'end_lineno', start_line + 10)
            return '\n'.join(lines[start_line:end_line])

    def chunk_code_by_functions(self, code: str, metadata: Dict = None) -> List[Dict]:
        """Chunk code by extracting individual functions and classes."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        try:
            tree = ast.parse(code)
            
            # Extract top-level functions and classes
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    node_code = ast.get_source_segment(code, node)
                    if node_code:
                        chunk_metadata = {
                            **metadata,
                            'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                            'name': node.name,
                            'line_start': node.lineno,
                            'line_end': getattr(node, 'end_lineno', node.lineno)
                        }
                        
                        chunk = self._create_enhanced_chunk(node_code, chunk_metadata)
                        chunks.append(chunk)
        
        except SyntaxError as e:
            logger.warning(f"Failed to parse code as Python: {e}")
            # Fall back to line-based chunking
            chunks = self._create_line_based_chunks(code, metadata)
        
        return chunks

    def chunk_code_by_complexity(self, code: str, metadata: Dict = None) -> List[Dict]:
        """Chunk code based on complexity thresholds."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Calculate simple complexity measure
                    complexity = self._calculate_node_complexity(node)
                    
                    if complexity >= 5:  # High complexity threshold
                        node_code = ast.get_source_segment(code, node)
                        if node_code:
                            chunk_metadata = {
                                **metadata,
                                'type': 'function',
                                'name': node.name,
                                'complexity': {'estimated': complexity},
                                'complexity_level': 'high' if complexity >= 10 else 'medium'
                            }
                            
                            chunk = self._create_enhanced_chunk(node_code, chunk_metadata)
                            chunks.append(chunk)
        
        except SyntaxError:
            chunks = self._create_line_based_chunks(code, metadata)
        
        return chunks

    def _calculate_node_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for an AST node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity

    def create_overlapping_chunks(self, code: str, metadata: Dict = None) -> List[Dict]:
        """Create overlapping chunks for better context coverage."""
        if metadata is None:
            metadata = {}
        
        lines = code.split('\n')
        chunks = []
        
        chunk_size = self.config.max_lines
        overlap = self.config.overlap_lines
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            
            if len(chunk_lines) >= self.config.min_lines:
                chunk_code = '\n'.join(chunk_lines)
                
                chunk_metadata = {
                    **metadata,
                    'type': 'overlapping_chunk',
                    'chunk_index': i,
                    'overlap_size': overlap,
                    'total_lines': len(lines)
                }
                
                chunk = self._create_enhanced_chunk(chunk_code, chunk_metadata)
                chunks.append(chunk)
        
        return chunks