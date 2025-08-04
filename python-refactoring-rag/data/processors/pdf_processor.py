"""PDF text extraction and processing utilities."""

import logging
from typing import List, Dict, Any
from pathlib import Path

from config.settings import PDFConfig, ContentType

logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")


class PDFProcessor:
    """Handles PDF text extraction and processing."""
    
    def __init__(self, config: PDFConfig):
        self.config = config
        
        if config.use_pymupdf and not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF requested but not available, falling back to PyPDF2")
            self.config.use_pymupdf = False
        
        if not self.config.use_pymupdf and not PYPDF2_AVAILABLE:
            raise ImportError("Neither PyMuPDF nor PyPDF2 is available for PDF processing")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using the configured method."""
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            if self.config.use_pymupdf and PYMUPDF_AVAILABLE:
                return self._extract_with_pymupdf(pdf_path)
            else:
                return self._extract_with_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better quality)."""
        text_content = []
        
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n".join(text_content)

    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback)."""
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n".join(text_content)

    def chunk_pdf_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split PDF text into chunks."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_size + paragraph_size > self.config.max_chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(self._create_text_chunk(chunk_text, chunk_id, metadata))
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, paragraph] if overlap_text else [paragraph]
                current_size = len(overlap_text) + paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(self._create_text_chunk(chunk_text, chunk_id, metadata))
        
        return chunks

    def _create_text_chunk(self, text: str, chunk_id: int, metadata: Dict = None) -> Dict:
        """Create a text chunk with metadata."""
        if metadata is None:
            metadata = {}
        
        return {
            'text': text,
            'code': text,  # For consistency with code chunks
            'metadata': {
                **metadata,
                'type': 'text',
                'content_type': ContentType.TEXT.value,
                'chunk_number': chunk_id,
                'word_count': len(text.split()),
                'char_count': len(text),
                'has_code_blocks': self._detect_code_blocks(text)
            },
            'chunk_id': f"text_{chunk_id}_{hash(text) % 10000}"
        }

    def _detect_code_blocks(self, text: str) -> bool:
        """Detect if text contains code blocks."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ',
            '```python', '```', 'if __name__',
            'return ', 'print(', 'for ', 'while '
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)

    def extract_code_blocks_from_text(self, text: str) -> List[Dict]:
        """Extract code blocks from PDF text."""
        if not self.config.extract_code_blocks:
            return []
        
        import re
        
        code_blocks = []
        
        # Pattern for code blocks marked with backticks
        backtick_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        backtick_matches = re.finditer(backtick_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in backtick_matches:
            code_content = match.group(1).strip()
            if len(code_content) > 20:  # Filter out very short snippets
                code_blocks.append({
                    'text': code_content,
                    'code': code_content,
                    'metadata': {
                        'type': 'code_block',
                        'source': 'pdf_extraction',
                        'extraction_method': 'backtick_pattern',
                        'char_count': len(code_content),
                        'line_count': len(code_content.split('\n'))
                    },
                    'chunk_id': f"pdf_code_{hash(code_content) % 10000}"
                })
        
        # Pattern for indented code blocks (common in technical books)
        indented_pattern = r'\n((?:    .*\n){3,})'
        indented_matches = re.finditer(indented_pattern, text)
        
        for match in indented_matches:
            code_content = match.group(1).strip()
            # Remove common indentation
            lines = code_content.split('\n')
            dedented_lines = [line[4:] if line.startswith('    ') else line for line in lines]
            dedented_code = '\n'.join(dedented_lines).strip()
            
            if len(dedented_code) > 20 and self._looks_like_python_code(dedented_code):
                code_blocks.append({
                    'text': dedented_code,
                    'code': dedented_code,
                    'metadata': {
                        'type': 'code_block',
                        'source': 'pdf_extraction',
                        'extraction_method': 'indentation_pattern',
                        'char_count': len(dedented_code),
                        'line_count': len(dedented_code.split('\n'))
                    },
                    'chunk_id': f"pdf_code_{hash(dedented_code) % 10000}"
                })
        
        return code_blocks

    def _looks_like_python_code(self, text: str) -> bool:
        """Heuristic to determine if text looks like Python code."""
        python_keywords = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'else:',
            'for ', 'while ', 'try:', 'except:', 'return ', 'yield ',
            'with ', 'as ', 'lambda ', 'pass', 'break', 'continue'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in python_keywords if keyword in text_lower)
        
        # Also check for Python-like patterns
        has_colons = text.count(':') > 0
        has_indentation = any(line.startswith('    ') or line.startswith('\t') 
                            for line in text.split('\n'))
        
        return keyword_count >= 2 and (has_colons or has_indentation)

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """Process multiple PDF files and return combined chunks."""
        all_chunks = []
        
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                text = self.extract_text_from_pdf(pdf_path)
                
                if text:
                    pdf_metadata = {
                        'source_file': Path(pdf_path).name,
                        'source_path': pdf_path,
                        'content_type': ContentType.TEXT.value
                    }
                    
                    # Extract regular text chunks
                    text_chunks = self.chunk_pdf_text(text, pdf_metadata)
                    all_chunks.extend(text_chunks)
                    
                    # Extract code blocks if enabled
                    if self.config.extract_code_blocks:
                        code_chunks = self.extract_code_blocks_from_text(text)
                        # Add source metadata to code chunks
                        for chunk in code_chunks:
                            chunk['metadata'].update(pdf_metadata)
                        all_chunks.extend(code_chunks)
                    
                    logger.info(f"Extracted {len(text_chunks)} text chunks and "
                              f"{len(code_chunks) if self.config.extract_code_blocks else 0} "
                              f"code chunks from {pdf_path}")
                else:
                    logger.warning(f"No text extracted from {pdf_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}")
                continue
        
        return all_chunks

    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        metadata = {
            'file_path': pdf_path,
            'file_name': Path(pdf_path).name,
            'file_size': 0,
            'page_count': 0,
            'title': '',
            'author': '',
            'creation_date': None
        }
        
        try:
            file_path = Path(pdf_path)
            if file_path.exists():
                metadata['file_size'] = file_path.stat().st_size
            
            if self.config.use_pymupdf and PYMUPDF_AVAILABLE:
                with fitz.open(pdf_path) as doc:
                    metadata['page_count'] = len(doc)
                    doc_metadata = doc.metadata
                    metadata['title'] = doc_metadata.get('title', '')
                    metadata['author'] = doc_metadata.get('author', '')
                    metadata['creation_date'] = doc_metadata.get('creationDate', '')
            
            elif PYPDF2_AVAILABLE:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata['page_count'] = len(reader.pages)
                    
                    if reader.metadata:
                        metadata['title'] = reader.metadata.get('/Title', '')
                        metadata['author'] = reader.metadata.get('/Author', '')
                        metadata['creation_date'] = reader.metadata.get('/CreationDate', '')
        
        except Exception as e:
            logger.error(f"Error extracting PDF metadata from {pdf_path}: {e}")
        
        return metadata