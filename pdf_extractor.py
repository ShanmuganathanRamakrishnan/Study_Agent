"""
PDF Text Extraction Module for Study Agent

Extracts text from PDFs using pdfplumber.
- Text-only extraction (no OCR, no image processing)
- Preserves reading order
- Preserves section headers where possible
- Ignores images and diagrams
- Inline equations preserved as text if extractable

LIMITATIONS:
- No equation solving or symbolic math
- Scanned PDFs (image-only) will return empty text
- Complex layouts may have reading order issues
- Mathematical notation may be incomplete or garbled
"""

import io
import sys
from typing import Tuple

# Maximum pages to process (defensive guard against extremely large PDFs)
MAX_PAGES = 100


def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, dict]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_bytes: Raw PDF file bytes
        
    Returns:
        Tuple of (extracted_text, metadata_dict)
        metadata_dict contains:
            - page_count: int
            - char_count: int
            - word_count: int
            - has_images: bool (informational only, images are ignored)
            - pages_processed: int (may be less than page_count if truncated)
    
    Raises:
        ValueError: If PDF cannot be parsed
    """
    import pdfplumber
    
    text_parts = []
    page_count = 0
    pages_processed = 0
    has_images = False
    
    print(f"[PDF] Starting extraction ({len(pdf_bytes)} bytes)...", flush=True)
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            print(f"[PDF] Found {page_count} pages", flush=True)
            
            # Defensive guard: limit pages to prevent hangs
            pages_to_process = min(page_count, MAX_PAGES)
            if page_count > MAX_PAGES:
                print(f"[PDF] WARNING: Truncating to first {MAX_PAGES} pages", flush=True)
            
            for i, page in enumerate(pdf.pages[:pages_to_process]):
                try:
                    # Log progress for debugging
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"[PDF] Processing page {i + 1}/{pages_to_process}...", flush=True)
                    
                    # Check for images (informational only)
                    if page.images:
                        has_images = True
                    
                    # Extract text in reading order
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    
                    pages_processed += 1
                    
                except Exception as page_error:
                    # Log but continue - don't fail entire PDF for one bad page
                    print(f"[PDF] WARNING: Page {i + 1} extraction failed: {page_error}", flush=True)
                    continue
                    
    except Exception as e:
        print(f"[PDF] ERROR: Failed to parse PDF: {e}", flush=True)
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    if pages_processed == 0:
        raise ValueError("PDF appears to be empty or contains no extractable text")
    
    print(f"[PDF] Extracted text from {pages_processed} pages", flush=True)
    
    full_text = "\n\n".join(text_parts)
    
    # Clean up common extraction artifacts
    full_text = clean_extracted_text(full_text)
    
    word_count = len(full_text.split()) if full_text else 0
    print(f"[PDF] Final: {word_count} words, {len(full_text)} chars", flush=True)
    
    metadata = {
        "page_count": page_count,
        "pages_processed": pages_processed,
        "char_count": len(full_text),
        "word_count": word_count,
        "has_images": has_images
    }
    
    return full_text, metadata


def clean_extracted_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace but preserve paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip trailing whitespace
        line = line.rstrip()
        # Skip completely empty lines but preserve paragraph structure
        if line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(line)
    
    # Join and normalize multiple blank lines to single paragraph breaks
    result = '\n'.join(cleaned_lines)
    
    # Replace multiple consecutive newlines with double newline (paragraph break)
    import re
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def validate_pdf_content(text: str, min_words: int = 50) -> Tuple[bool, str]:
    """
    Validate that extracted PDF text is suitable for indexing.
    
    Returns:
        Tuple of (is_valid, reason)
    """
    if not text:
        return False, "PDF appears to be empty or image-only (no extractable text)"
    
    word_count = len(text.split())
    
    if word_count < min_words:
        return False, f"Extracted text too short ({word_count} words, minimum {min_words})"
    
    # Check for garbled text (high ratio of special characters)
    alpha_count = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_count / len(text) < 0.5:
        return False, "PDF text appears garbled or contains mostly non-text content"
    
    return True, ""
