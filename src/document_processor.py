"""
Document Processor Module
Handles PDF text extraction and cleaning
"""

import os
from pathlib import Path
from typing import Dict, List
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text.strip()


def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def load_documents_from_folder(folder_path: str) -> Dict[str, str]:
    """Load all PDF documents from a folder and extract their text."""
    documents = {}
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return documents

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in sorted(pdf_files):
        filename = pdf_file.name
        print(f"  Processing: {filename}")
        text = extract_text_from_pdf(str(pdf_file))
        cleaned_text = clean_text(text)
        documents[filename] = cleaned_text if cleaned_text else ""

    return documents
