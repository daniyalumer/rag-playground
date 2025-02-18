import os
import re
import pandas as pd
import unicodedata
from PyPDF2 import PdfReader
from typing import List, Dict
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def sanitize_text(text: str) -> str:
    """Sanitize input text to remove invalid control characters while preserving Unicode."""
    # Remove all ASCII control characters except tabs (\t), newlines (\n, \r)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]', '', text)
    
    # Normalize Unicode (NFKC) to standardize text representation
    text = unicodedata.normalize("NFKC", text)
    
    return text.strip()
    
def preprocess_text(text: str) -> str:
    """Preprocess the input text by sanitizing it."""
    sanitized_text = sanitize_text(text)
    return sanitized_text

def process_pdfs(base_dir: str) -> pd.DataFrame:
    """Recursively process all PDFs in directory and subdirectories."""
    data = []
    base_path = Path(base_dir)

    # Walk through all subdirectories
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, filename)
                
                # Extract text from PDF
                extracted_text = extract_text_from_pdf(pdf_path)
                if not extracted_text:
                    continue

                # Preprocess the extracted text
                preprocessed_text = preprocess_text(extracted_text)
                if not preprocessed_text:
                    continue

                # Get relative path for better organization
                relative_path = os.path.relpath(pdf_path, base_path)
                
                # Add to data list
                data.append({
                    "filename": filename,
                    "path": relative_path,
                    "raw_text": extracted_text,
                    "preprocessed_text": preprocessed_text 
                })
                print(f"Processed: {relative_path}")

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    output_path = "data/extracted/resumes2.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df
