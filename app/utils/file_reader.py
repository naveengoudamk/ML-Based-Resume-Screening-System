import PyPDF2
import docx
import os

def extract_text_from_pdf(file_stream):
    """
    Extracts text from a PDF file stream or path.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text.strip()

def extract_text_from_docx(file_stream):
    """
    Extracts text from a DOCX file stream or path.
    """
    text = ""
    try:
        doc = docx.Document(file_stream)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""
    return text.strip()

def extract_text(file_path):
    """
    Determines file type and extracts text accordingly.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    with open(file_path, 'rb') as f:
        if ext == '.pdf':
            return extract_text_from_pdf(f)
        elif ext == '.docx':
            return extract_text_from_docx(f)
        else:
            return ""

import io

def extract_text_from_stream(file_stream, filename):
    """
    Helper for handling Flask uploads directly.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_stream)
    elif ext == '.docx':
        return extract_text_from_docx(file_stream)
    else:
        # Try to read as plain text if all else fails
        try:
            return file_stream.read().decode('utf-8')
        except:
            return ""
