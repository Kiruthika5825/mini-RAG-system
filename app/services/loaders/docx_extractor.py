# docx_extractor.py
from pathlib import Path

try:
    from docx import Document
except ImportError:
    Document = None   # Safe fallback


def process_docx(file_path):
    file_path = Path(file_path)

    # Docx not installed
    if Document is None:
        print("python-docx is not installed. Install it using: pip install python-docx")
        return []

    # File missing
    if not file_path.exists():
        print(f"DOCX file not found: {file_path}")
        return []

    results = []

    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for para in paragraphs:
            results.append({
                "text": para,
                "source": str(file_path),
                "type": "docx",
                "title": file_path.name,
                "chunk_index": len(results)   # running global counter
            })

        return results

    except Exception as e:
        print("DOCX extraction failed:", e)
        return []

