# pdf_loader.py
import fitz  # PyMuPDF
from pathlib import Path

def process_pdf(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"PDF file not found: {file_path}")
        return []

    results = []
    chunk_index = 0  # restart counting for every PDF

    try:
        # Open PDF
        pdf = fitz.open(file_path)

        for page in pdf:
            text = page.get_text("text")

            # Skip empty pages
            if not text.strip():
                continue

            # Split into meaningful paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para in paragraphs:
                results.append({
                    "text": para,
                    "source": str(file_path),
                    "type": "pdf",
                    "title": file_path.name,
                    "chunk_index": chunk_index
                })
                chunk_index += 1

        pdf.close()
        return results

    except Exception as e:
        print("PDF extraction failed:", e)
        return []

