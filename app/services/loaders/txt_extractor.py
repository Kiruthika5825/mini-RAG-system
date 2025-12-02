# process_txt.py
from pathlib import Path

def process_txt(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"TXT file not found: {file_path}")
        return []

    try:
        # Read file safely
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for badly encoded txt files
            text = file_path.read_text(encoding="latin-1")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into paragraphs
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "text": chunk,
                "source": str(file_path),
                "type": "txt",
                "title": file_path.name,
                "chunk_index": i
            })

        return results

    except Exception as e:
        print("TXT extraction failed:", e)
        return []

