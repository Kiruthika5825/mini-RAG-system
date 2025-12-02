# png_extractor.py
from pathlib import Path

# ---- Safe import handling (professional way) ----
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def process_image(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"image file not found: {file_path}")
        return []

    # Check if OCR libraries are available
    if pytesseract is None or Image is None:
        raise RuntimeError("pytesseract or Pillow is not installed")

    results = []

    try:
        # Open image
        img = Image.open(file_path)

        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(img)

        # Clean text
        text = text.strip()

        if not text:
            print(f"No text extracted from PNG: {file_path}")
            return []

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            results.append({
                "text": para,
                "source": str(file_path),
                "type": "image",
                "title": file_path.name,
                "chunk_index": len(results)  # global running index
            })

        return results

    except Exception as e:
        print("PNG extraction failed:", e)
        return []

