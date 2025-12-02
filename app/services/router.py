"""
router.py
---------
- Imports loaders safely (optional dependencies)
- Maps loader types to loader functions
- Provides one unified entrypoint: route_to_loader()
"""

from .detector import detect_input_type

# ---------------------------------------------
# Safe optional imports for each loader
# ---------------------------------------------

# URL Loader
try:
    from .loaders.url_extractor import process_url
except Exception:
    process_url = None

# TXT Loader
try:
    from .loaders.txt_extractor import process_txt
except Exception:
    process_txt = None

# PDF Loader
try:
    from .loaders.pdf_extractor import process_pdf
except Exception:
    process_pdf = None

# DOCX Loader
try:
    from .loaders.docx_extractor import process_docx
except Exception:
    process_docx = None

# IMAGE Loader
try:
    from .loaders.image_extractor import process_image
except Exception:
    process_image = None


# ---------------------------------------------
# MAP INPUT TYPES → LOADER FUNCTIONS
# ---------------------------------------------
LOADER_MAP = {
    "url": process_url,
    "txt": process_txt,
    "pdf": process_pdf,
    "docx": process_docx,
    "image": process_image,
}


# ---------------------------------------------
# MAIN ROUTER
# ---------------------------------------------
def route_to_loader(input_path: str):
    """
    - Detect input type
    - Call its loader
    - Return standardized structured extraction output

    ALWAYS returns a list of dicts:
    [
        {
            "text": "...",
            "source": "...",
            "title": "...",
            "type": "...",
            "chunk_index": 0
        }
    ]
    """

    input_type = detect_input_type(input_path)

    if input_type not in LOADER_MAP:
        raise ValueError(f"Unsupported input type: {input_type}")

    loader_fn = LOADER_MAP[input_type]

    if loader_fn is None:
        raise RuntimeError(
            f"Loader for '{input_type}' not installed. "
            f"Check optional dependencies."
        )

    return loader_fn(input_path)
