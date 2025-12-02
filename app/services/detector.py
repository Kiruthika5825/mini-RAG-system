"""
detector.py
-----------
Responsible ONLY for:
- URL detection
- MIME detection (magic)
- Extension fallback
- Mapping MIME → internal loader type

No extraction logic here.
"""

import os
from pathlib import Path

# Try python-magic (preferred)
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False


# ----------------------------------------------------
# URL CHECK
# ----------------------------------------------------
def is_url(input_str: str) -> bool:
    return isinstance(input_str, str) and input_str.startswith(("http://", "https://"))


# ----------------------------------------------------
# MAIN DETECTION FUNCTION
# ----------------------------------------------------
def detect_input_type(input_path: str) -> str:
    """
    Returns one of:
        url, txt, pdf, docx, image, json, unsupported
    """

    # ------------------------------
    # 1. URL CHECK
    # ------------------------------
    if is_url(input_path):
        return "url"

    # ------------------------------
    # 2. FILE EXISTS CHECK
    # ------------------------------
    if not os.path.exists(input_path):
        return "unsupported"

    mime_type = None

    # ------------------------------
    # 3. MIME DETECTION (magic)
    # ------------------------------
    if MAGIC_AVAILABLE:
        try:
            mime_type = magic.from_file(input_path, mime=True)
        except Exception:
            mime_type = None

    # ------------------------------
    # 4. EXTENSION FALLBACK
    # ------------------------------
    if mime_type is None:
        ext = Path(input_path).suffix.lower()

        ext_to_mime = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".json": "application/json",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }

        mime_type = ext_to_mime.get(ext)

    if mime_type is None:
        return "unsupported"

    # ------------------------------
    # 5. MAP MIME TO INTERNAL TYPES
    # ------------------------------
    if mime_type == "text/plain":
        return "txt"

    if mime_type == "application/pdf":
        return "pdf"

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return "docx"

    if mime_type in ("image/jpeg", "image/png"):
        return "image"

    if mime_type == "application/json":
        return "json"

    return "unsupported"
