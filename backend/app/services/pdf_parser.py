"""
WHY THIS CHANGE:
`fitz` (PyMuPDF), `easyocr`, and `PIL` are imported inside their respective functions/helpers
so their memory cost is only paid on the first actual PDF upload / OCR call, not at server boot.
This ensures fast application startup and avoids loading heavy deep learning models unnecessarily.
"""

# A page with fewer real characters than this is treated as "no usable
# text layer" (e.g. a scanned page saved as an image) and gets OCR'd.
_MIN_TEXT_CHARS = 20

# Render pages at this zoom before OCR — higher = better accuracy, slower.
_OCR_ZOOM = 2.0

# Global cache for the EasyOCR Reader instance to avoid re-loading model weights on every page.
_EASYOCR_READER = None


def _get_easyocr_reader():
    """Lazily initialize and cache the EasyOCR Reader."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["en"])
    return _EASYOCR_READER


def _ocr_page(page) -> str:
    """Render a PDF page to an image and run OCR on it using EasyOCR.
    Returns '' on any failure rather than raising, so a single bad/scanned
    page never takes down the whole upload."""
    try:
        import io
        import numpy as np
        import fitz  # PyMuPDF
        from PIL import Image

        # Render page to image
        matrix = fitz.Matrix(_OCR_ZOOM, _OCR_ZOOM)
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")

        # Load image and convert to numpy array for EasyOCR
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img)

        # Run EasyOCR
        reader = _get_easyocr_reader()
        # detail=0 returns a list of recognized text strings
        text_list = reader.readtext(img_np, detail=0)
        return " ".join(text_list)
    except Exception as e:
        # We swallow the error and print it to stdout for debugging,
        # letting the caller handle empty text gracefully.
        print(f"EasyOCR error: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str):
    import fitz  # PyMuPDF — imported lazily, only when a PDF is actually parsed

    doc = fitz.open(pdf_path)

    full_text = ""

    for page_num, page in enumerate(doc):

        text = page.get_text()

        # No (or barely any) embedded text layer -> likely a scanned image page.
        # Fall back to OCR so scanned RFPs still get analyzed instead of
        # silently producing an empty document.
        if len(text.strip()) < _MIN_TEXT_CHARS:
            ocr_text = _ocr_page(page)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text

        full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
        full_text += text

    return full_text