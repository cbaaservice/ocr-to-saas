from __future__ import annotations

import base64
import io
from pathlib import Path

import fitz
from PIL import Image, ImageOps

from backend.config import MAX_PDF_PAGES, PDF_RENDER_SCALE


def document_info(path: str) -> tuple[str, int]:
    suffix = Path(path).suffix.lower()
    try:
        with Path(path).open("rb") as file:
            header = file.read(5)
    except OSError as error:
        raise ValueError("The uploaded document could not be read.") from error

    if suffix == ".pdf" or header == b"%PDF-":
        with fitz.open(path) as document:
            total_pages = document.page_count
        if total_pages < 1:
            raise ValueError("The uploaded PDF has no pages.")
        if total_pages > MAX_PDF_PAGES:
            raise ValueError(
                f"This demo accepts up to {MAX_PDF_PAGES} PDF pages; received {total_pages}."
            )
        return "pdf", total_pages

    try:
        with Image.open(path) as source:
            source.verify()
    except Exception as error:
        raise ValueError("Please upload a valid PNG, JPEG, WebP, or PDF file.") from error
    return "image", 1


def load_document_page(path: str, document_type: str, page_index: int) -> Image.Image:
    if document_type == "pdf":
        with fitz.open(path) as document:
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE),
                colorspace=fitz.csRGB,
                alpha=False,
            )
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def page_preview_data_uri(page_image: Image.Image) -> str:
    preview = page_image.copy().convert("RGB")
    preview.thumbnail((1400, 1800), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=False)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
