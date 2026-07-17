from __future__ import annotations

import time
from typing import Any

from backend.config import ModelChoice, hub_id_for


def combine_pages(pages: list[dict[str, Any]], field: str) -> str:
    if len(pages) <= 1:
        return pages[0].get(field, "") if pages else ""
    return "\n\n---\n\n".join(
        f"<!-- Page {page['page_number']} -->\n\n{page.get(field, '')}" for page in pages
    )


def stream_payload(
    *,
    event: str,
    pages: list[dict[str, Any]],
    current_page: int,
    total_pages: int,
    document_type: str,
    started: float,
    model_choice: ModelChoice,
    page_preview: str | None = None,
    batch_complete: bool = False,
    batch_start_page: int | None = None,
    batch_end_page: int | None = None,
    backend: str = "transformers",
) -> dict[str, Any]:
    return {
        "event": event,
        "markdown": combine_pages(pages, "markdown"),
        "render_markdown": combine_pages(pages, "render_markdown"),
        "pages": pages,
        "current_page": current_page,
        "total_pages": total_pages,
        "document_type": document_type,
        "page_preview": page_preview,
        "batch_complete": batch_complete,
        "batch_start_page": batch_start_page,
        "batch_end_page": batch_end_page,
        "char_count": sum(len(page.get("markdown", "")) for page in pages),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": hub_id_for(model_choice),
        "model_choice": model_choice,
        "backend": backend,
        "mode": "base",
    }
