from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from typing import assert_never

from gradio.data_classes import FileData

from backend.adapters import glm as glm_adapter
from backend.adapters import ovis as ovis_adapter
from backend.bbox import clean_truncated_repeats, materialize_bbox_images, stream_safe_markdown
from backend.config import (
    PAGES_PER_GPU_REQUEST,
    TEST_MODE,
    ModelChoice,
    resolve_prompt,
)
from backend.documents import document_info, load_document_page, page_preview_data_uri
from backend.stream_protocol import stream_payload


def _file_path(file_data: FileData | dict[str, Any]) -> str:
    if isinstance(file_data, dict):
        path = file_data.get("path")
    else:
        path = getattr(file_data, "path", None)
    if not path:
        raise ValueError("No uploaded document was provided.")
    return str(path)


def run_ocr_batch(
    image_path: FileData | dict[str, Any],
    page_index: int = 0,
    page_count: int = PAGES_PER_GPU_REQUEST,
    model_choice: str = "OvisOCR2",
    prompt: str = "",
) -> Iterator[dict[str, Any]]:
    """Stream a bounded group of pages within one ZeroGPU reservation."""
    started = time.perf_counter()
    choice: ModelChoice
    if model_choice == "OvisOCR2":
        choice = "OvisOCR2"
    elif model_choice == "GLM-OCR":
        choice = "GLM-OCR"
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")

    path = _file_path(image_path)
    document_type, total_pages = document_info(path)
    page_index = int(page_index)
    if page_index < 0 or page_index >= total_pages:
        raise ValueError(
            f"Requested PDF page {page_index + 1}, but this document has {total_pages} pages."
        )

    requested_count = max(1, min(PAGES_PER_GPU_REQUEST, int(page_count)))
    batch_end_index = min(total_pages, page_index + requested_count)
    batch_start_page = page_index + 1
    batch_end_page = batch_end_index
    completed_pages: list[dict[str, Any]] = []
    active_prompt = resolve_prompt(choice, prompt)
    backend_name = "mock" if TEST_MODE else "transformers"

    print(
        f"[ocr] {choice} batch start pages {batch_start_page}-{batch_end_page}/{total_pages}",
        flush=True,
    )

    for current_index in range(page_index, batch_end_index):
        page_number = current_index + 1
        page_image = load_document_page(path, document_type, current_index)
        page_started = time.perf_counter()
        current = {
            "page_number": page_number,
            "markdown": "",
            "render_markdown": "",
            "status": "streaming",
            "elapsed_seconds": 0.0,
        }
        yield stream_payload(
            event="page_start",
            pages=[current],
            current_page=page_number,
            total_pages=total_pages,
            document_type=document_type,
            started=started,
            model_choice=choice,
            page_preview=page_preview_data_uri(page_image),
            batch_start_page=batch_start_page,
            batch_end_page=batch_end_page,
            backend=backend_name,
        )

        if choice == "OvisOCR2":
            stream = ovis_adapter.infer_stream(page_image, active_prompt)
        elif choice == "GLM-OCR":
            stream = glm_adapter.infer_stream(page_image, active_prompt)
        else:
            assert_never(choice)

        markdown = ""
        for partial in stream:
            markdown = partial
            if choice == "OvisOCR2":
                render = stream_safe_markdown(markdown)
            else:
                render = markdown
            current = {
                "page_number": page_number,
                "markdown": markdown,
                "render_markdown": render,
                "status": "streaming",
                "elapsed_seconds": round(time.perf_counter() - page_started, 3),
            }
            yield stream_payload(
                event="stream",
                pages=[current],
                current_page=page_number,
                total_pages=total_pages,
                document_type=document_type,
                started=started,
                model_choice=choice,
                batch_start_page=batch_start_page,
                batch_end_page=batch_end_page,
                backend=backend_name,
            )

        markdown = clean_truncated_repeats(markdown.strip())
        if not markdown:
            raise RuntimeError(f"The model returned an empty result for page {page_number}.")

        if choice == "OvisOCR2":
            render_final = materialize_bbox_images(markdown, page_image)
        elif choice == "GLM-OCR":
            render_final = markdown
        else:
            assert_never(choice)

        completed_page = {
            "page_number": page_number,
            "markdown": markdown,
            "render_markdown": render_final,
            "status": "complete",
            "elapsed_seconds": round(time.perf_counter() - page_started, 3),
        }
        completed_pages.append(completed_page)
        print(
            f"[ocr] page {page_number}/{total_pages} complete "
            f"({len(markdown)} chars, {completed_page['elapsed_seconds']}s)",
            flush=True,
        )
        yield stream_payload(
            event="page_complete",
            pages=[completed_page],
            current_page=page_number,
            total_pages=total_pages,
            document_type=document_type,
            started=started,
            model_choice=choice,
            batch_start_page=batch_start_page,
            batch_end_page=batch_end_page,
            backend=backend_name,
        )

    print(
        f"[ocr] batch complete pages {batch_start_page}-{batch_end_page}/{total_pages}",
        flush=True,
    )
    yield stream_payload(
        event="complete",
        pages=completed_pages,
        current_page=batch_end_page,
        total_pages=total_pages,
        document_type=document_type,
        started=started,
        model_choice=choice,
        batch_complete=True,
        batch_start_page=batch_start_page,
        batch_end_page=batch_end_page,
        backend=backend_name,
    )
