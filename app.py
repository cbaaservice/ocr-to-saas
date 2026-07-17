from __future__ import annotations

import mimetypes
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from backend.spaces_shim import spaces  # noqa: E402  (must precede torch)

import gradio as gr  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from gradio.data_classes import FileData  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from backend.config import (  # noqa: E402
    DIST_DIR,
    GPU_DURATION_CEILING,
    GPU_DURATION_FLOOR,
    GPU_SECONDS_PER_PAGE,
    MAX_PDF_PAGES,
    MODEL_GLM,
    MODEL_OVIS,
    PAGES_PER_GPU_REQUEST,
    TEST_MODE,
)
from backend.documents import document_info  # noqa: E402
from backend.load_models import load_models, models_loaded  # noqa: E402
from backend.orchestrator import _file_path, run_ocr_batch  # noqa: E402


def server_config() -> tuple[int, str | None, str | None]:
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    configured_root = (
        os.getenv("OCR_ROOT_PATH", "").strip() or os.getenv("GRADIO_ROOT_PATH", "").strip()
    )
    public_url = None
    root_path = None
    if configured_root.startswith(("http://", "https://")):
        public_url = configured_root
        path = urlsplit(configured_root).path.rstrip("/")
        root_path = path or None
    elif configured_root:
        root_path = configured_root.rstrip("/") or None
    return port, root_path, public_url


SERVER_PORT, ROOT_PATH, PUBLIC_URL = server_config()


class CachedStaticFiles(StaticFiles):
    """Serve immutable production assets from the browser cache after first load."""

    async def get_response(self, path: str, scope: dict[str, Any]) -> Any:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


EXAMPLE_ASSETS = (
    {
        path.name: (
            path.read_bytes(),
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        for path in (DIST_DIR / "examples").iterdir()
        if path.is_file()
    }
    if (DIST_DIR / "examples").is_dir()
    else {}
)

load_models()

app = gr.Server()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _gpu_duration(
    image_path: FileData | dict[str, Any],
    page_index: int = 0,
    page_count: int = PAGES_PER_GPU_REQUEST,
    model_choice: str = "OvisOCR2",
    prompt: str = "",
) -> int:
    del model_choice, prompt
    configured_duration = os.getenv("OCR_GPU_DURATION", "").strip()
    if configured_duration:
        return int(configured_duration)

    requested_count = max(1, min(PAGES_PER_GPU_REQUEST, int(page_count)))
    try:
        path = _file_path(image_path)
        _, total_pages = document_info(path)
        remaining_pages = max(1, total_pages - int(page_index))
        requested_count = min(requested_count, remaining_pages)
    except Exception:
        pass

    return max(
        GPU_DURATION_FLOOR,
        min(GPU_DURATION_CEILING, requested_count * GPU_SECONDS_PER_PAGE),
    )


@spaces.GPU(duration=_gpu_duration)
def _run_ocr_gpu(
    image_path: FileData,
    page_index: int = 0,
    page_count: int = PAGES_PER_GPU_REQUEST,
    model_choice: str = "OvisOCR2",
    prompt: str = "",
) -> Iterator[dict[str, Any]]:
    yield from run_ocr_batch(
        image_path,
        page_index=page_index,
        page_count=page_count,
        model_choice=model_choice,
        prompt=prompt,
    )


@app.api(name="run_ocr", concurrency_limit=1, time_limit=300)
@app.mcp.tool(name="run_ocr")
def run_ocr(
    image_path: FileData,
    page_index: int = 0,
    page_count: int = PAGES_PER_GPU_REQUEST,
    model_choice: str = "OvisOCR2",
    prompt: str = "",
) -> Iterator[dict[str, Any]]:
    """Stream OCR Markdown for a bounded batch of document pages."""
    yield from _run_ocr_gpu(
        image_path,
        page_index=page_index,
        page_count=page_count,
        model_choice=model_choice,
        prompt=prompt,
    )


@app.get("/healthz")
def healthz() -> JSONResponse:
    loaded = models_loaded()
    return JSONResponse(
        {
            "status": "ok",
            "models": {
                "OvisOCR2": MODEL_OVIS,
                "GLM-OCR": MODEL_GLM,
            },
            "loaded": loaded,
            "backend": "mock" if TEST_MODE else "transformers",
            "max_pdf_pages": MAX_PDF_PAGES,
            "pages_per_gpu_request": PAGES_PER_GPU_REQUEST,
            "gpu_seconds_per_page": GPU_SECONDS_PER_PAGE,
            "gpu_duration_floor": GPU_DURATION_FLOOR,
            "gpu_duration_ceiling": GPU_DURATION_CEILING,
            "root_path": ROOT_PATH,
            "public_url": PUBLIC_URL,
        }
    )


@app.get("/examples/{filename}")
def example_asset(filename: str) -> Response:
    asset = EXAMPLE_ASSETS.get(filename)
    if asset is None:
        raise HTTPException(status_code=404, detail="Example not found")
    content, media_type = asset
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


if DIST_DIR.is_dir():
    for route, directory in (
        ("/assets", DIST_DIR / "assets"),
        ("/brand", DIST_DIR / "brand"),
        ("/vendor", DIST_DIR / "vendor"),
    ):
        if directory.is_dir():
            app.mount(
                route,
                CachedStaticFiles(directory=directory),
                name=route.strip("/").replace("/", "-"),
            )


@app.get("/")
def homepage() -> FileResponse:
    index_path = DIST_DIR / "index.html"
    if not index_path.is_file():
        raise RuntimeError(
            "Frontend build missing. Run `cd frontend && npm ci && npm run build` "
            "before launching app.py."
        )
    return FileResponse(index_path, headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    favicon_path = DIST_DIR / "favicon.ico"
    if not favicon_path.is_file():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(
        favicon_path,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


if __name__ == "__main__":
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=SERVER_PORT,
        root_path=ROOT_PATH,
        show_error=True,
    )
