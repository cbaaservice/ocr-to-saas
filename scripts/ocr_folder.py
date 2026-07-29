#!/usr/bin/env python3
"""Batch-OCR every image/PDF in a folder via the Gradio /run_ocr API.

Example:
  python scripts/ocr_folder.py ./scans --out ./ocr_out --model OvisOCR2
  python scripts/ocr_folder.py ./scans --space https://tonic-ocr-ebola.hf.space --recursive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = IMAGE_SUFFIXES | PDF_SUFFIXES
DEFAULT_SPACE = "Tonic/ocr-ebola"
PAGES_PER_REQUEST = 4


def discover_files(folder: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in sorted(folder.glob(pattern))
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]
    return files


def unwrap_chunk(chunk: Any) -> dict[str, Any] | None:
    """Normalize Gradio Client stream payloads to a dict."""
    if isinstance(chunk, dict) and "event" in chunk:
        return chunk
    if isinstance(chunk, (list, tuple)) and chunk:
        first = chunk[0]
        if isinstance(first, dict) and "event" in first:
            return first
    return None


def ocr_file(
    client: Client,
    path: Path,
    *,
    model: str,
    prompt: str,
    page_count: int,
) -> dict[str, Any]:
    """OCR one document, paging through ZeroGPU batches until complete."""
    pages: dict[int, dict[str, Any]] = {}
    total_pages = 1
    page_index = 0
    started = time.perf_counter()

    while page_index < total_pages:
        job = client.submit(
            handle_file(str(path)),
            int(page_index),
            int(page_count),
            model,
            prompt,
            api_name="/run_ocr",
        )

        batch_complete = False
        saw_data = False
        last_payload: dict[str, Any] | None = None

        for raw in job:
            payload = unwrap_chunk(raw)
            if payload is None:
                continue
            saw_data = True
            last_payload = payload
            total_pages = int(payload.get("total_pages") or total_pages)

            for page in payload.get("pages") or []:
                number = int(page.get("page_number") or 0)
                if number < 1:
                    continue
                previous = pages.get(number)
                if (
                    previous is None
                    or page.get("status") == "complete"
                    or len(page.get("markdown") or "")
                    >= len(previous.get("markdown") or "")
                ):
                    pages[number] = page

            event = payload.get("event")
            current = payload.get("current_page")
            chars = payload.get("char_count")
            print(
                f"    [{path.name}] {event} page={current}/{total_pages} chars={chars}",
                flush=True,
            )
            if payload.get("batch_complete"):
                batch_complete = True
                break

        if not saw_data:
            raise RuntimeError(f"No data received for {path.name} at page_index={page_index}")

        next_index = first_incomplete_index(pages, total_pages)
        if next_index >= total_pages:
            break
        if not batch_complete and next_index == page_index:
            raise RuntimeError(f"OCR stalled on {path.name} at page {page_index + 1}")
        page_index = next_index

    ordered = [pages[i] for i in range(1, total_pages + 1) if i in pages]
    if len(ordered) != total_pages:
        missing = [i for i in range(1, total_pages + 1) if i not in pages]
        raise RuntimeError(f"Incomplete OCR for {path.name}; missing pages {missing}")

    if total_pages <= 1:
        markdown = ordered[0].get("markdown") or ""
    else:
        markdown = "\n\n---\n\n".join(
            f"<!-- Page {page['page_number']} -->\n\n{page.get('markdown') or ''}"
            for page in ordered
        )

    return {
        "source": str(path),
        "model": model,
        "total_pages": total_pages,
        "char_count": sum(len(page.get("markdown") or "") for page in ordered),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "markdown": markdown,
        "pages": ordered,
        "last_payload": {
            "model": (last_payload or {}).get("model"),
            "backend": (last_payload or {}).get("backend"),
        },
    }


def first_incomplete_index(pages: dict[int, dict[str, Any]], total_pages: int) -> int:
    for number in range(1, total_pages + 1):
        page = pages.get(number)
        if page is None or page.get("status") != "complete":
            return number - 1
    return total_pages


def output_path_for(source: Path, folder: Path, out_dir: Path) -> Path:
    relative = source.relative_to(folder)
    target = out_dir / relative
    return target.with_suffix(target.suffix + ".md") if target.suffix else target.with_suffix(".md")


def write_result(result: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result["markdown"], encoding="utf-8")
    meta_path = destination.with_suffix(".json")
    meta = {key: value for key, value in result.items() if key not in {"markdown", "pages"}}
    meta["pages"] = [
        {
            "page_number": page.get("page_number"),
            "status": page.get("status"),
            "elapsed_seconds": page.get("elapsed_seconds"),
            "char_count": len(page.get("markdown") or ""),
        }
        for page in result.get("pages") or []
    ]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR all images/PDFs in a folder via Tonic/ocr-ebola (or another Space).",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing PNG/JPEG/WebP/PDF files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for .md (+ .json sidecars). Default: <folder>/ocr_out",
    )
    parser.add_argument(
        "--space",
        default=DEFAULT_SPACE,
        help=f"Gradio Space id or URL (default: {DEFAULT_SPACE})",
    )
    parser.add_argument(
        "--model",
        choices=("OvisOCR2", "GLM-OCR"),
        default="OvisOCR2",
        help="Model choice passed to /run_ocr",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt override (empty = model default)",
    )
    parser.add_argument(
        "--page-count",
        type=int,
        default=PAGES_PER_REQUEST,
        help="Pages per ZeroGPU lease (1-5, default 4)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose .md already exists in --out",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per file on failure (default: 2)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between files (default: 1.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 2

    out_dir = (args.out or (folder / "ocr_out")).expanduser().resolve()
    page_count = max(1, min(5, int(args.page_count)))
    files = discover_files(folder, recursive=args.recursive)
    if not files:
        print(f"No supported files in {folder} ({', '.join(sorted(SUPPORTED))})")
        return 1

    print(f"Space:  {args.space}")
    print(f"Model:  {args.model}")
    print(f"Files:  {len(files)}")
    print(f"Output: {out_dir}")
    client = Client(args.space)

    ok = 0
    failed: list[str] = []
    for index, path in enumerate(files, start=1):
        destination = output_path_for(path, folder, out_dir)
        if args.skip_existing and destination.is_file():
            print(f"[{index}/{len(files)}] skip existing {destination.name}")
            ok += 1
            continue

        print(f"[{index}/{len(files)}] OCR {path}")
        attempt = 0
        while True:
            attempt += 1
            try:
                result = ocr_file(
                    client,
                    path,
                    model=args.model,
                    prompt=args.prompt,
                    page_count=page_count,
                )
                write_result(result, destination)
                print(
                    f"    wrote {destination} "
                    f"({result['char_count']} chars, {result['total_pages']} page(s), "
                    f"{result['elapsed_seconds']}s)",
                    flush=True,
                )
                ok += 1
                break
            except Exception as error:
                print(f"    error (attempt {attempt}/{args.retries + 1}): {error}", file=sys.stderr)
                if attempt > args.retries:
                    failed.append(str(path))
                    break
                time.sleep(min(8.0, args.sleep * attempt * 2))

        if index < len(files) and args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done: {ok}/{len(files)} succeeded")
    if failed:
        print("Failed:", file=sys.stderr)
        for item in failed:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
