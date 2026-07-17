from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator

import torch
from PIL import Image, ImageOps

from backend.bbox import clean_truncated_repeats
from backend.config import (
    MAX_NEW_TOKENS,
    MOCK_MARKDOWN,
    STREAM_MAX_INTERVAL,
    STREAM_MIN_CHARS,
    TEST_MODE,
)
from backend import load_models


def infer_stream(page_image: Image.Image, prompt: str) -> Iterator[str]:
    if TEST_MODE:
        # GLM path: no bbox tags in mock
        mock = MOCK_MARKDOWN.replace(
            '<img src="images/bbox_120_130_880_420.jpg" />',
            "(figure omitted)",
        )
        for end in range(64, len(mock) + 64, 64):
            yield mock[:end]
        return

    processor = load_models.glm_processor
    model = load_models.glm_model
    if processor is None or model is None:
        raise RuntimeError("GLM-OCR is not loaded.")

    from transformers import TextIteratorStreamer

    if page_image.mode in ("RGBA", "LA", "P"):
        page_image = page_image.convert("RGB")
    page_image = ImageOps.exif_transpose(page_image)

    tmp_path: str | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        page_image.save(tmp.name, "PNG")
        tmp_path = tmp.name
        tmp.close()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": tmp_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        streamer = TextIteratorStreamer(
            processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
            except BaseException as error:
                errors.append(error)
                try:
                    streamer.end()
                except Exception:
                    pass

        worker = threading.Thread(target=generate, name="glm-ocr-generate", daemon=True)
        worker.start()
        text = ""
        last_yielded = ""
        last_yield_time = time.monotonic()
        for fragment in streamer:
            text += fragment
            now = time.monotonic()
            if (
                len(text) - len(last_yielded) >= STREAM_MIN_CHARS
                or now - last_yield_time >= STREAM_MAX_INTERVAL
            ):
                yield text
                last_yielded = text
                last_yield_time = now

        worker.join()
        if errors:
            raise RuntimeError("GLM-OCR generation failed.") from errors[0]
        final_text = clean_truncated_repeats(text.strip())
        if final_text and final_text != last_yielded:
            yield final_text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
