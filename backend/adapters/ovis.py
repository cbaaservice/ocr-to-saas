from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import torch
from PIL import Image

from backend.bbox import clean_truncated_repeats
from backend.config import (
    MAX_NEW_TOKENS,
    MOCK_MARKDOWN,
    STREAM_MAX_INTERVAL,
    STREAM_MIN_CHARS,
    TEST_MODE,
)
from backend import load_models


def generation_token_ids(processor: Any) -> dict[str, int]:
    tokenizer = processor.tokenizer
    return {
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
    }


def infer_stream(page_image: Image.Image, prompt: str) -> Iterator[str]:
    if TEST_MODE:
        for end in range(64, len(MOCK_MARKDOWN) + 64, 64):
            yield MOCK_MARKDOWN[:end]
        return

    processor = load_models.ovis_processor
    model = load_models.ovis_model
    if processor is None or model is None:
        raise RuntimeError("OvisOCR2 is not loaded.")

    from transformers import TextIteratorStreamer

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": page_image},
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
        enable_thinking=False,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    **generation_token_ids(processor),
                )
        except BaseException as error:
            errors.append(error)
            streamer.on_finalized_text("", stream_end=True)

    worker = threading.Thread(target=generate, name="ovisocr2-generate", daemon=True)
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
        raise RuntimeError("OvisOCR2 generation failed.") from errors[0]
    final_text = clean_truncated_repeats(text.strip())
    if final_text and final_text != last_yielded:
        yield final_text
