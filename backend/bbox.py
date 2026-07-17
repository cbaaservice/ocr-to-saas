from __future__ import annotations

import base64
import html
import io
import re

from PIL import Image

BBOX_IMAGE_PATTERN = re.compile(
    r'<img\s+src=["\']images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg["\']\s*/?>',
    flags=re.IGNORECASE,
)

UNMATERIALIZED_BBOX_IMAGE_PATTERN = re.compile(
    r'<img\b[^>]*\bsrc=["\']images/bbox_[^"\']+["\'][^>]*>',
    flags=re.IGNORECASE,
)


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Remove a repeated suffix created when generation reaches its token ceiling."""
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]
    return text


def neutralize_unmaterialized_bbox_images(markdown: str) -> str:
    """Render placeholder examples as code instead of issuing broken requests."""

    def replace(match: re.Match[str]) -> str:
        escaped = html.escape(match.group(0), quote=False)
        return f'<code class="unresolved-image-reference">{escaped}</code>'

    return UNMATERIALIZED_BBOX_IMAGE_PATTERN.sub(replace, markdown)


def stream_safe_markdown(markdown: str) -> str:
    """Avoid broken image requests until a page's bbox crops are materialized."""
    return neutralize_unmaterialized_bbox_images(
        BBOX_IMAGE_PATTERN.sub(
            '<div class="visual-placeholder">Preparing visual region…</div>',
            markdown,
        )
    )


def materialize_bbox_images(markdown: str, page_image: Image.Image) -> str:
    """Replace bbox image placeholders in rendered output with safe data-URI crops."""
    width, height = page_image.size

    def replace(match: re.Match[str]) -> str:
        left, top, right, bottom = (int(value) for value in match.groups())
        x1 = max(0, min(width, round(left * width / 1000)))
        y1 = max(0, min(height, round(top * height / 1000)))
        x2 = max(0, min(width, round(right * width / 1000)))
        y2 = max(0, min(height, round(bottom * height / 1000)))
        if x2 <= x1 or y2 <= y1:
            return match.group(0)

        crop = page_image.crop((x1, y1, x2, y2)).convert("RGB")
        crop.thumbnail((1200, 1200), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=85, optimize=False)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return (
            f'<img src="data:image/jpeg;base64,{payload}" alt="Visual region" '
            'loading="lazy" decoding="async" />'
        )

    return neutralize_unmaterialized_bbox_images(BBOX_IMAGE_PATTERN.sub(replace, markdown))
