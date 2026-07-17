from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "dist"

ModelChoice = Literal["OvisOCR2", "GLM-OCR"]

MODEL_OVIS = "ATH-MaaS/OvisOCR2"
MODEL_GLM = "zai-org/GLM-OCR"

TEST_MODE = os.getenv("OCR_TEST_MODE", os.getenv("OVISOCR_TEST_MODE", "0")).lower() in {
    "1",
    "true",
    "yes",
}

MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "16384"))
MAX_PDF_PAGES = int(os.getenv("OCR_MAX_PDF_PAGES", "50"))
PAGES_PER_GPU_REQUEST = max(1, min(5, int(os.getenv("OCR_PAGES_PER_GPU_REQUEST", "4"))))
GPU_SECONDS_PER_PAGE = max(15, int(os.getenv("OCR_GPU_SECONDS_PER_PAGE", "30")))
GPU_DURATION_FLOOR = max(15, int(os.getenv("OCR_GPU_DURATION_FLOOR", "45")))
GPU_DURATION_CEILING = max(
    GPU_DURATION_FLOOR,
    int(os.getenv("OCR_GPU_DURATION_CEILING", "120")),
)
PDF_RENDER_SCALE = float(os.getenv("OCR_PDF_RENDER_SCALE", "2.0"))
STREAM_MIN_CHARS = int(os.getenv("OCR_STREAM_MIN_CHARS", "64"))
STREAM_MAX_INTERVAL = float(os.getenv("OCR_STREAM_MAX_INTERVAL", "0.25"))
MIN_PIXELS = 448 * 448
MAX_PIXELS = 2880 * 2880

OVIS_OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)

GLM_DEFAULT_PROMPT = "Text Recognition:"
GLM_TASK_PROMPTS = {
    "Text": "Text Recognition:",
    "Formula": "Formula Recognition:",
    "Table": "Table Recognition:",
}

MOCK_MARKDOWN = r"""# Contact tracing — sample page

| Field | Value |
|-------|-------|
| Case ID | EVD-2024-0042 |
| Nom | Mbuyi, Jean |
| Contact | +243 81 000 0000 |
| Localité | Beni, Nord-Kivu |

## Notes

Fever onset $t_0$ reported 3 days prior to interview.

\[
R_t = \frac{C_{t}}{C_{t-1}}
\]

<img src="images/bbox_120_130_880_420.jpg" />

Source: field form (mock)."""


def default_prompt(model_choice: ModelChoice) -> str:
    if model_choice == "OvisOCR2":
        return OVIS_OCR_PROMPT
    if model_choice == "GLM-OCR":
        return GLM_DEFAULT_PROMPT
    raise ValueError(f"Unknown model_choice: {model_choice}")


def resolve_prompt(model_choice: ModelChoice, prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return default_prompt(model_choice)
    if model_choice == "GLM-OCR" and text in GLM_TASK_PROMPTS:
        return GLM_TASK_PROMPTS[text]
    return text


def hub_id_for(model_choice: ModelChoice) -> str:
    if model_choice == "OvisOCR2":
        return MODEL_OVIS
    if model_choice == "GLM-OCR":
        return MODEL_GLM
    raise ValueError(f"Unknown model_choice: {model_choice}")
