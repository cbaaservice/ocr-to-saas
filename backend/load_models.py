from __future__ import annotations

from typing import Any

import torch

from backend.config import (
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_GLM,
    MODEL_OVIS,
    TEST_MODE,
)

ovis_processor: Any = None
ovis_model: Any = None
glm_processor: Any = None
glm_model: Any = None


def load_models() -> None:
    """Eager-load both OCR models onto CUDA (ZeroGPU contract)."""
    global ovis_processor, ovis_model, glm_processor, glm_model
    if TEST_MODE:
        return

    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
    )

    ovis_processor = AutoProcessor.from_pretrained(
        MODEL_OVIS,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    ovis_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_OVIS,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    ovis_model.eval()

    glm_processor = AutoProcessor.from_pretrained(MODEL_GLM, trust_remote_code=True)
    glm_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_GLM,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    glm_model.eval()


def models_loaded() -> dict[str, bool]:
    if TEST_MODE:
        return {"ovis": True, "glm": True, "test_mode": True}
    return {
        "ovis": ovis_processor is not None and ovis_model is not None,
        "glm": glm_processor is not None and glm_model is not None,
        "test_mode": False,
    }
