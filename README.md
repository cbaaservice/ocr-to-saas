---
title: OCR Capacity Building as a Service
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
python_version: "3.12.12"
app_file: app.py
pinned: false
license: mit
hardware: zero-gpu
startup_duration_timeout: 30m
models:
  - ATH-MaaS/OvisOCR2
  - zai-org/GLM-OCR
preload_from_hub:
  - ATH-MaaS/OvisOCR2
  - zai-org/GLM-OCR
short_description: Open OCR API for DRC Ebola contact-tracing docs.
---

07/17 trigger rebuild

# OCR Capacity Building as a Service

Open OCR API for Ebola contact-tracing documents in the DRC. Choose between
[ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) and
[zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), upload a page image or
multi-page PDF, and stream Markdown (with LaTeX formulas and HTML tables).

**OvisOCR2** emits figure placeholders that this Space materializes as cropped
visual regions. **GLM-OCR** returns text recognition output without bbox crops.

## Supported inputs

- Images: PNG, JPEG, WebP
- PDFs: up to 50 pages (rasterized locally; up to 4 pages per ZeroGPU lease)

## API

Streaming endpoint: `/run_ocr` (also registered as MCP tool `run_ocr`).

On the live Space UI, open **Use via API** for copy-paste clients. Machine-readable docs:

- Gradio schema: [`/gradio_api/info`](https://tonic-ocr-ebola.hf.space/gradio_api/info)
- OpenAPI: [`/docs`](https://tonic-ocr-ebola.hf.space/docs) · [`/openapi.json`](https://tonic-ocr-ebola.hf.space/openapi.json)
- Health: [`/healthz`](https://tonic-ocr-ebola.hf.space/healthz)

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `image_path` | FileData | required | Uploaded image or PDF |
| `page_index` | int | `0` | 0-based start page for this lease |
| `page_count` | int | `4` | Max pages in this GPU batch |
| `model_choice` | str | `"OvisOCR2"` | `"OvisOCR2"` or `"GLM-OCR"` |
| `prompt` | str | `""` | Override; empty → model default |

```python
from gradio_client import Client, handle_file

client = Client("Tonic/ocr-ebola")
job = client.submit(
    handle_file("form.pdf"),
    0,
    4,
    "OvisOCR2",
    "",
    api_name="/run_ocr",
)
for chunk in job:
    print(chunk["event"], chunk.get("current_page"), chunk.get("char_count"))
```

Also available as an MCP tool named `run_ocr`. Health check: `GET /healthz`.

## Local run

```bash
# Backend deps (Space installs requirements.txt automatically)
pip install -r requirements.txt

# Frontend (build once; commit dist/ for Spaces)
cd frontend && npm install && npm run build && cd ..

# Mock stream without loading weights (Windows PowerShell: $env:OCR_TEST_MODE=1)
OCR_TEST_MODE=1 python app.py

# Real models (needs CUDA)
python app.py
```

Open `http://127.0.0.1:7860`.
