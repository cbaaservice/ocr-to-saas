import { useMemo, useState } from "react";

const SPACE_ID = "Tonic/ocr-ebola";
const SPACE_URL = "https://tonic-ocr-ebola.hf.space";

const PYTHON_SNIPPET = `from gradio_client import Client, handle_file

client = Client("${SPACE_ID}")
job = client.submit(
    handle_file("form.pdf"),  # or a .png / .jpg / .webp
    0,                        # page_index (0-based)
    4,                        # page_count per ZeroGPU lease
    "OvisOCR2",               # or "GLM-OCR"
    "",                       # prompt override (empty = model default)
    api_name="/run_ocr",
)
for chunk in job:
    print(chunk["event"], chunk.get("current_page"), chunk.get("char_count"))
    if chunk.get("event") == "complete":
        print(chunk["markdown"][:500])
`;

const JS_SNIPPET = `import { Client } from "@gradio/client";

const client = await Client.connect("${SPACE_ID}");
const stream = client.submit("/run_ocr", {
  image_path: file,       // File / Blob / FileData
  page_index: 0,
  page_count: 4,
  model_choice: "OvisOCR2",
  prompt: "",
});
for await (const msg of stream) {
  if (msg.type === "data") console.log(msg.data);
}
`;

type Tab = "python" | "javascript";

export default function ApiDocs() {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<Tab>("python");
  const [copied, setCopied] = useState(false);

  const snippet = useMemo(
    () => (tab === "python" ? PYTHON_SNIPPET : JS_SNIPPET),
    [tab],
  );

  async function copySnippet() {
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <section className="api-docs">
      <div className="api-docs-bar">
        <button type="button" className="btn secondary" onClick={() => setOpen((v) => !v)}>
          {open ? "Hide API" : "Use via API"}
        </button>
        <a className="api-link" href={`${SPACE_URL}/docs`} target="_blank" rel="noreferrer">
          OpenAPI /docs
        </a>
        <a
          className="api-link"
          href={`${SPACE_URL}/gradio_api/info`}
          target="_blank"
          rel="noreferrer"
        >
          Gradio /gradio_api/info
        </a>
        <a className="api-link" href={`${SPACE_URL}/healthz`} target="_blank" rel="noreferrer">
          /healthz
        </a>
      </div>

      {open && (
        <div className="api-panel">
          <p className="api-lead">
            Custom Vite UI — Gradio&apos;s built-in footer is not shown. Endpoint{" "}
            <code>/run_ocr</code> streams page batches (events:{" "}
            <code>page_start</code>, <code>stream</code>, <code>page_complete</code>,{" "}
            <code>complete</code>). Prefer <code>client.submit</code> over{" "}
            <code>predict</code> so you receive every chunk.
          </p>
          <div className="api-meta">
            <span>
              Space: <code>{SPACE_ID}</code>
            </span>
            <span>
              URL: <code>{SPACE_URL}</code>
            </span>
          </div>
          <div className="view-toggle api-tabs">
            <button
              type="button"
              className={tab === "python" ? "active" : ""}
              onClick={() => setTab("python")}
            >
              Python
            </button>
            <button
              type="button"
              className={tab === "javascript" ? "active" : ""}
              onClick={() => setTab("javascript")}
            >
              JavaScript
            </button>
          </div>
          <div className="api-code-wrap">
            <button type="button" className="api-copy" onClick={copySnippet}>
              {copied ? "Copied" : "Copy"}
            </button>
            <pre className="api-code">{snippet}</pre>
          </div>
          <ul className="api-params">
            <li>
              <code>image_path</code> — uploaded image or PDF
            </li>
            <li>
              <code>page_index</code> — 0-based start page (default 0)
            </li>
            <li>
              <code>page_count</code> — pages per GPU lease (default 4, max 5)
            </li>
            <li>
              <code>model_choice</code> — <code>OvisOCR2</code> or <code>GLM-OCR</code>
            </li>
            <li>
              <code>prompt</code> — optional override; empty uses the model default
            </li>
          </ul>
        </div>
      )}
    </section>
  );
}
