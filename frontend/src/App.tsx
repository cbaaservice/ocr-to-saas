import { useEffect, useMemo, useRef, useState } from "react";
import { runOcrDocument, uploadDocument } from "./api/client";
import ApiDocs from "./components/ApiDocs";
import type { ModelChoice, PageResult } from "./types";

const EXAMPLES = [
  { file: "table-sample.webp", label: "Financial table" },
  { file: "handwriting-sample.jpg", label: "Handwriting" },
  { file: "paper-sample.png", label: "Research page" },
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [modelChoice, setModelChoice] = useState<ModelChoice>("OvisOCR2");
  const [prompt, setPrompt] = useState("");
  const [promptOpen, setPromptOpen] = useState(false);
  const [status, setStatus] = useState("Upload a document to begin");
  const [running, setRunning] = useState(false);
  const [markdown, setMarkdown] = useState("");
  const [renderMarkdown, setRenderMarkdown] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [pages, setPages] = useState<PageResult[]>([]);
  const [view, setView] = useState<"render" | "source">("render");
  const abortRef = useRef<AbortController | null>(null);
  const renderRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [objectUrl]);

  useEffect(() => {
    const node = renderRef.current;
    if (!node || view !== "render") {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        if (window.MathJax?.startup?.promise) {
          await window.MathJax.startup.promise;
        }
        if (!cancelled && window.MathJax?.typesetPromise) {
          await window.MathJax.typesetPromise([node]);
        }
      } catch {
        // MathJax optional during early load
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [renderMarkdown, view]);

  const previewSrc = useMemo(() => preview || objectUrl, [preview, objectUrl]);
  const isPdf = Boolean(file?.type === "application/pdf" || file?.name.toLowerCase().endsWith(".pdf"));

  function onPickFile(next: File | null) {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
    setFile(next);
    setObjectUrl(next && !next.type.includes("pdf") ? URL.createObjectURL(next) : null);
    setPreview(null);
    setMarkdown("");
    setRenderMarkdown("");
    setPages([]);
    setStatus(next ? `Ready · ${next.name}` : "Upload a document to begin");
  }

  async function loadExample(filename: string) {
    const response = await fetch(`./examples/${filename}`);
    if (!response.ok) {
      setStatus(`Example not found: ${filename}`);
      return;
    }
    const blob = await response.blob();
    const exampleFile = new File([blob], filename, { type: blob.type || "application/octet-stream" });
    onPickFile(exampleFile);
  }

  async function onRun() {
    if (!file || running) {
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setStatus("Uploading…");
    try {
      const fileData = await uploadDocument(file);
      await runOcrDocument({
        fileData,
        modelChoice,
        prompt,
        signal: controller.signal,
        handlers: {
          onStatus: setStatus,
          onUpdate: (state) => {
            setPages(state.pages);
            setMarkdown(state.markdown);
            setRenderMarkdown(state.renderMarkdown);
            if (state.preview) {
              setPreview(state.preview);
            }
          },
        },
      });
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        setStatus("Cancelled");
      } else {
        setStatus((error as Error).message || "OCR failed");
      }
    } finally {
      setRunning(false);
    }
  }

  function onCancel() {
    abortRef.current?.abort();
    setRunning(false);
    setStatus("Cancelled");
  }

  return (
    <div className="app">
      <header className="hero">
        <p className="eyebrow">DRC · Contact tracing OCR</p>
        <h1>OCR Capacity Building as a Service</h1>
        <p className="subtitle">
          Open OCR for Ebola contact-tracing forms — OvisOCR2 and GLM-OCR on ZeroGPU.
        </p>
        <p className="guidance">Upload a page or PDF → choose a model → Run</p>

        <div className="cta-row">
          <button
            type="button"
            className="btn secondary"
            onClick={() => fileInputRef.current?.click()}
            disabled={running}
          >
            Upload
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,application/pdf"
            hidden
            onChange={(event) => onPickFile(event.target.files?.[0] ?? null)}
          />
          <div className="model-toggle" role="group" aria-label="Model">
            {(["OvisOCR2", "GLM-OCR"] as ModelChoice[]).map((model) => (
              <button
                key={model}
                type="button"
                className={modelChoice === model ? "active" : ""}
                onClick={() => setModelChoice(model)}
                disabled={running}
              >
                {model}
              </button>
            ))}
          </div>
          {!running ? (
            <button type="button" className="btn primary" onClick={onRun} disabled={!file}>
              Run OCR
            </button>
          ) : (
            <button type="button" className="btn danger" onClick={onCancel}>
              Cancel
            </button>
          )}
        </div>

        <button
          type="button"
          className="prompt-toggle"
          onClick={() => setPromptOpen((open) => !open)}
        >
          {promptOpen ? "Hide prompt override" : "Optional prompt override"}
        </button>
        {promptOpen && (
          <textarea
            className="prompt-box"
            rows={4}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Leave empty to use the model default prompt"
            disabled={running}
          />
        )}

        <p className="status" aria-live="polite">
          {status}
          {pages.length > 0 ? ` · ${pages.filter((p) => p.status === "complete").length}/${pages.length || "?"} pages` : ""}
        </p>

        <ApiDocs />
      </header>

      <main className="workspace">
        <section className="pane preview-pane">
          <div className="pane-label">Document</div>
          {previewSrc && !isPdf ? (
            <img src={previewSrc} alt="Document preview" className="preview-image" />
          ) : file && isPdf ? (
            <div className="preview-fallback">
              <strong>{file.name}</strong>
              <span>PDF — page previews appear while streaming</span>
              {preview ? <img src={preview} alt="Current PDF page" className="preview-image" /> : null}
            </div>
          ) : (
            <div className="preview-fallback">Drop a form image or PDF to start</div>
          )}
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example.file}
                type="button"
                className="example-chip"
                onClick={() => loadExample(example.file)}
                disabled={running}
              >
                {example.label}
              </button>
            ))}
          </div>
        </section>

        <section className="pane result-pane">
          <div className="result-toolbar">
            <div className="pane-label">Result</div>
            <div className="view-toggle">
              <button
                type="button"
                className={view === "render" ? "active" : ""}
                onClick={() => setView("render")}
              >
                Rendered
              </button>
              <button
                type="button"
                className={view === "source" ? "active" : ""}
                onClick={() => setView("source")}
              >
                Markdown
              </button>
            </div>
          </div>
          {view === "source" ? (
            <pre className="markdown-source">{markdown || "Markdown will stream here."}</pre>
          ) : (
            <div
              ref={renderRef}
              className="markdown-render"
              dangerouslySetInnerHTML={{
                __html: renderMarkdown || "<p class='muted'>Rendered output will appear here.</p>",
              }}
            />
          )}
        </section>
      </main>
    </div>
  );
}
