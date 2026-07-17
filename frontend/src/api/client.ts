import { Client } from "@gradio/client";
import type { GradioFileData, ModelChoice, OcrPayload, PageResult } from "../types";

const PAGES_PER_GPU_REQUEST = 4;

export function appRoot(): string {
  const { origin, pathname } = window.location;
  const trimmed = pathname.replace(/\/+$/, "");
  if (!trimmed || trimmed === "/") {
    return origin;
  }
  return `${origin}${trimmed}`;
}

export async function uploadDocument(file: File): Promise<GradioFileData> {
  const form = new FormData();
  form.append("files", file, file.name);
  const response = await fetch(new URL("gradio_api/upload", appRoot() + "/"), {
    method: "POST",
    body: form,
    credentials: "include",
  });
  if (!response.ok) {
    throw new Error(`Document upload failed (${response.status}).`);
  }
  const payload = (await response.json()) as string[] | { path?: string }[];
  const path =
    typeof payload[0] === "string"
      ? payload[0]
      : (payload[0] as { path?: string })?.path;
  if (!path) {
    throw new Error("Upload did not return a file path.");
  }
  return {
    path,
    orig_name: file.name,
    size: file.size,
    mime_type: file.type || undefined,
    meta: { _type: "gradio.FileData" },
  };
}

function mergePages(existing: PageResult[], incoming: PageResult[]): PageResult[] {
  const byNumber = new Map<number, PageResult>();
  for (const page of existing) {
    byNumber.set(page.page_number, page);
  }
  for (const page of incoming) {
    const prev = byNumber.get(page.page_number);
    if (!prev || page.status === "complete" || (page.markdown?.length ?? 0) >= (prev.markdown?.length ?? 0)) {
      byNumber.set(page.page_number, page);
    }
  }
  return [...byNumber.values()].sort((a, b) => a.page_number - b.page_number);
}

function firstIncompleteIndex(pages: PageResult[], totalPages: number): number {
  for (let i = 1; i <= totalPages; i += 1) {
    const page = pages.find((p) => p.page_number === i);
    if (!page || page.status !== "complete") {
      return i - 1;
    }
  }
  return totalPages;
}

function combineField(pages: PageResult[], field: "markdown" | "render_markdown"): string {
  if (pages.length <= 1) {
    return pages[0]?.[field] ?? "";
  }
  return pages
    .map((page) => `<!-- Page ${page.page_number} -->\n\n${page[field] ?? ""}`)
    .join("\n\n---\n\n");
}

export type RunOcrHandlers = {
  onStatus: (text: string) => void;
  onUpdate: (state: {
    pages: PageResult[];
    markdown: string;
    renderMarkdown: string;
    preview?: string | null;
    totalPages: number;
    currentPage: number;
    charCount: number;
  }) => void;
};

export async function runOcrDocument(options: {
  fileData: GradioFileData;
  modelChoice: ModelChoice;
  prompt: string;
  signal?: AbortSignal;
  handlers: RunOcrHandlers;
}): Promise<void> {
  const { fileData, modelChoice, prompt, signal, handlers } = options;
  const client = await Client.connect(appRoot());
  let pages: PageResult[] = [];
  let totalPages = 1;
  let pageIndex = 0;
  let preview: string | null | undefined;

  while (pageIndex < totalPages) {
    if (signal?.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }

    handlers.onStatus(
      totalPages > 1
        ? `Running pages ${pageIndex + 1}–${Math.min(totalPages, pageIndex + PAGES_PER_GPU_REQUEST)} of ${totalPages}…`
        : "Running OCR…",
    );

    const stream = client.submit("/run_ocr", {
      image_path: fileData,
      page_index: pageIndex,
      page_count: PAGES_PER_GPU_REQUEST,
      model_choice: modelChoice,
      prompt,
    });

    let batchComplete = false;
    let sawData = false;

    for await (const message of stream) {
      if (signal?.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }
      if (message.type === "status") {
        const stage = (message as { stage?: string }).stage ?? "";
        if (stage === "error") {
          throw new Error("OCR job failed in the queue.");
        }
        continue;
      }
      if (message.type !== "data") {
        continue;
      }
      sawData = true;
      const raw = Array.isArray(message.data) ? message.data[0] : message.data;
      const payload = raw as OcrPayload;
      if (!payload || typeof payload !== "object") {
        continue;
      }

      totalPages = payload.total_pages || totalPages;
      if (payload.page_preview) {
        preview = payload.page_preview;
      }
      pages = mergePages(pages, payload.pages ?? []);
      batchComplete = Boolean(payload.batch_complete);

      const label =
        payload.event === "page_start"
          ? `Preparing page ${payload.current_page}/${payload.total_pages}`
          : payload.event === "stream"
            ? `Streaming page ${payload.current_page}/${payload.total_pages} · ${payload.char_count.toLocaleString()} chars`
            : payload.event === "page_complete"
              ? `Page ${payload.current_page}/${payload.total_pages} complete`
              : `Batch complete · ${payload.char_count.toLocaleString()} chars`;

      handlers.onStatus(label);
      handlers.onUpdate({
        pages,
        markdown: combineField(pages, "markdown"),
        renderMarkdown: combineField(pages, "render_markdown"),
        preview,
        totalPages,
        currentPage: payload.current_page,
        charCount: pages.reduce((sum, page) => sum + (page.markdown?.length ?? 0), 0),
      });

      if (batchComplete) {
        break;
      }
    }

    if (!sawData) {
      throw new Error("OCR stream closed without data.");
    }

    const next = firstIncompleteIndex(pages, totalPages);
    if (next >= totalPages) {
      break;
    }
    // Advance to the first unfinished page (batch complete or reconnect).
    if (!batchComplete && next === pageIndex) {
      // No progress — avoid infinite reconnect loops.
      throw new Error(`OCR stalled on page ${pageIndex + 1}.`);
    }
    pageIndex = next;
  }

  handlers.onStatus(
    `Done · ${pages.reduce((s, p) => s + (p.markdown?.length ?? 0), 0).toLocaleString()} chars`,
  );
}
