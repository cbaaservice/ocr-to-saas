/// <reference types="vite/client" />

export type ModelChoice = "OvisOCR2" | "GLM-OCR";

export type PageResult = {
  page_number: number;
  markdown: string;
  render_markdown: string;
  status: string;
  elapsed_seconds: number;
};

export type OcrPayload = {
  event: string;
  markdown: string;
  render_markdown: string;
  pages: PageResult[];
  current_page: number;
  total_pages: number;
  document_type: string;
  page_preview?: string | null;
  batch_complete?: boolean;
  batch_start_page?: number | null;
  batch_end_page?: number | null;
  char_count: number;
  elapsed_seconds: number;
  model: string;
  model_choice: ModelChoice;
  backend: string;
  mode: string;
};

export type GradioFileData = {
  path: string;
  url?: string | null;
  orig_name?: string;
  size?: number;
  mime_type?: string;
  meta: { _type: "gradio.FileData" };
};

declare global {
  interface Window {
    MathJax?: {
      startup?: { promise?: Promise<void> };
      typesetPromise?: (elements?: HTMLElement[]) => Promise<void>;
    };
  }
}
