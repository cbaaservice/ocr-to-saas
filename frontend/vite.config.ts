import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: {
    port: 5173,
    proxy: {
      "/gradio_api": "http://127.0.0.1:7860",
      "/config": "http://127.0.0.1:7860",
      "/healthz": "http://127.0.0.1:7860",
      "/examples": "http://127.0.0.1:7860",
      "/call": "http://127.0.0.1:7860",
      "/queue": "http://127.0.0.1:7860",
    },
  },
});
