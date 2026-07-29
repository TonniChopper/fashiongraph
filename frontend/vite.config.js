import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api → the FastAPI backend, so the frontend can call
// `/api/agent` with no CORS fuss during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
