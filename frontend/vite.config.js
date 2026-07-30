import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API routes → the FastAPI backend (same paths the
// production single-origin build uses), so the frontend code is identical in both.
const API = "^/(agent|analyze|compose|boards|health)";
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      [API]: { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
