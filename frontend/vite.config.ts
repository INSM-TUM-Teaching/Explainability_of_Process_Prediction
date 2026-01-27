import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"

export default defineConfig({
  plugins: [react()],
  // Avoid writing temporary build artifacts under node_modules (can be blocked on some systems).
  cacheDir: ".vite",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
