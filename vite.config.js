import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    strictPort: true,
    open: true,
    host: true,
    proxy: {
      // Not used directly since we'll call http://localhost:8000, but can be enabled if desired
      // '/api': 'http://localhost:8000'
    }
  }
})


