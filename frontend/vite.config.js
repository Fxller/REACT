import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
    server: {
    host: true,
    port: 5173,
    proxy: {
      '/process': {
        target: 'http://orchestrator:8000',
        changeOrigin: true,
        rewrite: (path) => path,
      },
    },
  },
})
