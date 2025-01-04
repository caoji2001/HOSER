import { resolve } from 'path'

export default {
  root: resolve(__dirname, 'src'),
  base: '/HOSER/',
  build: {
    outDir: '../dist'
  },
  server: {
    port: 8080
  }
}