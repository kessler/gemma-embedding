export type DeviceType = 'cpu' | 'wasm' | 'webgpu'

export interface GemmaEmbeddingOptions {
  /** Local filesystem path or custom HuggingFace model ID. Overrides the default remote model */
  modelPath?: string
  /** Device for inference. Defaults to 'cpu' (Node.js) or 'wasm' (browser) when omitted */
  device?: DeviceType
  /** Quantization type. Defaults to 'fp32' (Node.js/cpu) or 'q8' (browser/wasm) when omitted. Note: 'q4f16' is WebGPU-only */
  dtype?: string
  /** Progress callback during model download/load */
  onProgress?: (info: ProgressInfo) => void
}

export interface ProgressInfo {
  status: 'loading' | 'ready' | 'error'
  /** Overall progress percentage (0-100) */
  progress?: number
  /** Current file being downloaded */
  file?: string
  /** Error message if status is 'error' */
  error?: string
}

export type EmbedMode = 'query' | 'document'
