import { AutoModel, AutoTokenizer } from '@huggingface/transformers'
import type { GemmaEmbeddingOptions, ProgressInfo, EmbedMode } from './types.js'

const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX'

export class GemmaEmbedding {
  private model: Awaited<ReturnType<typeof AutoModel.from_pretrained>> | null = null
  private tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null
  private loading = false

  private readonly modelSource: string
  private readonly device?: string
  private readonly dtype?: string
  private readonly onProgress?: (info: ProgressInfo) => void

  readonly dimensions = 768

  constructor(options: GemmaEmbeddingOptions = {}) {
    this.modelSource = options.modelPath ?? MODEL_ID
    this.device = options.device
    this.dtype = options.dtype
    this.onProgress = options.onProgress
  }

  async load(): Promise<void> {
    if (this.model) return
    if (this.loading) return
    this.loading = true

    const fileProgress = new Map<string, { loaded: number; total: number }>()
    let lastReportedProgress = -1

    const progress_callback = (info: { status: string; file?: string; progress?: number; loaded?: number; total?: number }) => {
      if (info.status === 'progress' && info.file != null) {
        fileProgress.set(info.file, { loaded: info.loaded ?? 0, total: info.total ?? 0 })
        let totalBytes = 0, loadedBytes = 0
        for (const entry of fileProgress.values()) {
          totalBytes += entry.total
          loadedBytes += entry.loaded
        }
        const overall = totalBytes > 0 ? Math.round((loadedBytes / totalBytes) * 100) : 0
        if (overall !== lastReportedProgress) {
          lastReportedProgress = overall
          this.onProgress?.({ status: 'loading', progress: overall, file: info.file })
        }
      } else if (info.status === 'ready') {
        this.onProgress?.({ status: 'ready' })
      }
    }

    try {
      const modelOptions: Record<string, unknown> = { progress_callback }
      if (this.dtype) modelOptions.dtype = this.dtype
      if (this.device) modelOptions.device = this.device

      const [model, tokenizer] = await Promise.all([
        AutoModel.from_pretrained(this.modelSource, modelOptions),
        AutoTokenizer.from_pretrained(this.modelSource),
      ])

      this.model = model
      this.tokenizer = tokenizer
      this.loading = false
      this.onProgress?.({ status: 'ready' })
    } catch (e) {
      this.loading = false
      this.onProgress?.({ status: 'error', error: String(e) })
      throw e
    }
  }

  isLoaded(): boolean {
    return this.model !== null
  }

  async unload(): Promise<void> {
    if (this.model) {
      await (this.model as any).dispose()
      this.model = null
    }
    this.tokenizer = null
    this.loading = false
  }

  async embed(text: string, mode: EmbedMode = 'document'): Promise<number[]> {
    if (!this.model || !this.tokenizer) throw new Error('Model not loaded. Call load() first.')

    const prefixed = mode === 'query'
      ? `task: search result | query: ${text}`
      : `title: none | text: ${text}`

    const inputs = await this.tokenizer(prefixed, { padding: true, truncation: true })
    const output = await this.model(inputs)
    return Array.from((output as any).sentence_embedding.data as Float32Array)
  }

  async embedBatch(texts: string[], mode: EmbedMode = 'document'): Promise<number[][]> {
    const results: number[][] = []
    for (const text of texts) {
      results.push(await this.embed(text, mode))
    }
    return results
  }
}
