# Changelog

## 1.0.0 (2026-05-03)

- Initial release
- `GemmaEmbedding` class with `load()`, `embed()`, `embedBatch()`, `unload()`
- Asymmetric query/document embedding modes
- `cosine()` similarity utility
- Support for Node.js (cpu) and browser (wasm/webgpu)
- Optional local model loading via `modelPath`
- Progress callbacks during model download
