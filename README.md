![gemma-embedding](logo.svg)

# @kessler/gemma-embedding

Local vector embeddings using Google's [EmbeddingGemma 300M](https://ai.google.dev/gemma/docs/embeddinggemma) model via ONNX. Works in Node.js and browser.

## Install

```bash
npm install @kessler/gemma-embedding
```

In Node.js, `onnxruntime-node` is automatically installed for faster native inference. Browser environments use WASM instead.

## Usage

```javascript
import { GemmaEmbedding, cosine } from '@kessler/gemma-embedding'

const embedding = new GemmaEmbedding()
await embedding.load()

// Embed documents
const doc1 = await embedding.embed('The cat sat on the mat')
const doc2 = await embedding.embed('A kitten was resting on the rug')
const doc3 = await embedding.embed('Quantum physics explains entanglement')

// Compute similarity
cosine(doc1, doc2) // ~0.85 (related)
cosine(doc1, doc3) // ~0.25 (unrelated)

// Query embedding (asymmetric — use for search queries)
const query = await embedding.embed('small pet on furniture', 'query')
cosine(query, doc1) // high similarity

// Batch embedding
const vectors = await embedding.embedBatch(['hello', 'world'])

// Cleanup
await embedding.unload()
```

## Query vs Document Embedding

The model uses asymmetric prefixes for retrieval tasks:

- **Document** (`'document'`, default): `"title: none | text: <your text>"` — use when indexing content
- **Query** (`'query'`): `"task: search result | query: <your text>"` — use when searching

Embed your corpus with `'document'` mode, then embed search queries with `'query'` mode for best retrieval quality.

## Options

```javascript
const embedding = new GemmaEmbedding({
  // Load from a local path instead of HuggingFace
  modelPath: '/path/to/local/model',

  // Device: 'cpu' (Node.js default), 'wasm' (browser default), 'webgpu'
  device: 'cpu',

  // Quantization: omit for environment default (fp32 on cpu, q8 on wasm)
  // Options: 'fp32', 'q8', 'q4', 'q4f16' (WebGPU-only)
  dtype: 'q8',

  // Progress callback during model download
  onProgress: (info) => {
    if (info.status === 'loading') console.log(`${info.progress}%`)
    if (info.status === 'ready') console.log('Model ready')
    if (info.status === 'error') console.error(info.error)
  },
})
```

## Model Details

- **Model**: [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
- **Parameters**: ~308M
- **Embedding dimensions**: 768
- **Context length**: 2048 tokens
- **Languages**: 100+

## License

Apache-2.0
