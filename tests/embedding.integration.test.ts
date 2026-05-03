import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'
import { GemmaEmbedding } from '../lib/embedding.js'
import { cosine } from '../lib/similarity.js'

describe('GemmaEmbedding', { timeout: 600_000 }, () => {
  let embedding: GemmaEmbedding

  beforeAll(async () => {
    embedding = new GemmaEmbedding()
    await embedding.load()
  }, 600_000)

  afterAll(async () => {
    await embedding.unload()
  })

  it('embeds a document and returns a number array', async () => {
    const vector = await embedding.embed('Hello world')
    expect(Array.isArray(vector)).toBe(true)
    expect(vector.length).toBeGreaterThan(0)
    expect(typeof vector[0]).toBe('number')
  })

  it('returns the same vector for the same text', async () => {
    const first = await embedding.embed('deterministic output')
    const second = await embedding.embed('deterministic output')
    expect(first).toEqual(second)
  })

  it('query and document produce different vectors for same text', async () => {
    const docVec = await embedding.embed('TypeScript programming', 'document')
    const queryVec = await embedding.embed('TypeScript programming', 'query')
    expect(docVec).not.toEqual(queryVec)
  })

  it('related texts score higher than unrelated', async () => {
    const cat = await embedding.embed('The cat sat on the mat')
    const kitten = await embedding.embed('A kitten was resting on the rug')
    const quantum = await embedding.embed('Quantum physics explains entanglement')

    const catKitten = cosine(cat, kitten)
    const catQuantum = cosine(cat, quantum)

    expect(catKitten).toBeGreaterThan(catQuantum)
  })

  it('embedBatch returns correct number of vectors', async () => {
    const vectors = await embedding.embedBatch(['hello', 'world', 'test'])
    expect(vectors).toHaveLength(3)
    vectors.forEach(v => expect(v.length).toBeGreaterThan(0))
  })

  it('load is idempotent', async () => {
    await embedding.load() // second call should be a no-op
    expect(embedding.isLoaded()).toBe(true)
  })

  it('throws when embedding before load', async () => {
    const fresh = new GemmaEmbedding()
    await expect(fresh.embed('hello')).rejects.toThrow('Model not loaded')
  })

  it('unload resets state', async () => {
    const temp = new GemmaEmbedding()
    await temp.load()
    expect(temp.isLoaded()).toBe(true)
    await temp.unload()
    expect(temp.isLoaded()).toBe(false)
  })

  it('reports progress during load', async () => {
    const onProgress = vi.fn()
    const temp = new GemmaEmbedding({ onProgress })
    await temp.load()
    expect(onProgress).toHaveBeenCalledWith(expect.objectContaining({ status: 'ready' }))
    await temp.unload()
  })

})
