import { describe, it, expect } from 'vitest'
import { cosine } from '../lib/similarity.js'

describe('cosine', () => {
  it('returns 1 for identical vectors', () => {
    expect(cosine([1, 0, 0], [1, 0, 0])).toBeCloseTo(1)
  })

  it('returns 0 for orthogonal vectors', () => {
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0)
  })

  it('returns -1 for opposite vectors', () => {
    expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1)
  })

  it('returns 1 for scalar multiples', () => {
    expect(cosine([1, 2, 3], [2, 4, 6])).toBeCloseTo(1)
  })

  it('returns 0 for zero vector (not NaN)', () => {
    expect(cosine([0, 0, 0], [1, 2, 3])).toBe(0)
  })

  it('returns a value between -1 and 1 for arbitrary vectors', () => {
    const result = cosine([0.5, -0.3, 0.8, 0.1], [0.2, 0.7, -0.4, 0.6])
    expect(result).toBeGreaterThanOrEqual(-1)
    expect(result).toBeLessThanOrEqual(1)
  })
})
