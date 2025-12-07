/**
 * TypeScript type definitions for the CQT WASM module.
 */

export interface CQTModule {
  CQT: CQTConstructor;
}

export interface CQTConstructor {
  new(): CQT;
  new(sampleRate: number, hopLength: number, binsPerOctave: number, nBins: number, fmin: number): CQT;
}

export interface CQT {
  /**
   * Compute CQT magnitude spectrogram.
   * @param audio - Input audio as Float32Array
   * @returns Float32Array of shape [numFrames * nBins] (row-major, frames first)
   */
  compute(audio: Float32Array): Float32Array;

  /** Get sample rate */
  getSampleRate(): number;

  /** Get hop length */
  getHopLength(): number;

  /** Get bins per octave */
  getBinsPerOctave(): number;

  /** Get total number of bins */
  getNBins(): number;

  /** Get minimum frequency (Hz) */
  getFmin(): number;

  /** Get Q factor */
  getQ(): number;

  /** Get frequencies for each bin as Float64Array */
  getFrequencies(): Float64Array;

  /** Get filter lengths for each bin as Int32Array */
  getFilterLengths(): Int32Array;

  /** Get output shape [numFrames, nBins] for given audio length */
  getOutputShape(audioLength: number): [number, number];

  /** Clean up resources */
  delete(): void;
}

/**
 * Factory function to create the CQT module.
 * Call this to initialize the WASM module.
 */
declare function createCQTModule(): Promise<CQTModule>;

export default createCQTModule;
