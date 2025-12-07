/**
 * CQT-Web - librosa-compatible CQT implementation in WebAssembly
 *
 * Provides multiple CQT variants matching librosa's implementations:
 * - HybridCQT (hybrid_cqt) - Default, uses early downsampling
 * - StandardCQT (cqt) - Full resolution, no downsampling
 * - PseudoCQT (pseudo_cqt) - STFT-based approximation, fastest
 * - VQT - Variable-Q Transform with gamma parameter
 */

/**
 * Progress callback function type
 * @param progress Progress value between 0.0 and 1.0
 * @param stage Current processing stage description
 */
export type ProgressCallback = (progress: number, stage: string) => void;

/**
 * Base interface for all CQT variants
 */
export interface CQTBase {
  /**
   * Compute CQT magnitude spectrogram
   * @param audio Audio samples as Float32Array
   * @returns Flattened magnitude spectrogram [numFrames * nBins]
   */
  compute(audio: Float32Array): Float32Array;

  /**
   * Compute CQT with progress reporting
   * @param audio Audio samples as Float32Array
   * @param callback Optional progress callback
   * @returns Flattened magnitude spectrogram [numFrames * nBins]
   */
  computeWithProgress(audio: Float32Array, callback: ProgressCallback | null): Float32Array;

  /** Get sample rate */
  getSampleRate(): number;

  /** Get hop length in samples */
  getHopLength(): number;

  /** Get bins per octave */
  getBinsPerOctave(): number;

  /** Get total number of frequency bins */
  getNBins(): number;

  /** Get minimum frequency (Hz) */
  getFmin(): number;

  /** Get Q factor */
  getQ(): number;

  /** Get frequencies for each bin (Hz) */
  getFrequencies(): Float64Array;

  /** Get filter lengths for each bin (samples) */
  getFilterLengths(): Int32Array;

  /**
   * Get output shape for given audio length
   * @param audioLength Number of audio samples
   * @returns [numFrames, nBins]
   */
  getOutputShape(audioLength: number): [number, number];

  /**
   * Release WASM memory
   * Call this when done using the CQT instance
   */
  delete(): void;
}

/**
 * Hybrid CQT matching librosa.hybrid_cqt
 *
 * Uses early downsampling for lower octaves to improve efficiency.
 * This is the default for CNN-LSTM chord recognition model.
 *
 * Default parameters:
 * - sampleRate: 22050
 * - hopLength: 512
 * - binsPerOctave: 36
 * - nBins: 288
 * - fmin: 23.12 (F#0)
 */
export class HybridCQT implements CQTBase {
  /** Create with default parameters (CNN-LSTM model) */
  constructor();
  /** Create with custom parameters */
  constructor(sampleRate: number, hopLength: number, binsPerOctave: number, nBins: number, fmin: number);

  compute(audio: Float32Array): Float32Array;
  computeWithProgress(audio: Float32Array, callback: ProgressCallback | null): Float32Array;
  getSampleRate(): number;
  getHopLength(): number;
  getBinsPerOctave(): number;
  getNBins(): number;
  getFmin(): number;
  getQ(): number;
  getFrequencies(): Float64Array;
  getFilterLengths(): Int32Array;
  getOutputShape(audioLength: number): [number, number];
}

/**
 * Standard CQT matching librosa.cqt
 *
 * No early downsampling - all frequencies processed at original sample rate.
 * More accurate but slower than HybridCQT.
 * Used by BTC chord recognition model.
 *
 * Default parameters:
 * - sampleRate: 22050
 * - hopLength: 2048
 * - binsPerOctave: 24
 * - nBins: 144
 * - fmin: 32.7 (C1)
 */
export class StandardCQT implements CQTBase {
  /** Create with default parameters (BTC model) */
  constructor();
  /** Create with custom parameters */
  constructor(sampleRate: number, hopLength: number, binsPerOctave: number, nBins: number, fmin: number);

  compute(audio: Float32Array): Float32Array;
  computeWithProgress(audio: Float32Array, callback: ProgressCallback | null): Float32Array;
  getSampleRate(): number;
  getHopLength(): number;
  getBinsPerOctave(): number;
  getNBins(): number;
  getFmin(): number;
  getQ(): number;
  getFrequencies(): Float64Array;
  getFilterLengths(): Int32Array;
  getOutputShape(audioLength: number): [number, number];
}

/**
 * Pseudo CQT matching librosa.pseudo_cqt
 *
 * Uses STFT-based approach with frequency mapping.
 * Faster than standard CQT but less accurate at low frequencies.
 */
export class PseudoCQT implements CQTBase {
  /** Create with default parameters */
  constructor();
  /** Create with custom parameters */
  constructor(sampleRate: number, hopLength: number, binsPerOctave: number, nBins: number, fmin: number);

  compute(audio: Float32Array): Float32Array;
  computeWithProgress(audio: Float32Array, callback: ProgressCallback | null): Float32Array;
  getSampleRate(): number;
  getHopLength(): number;
  getBinsPerOctave(): number;
  getNBins(): number;
  getFmin(): number;
  getQ(): number;
  getFrequencies(): Float64Array;
  getFilterLengths(): Int32Array;
  getOutputShape(audioLength: number): [number, number];
}

/**
 * Variable-Q Transform matching librosa.vqt
 *
 * Unlike CQT which has constant Q across all frequencies,
 * VQT allows the Q factor to vary with gamma parameter.
 *
 * Q(k) = Q0 * (1 + gamma / freq(k))
 *
 * When gamma = 0, VQT is equivalent to CQT.
 */
export class VQT implements CQTBase {
  /** Create with default parameters */
  constructor();
  /** Create with custom parameters */
  constructor(sampleRate: number, hopLength: number, binsPerOctave: number, nBins: number, fmin: number, gamma?: number);

  compute(audio: Float32Array): Float32Array;
  computeWithProgress(audio: Float32Array, callback: ProgressCallback | null): Float32Array;
  getSampleRate(): number;
  getHopLength(): number;
  getBinsPerOctave(): number;
  getNBins(): number;
  getFmin(): number;
  getQ(): number;
  /** Get gamma parameter */
  getGamma(): number;
  getFrequencies(): Float64Array;
  getFilterLengths(): Int32Array;
  getOutputShape(audioLength: number): [number, number];
}

/**
 * Backwards compatibility alias for HybridCQT
 */
export class CQT extends HybridCQT {}

/**
 * FFT result object
 */
export interface FFTResult {
  /** Real part of FFT */
  real: Float64Array;
  /** Imaginary part of FFT */
  imag: Float64Array;
}

/**
 * FFT utilities
 */
export namespace FFT {
  /**
   * Compute forward FFT of real signal
   * @param real Real input signal
   * @returns Object with real and imaginary parts
   */
  function forward(real: Float64Array): FFTResult;

  /**
   * Compute inverse FFT
   * @param real Real part of spectrum
   * @param imag Imaginary part of spectrum
   * @returns Real output signal
   */
  function inverse(real: Float64Array, imag: Float64Array): Float64Array;
}

// Utility functions

/**
 * Convert note name to frequency in Hz
 * @param note Note name (e.g., "A4", "C#3", "Bb2")
 * @returns Frequency in Hz, or -1 if invalid
 * @example noteToHz("A4") // returns 440.0
 */
export function noteToHz(note: string): number;

/**
 * Convert MIDI note number to frequency in Hz
 * @param midi MIDI note number (0-127)
 * @returns Frequency in Hz
 * @example midiToHz(69) // returns 440.0
 */
export function midiToHz(midi: number): number;

/**
 * Convert frequency in Hz to MIDI note number
 * @param hz Frequency in Hz
 * @returns MIDI note number (fractional)
 * @example hzToMidi(440.0) // returns 69.0
 */
export function hzToMidi(hz: number): number;

/**
 * Convert frequency in Hz to nearest MIDI note number
 * @param hz Frequency in Hz
 * @returns Nearest MIDI note number (integer)
 */
export function hzToMidiRounded(hz: number): number;

/**
 * Convert frame index to time in seconds
 * @param frames Frame index
 * @param sr Sample rate
 * @param hopLength Hop length in samples
 * @returns Time in seconds
 */
export function framesToTime(frames: number, sr: number, hopLength: number): number;

/**
 * Convert time in seconds to frame index
 * @param time Time in seconds
 * @param sr Sample rate
 * @param hopLength Hop length in samples
 * @returns Frame index
 */
export function timeToFrames(time: number, sr: number, hopLength: number): number;

/**
 * Convert samples to time in seconds
 * @param samples Number of samples
 * @param sr Sample rate
 * @returns Time in seconds
 */
export function samplesToTime(samples: number, sr: number): number;

/**
 * Convert time in seconds to samples
 * @param time Time in seconds
 * @param sr Sample rate
 * @returns Number of samples
 */
export function timeToSamples(time: number, sr: number): number;

/**
 * Convert MIDI note number to note name
 * @param midi MIDI note number
 * @param useSharps Use sharps (true) or flats (false)
 * @returns Note name (e.g., "C4", "F#3", "Bb5")
 * @example midiToNote(69, true) // returns "A4"
 */
export function midiToNote(midi: number, useSharps?: boolean): string;

/**
 * CQT Module interface
 */
export interface CQTModule {
  HybridCQT: typeof HybridCQT;
  StandardCQT: typeof StandardCQT;
  PseudoCQT: typeof PseudoCQT;
  VQT: typeof VQT;
  CQT: typeof CQT;
  FFT: typeof FFT;
  noteToHz: typeof noteToHz;
  midiToHz: typeof midiToHz;
  hzToMidi: typeof hzToMidi;
  hzToMidiRounded: typeof hzToMidiRounded;
  framesToTime: typeof framesToTime;
  timeToFrames: typeof timeToFrames;
  samplesToTime: typeof samplesToTime;
  timeToSamples: typeof timeToSamples;
  midiToNote: typeof midiToNote;
}

/**
 * Create and initialize the CQT WASM module
 * @returns Promise that resolves to the CQT module
 */
export function createCQTModule(): Promise<CQTModule>;

export default createCQTModule;
