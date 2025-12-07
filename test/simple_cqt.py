#!/usr/bin/env python3
"""Simple CQT implementation to understand the exact algorithm."""

import numpy as np
import librosa
from scipy import signal

# Parameters
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

def simple_cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS,
               bins_per_octave=BINS_PER_OCTAVE):
    """Simple single-rate CQT implementation."""
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1)
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    # Number of frames
    n_frames = int(np.ceil(len(y) / hop_length))

    # Result
    result = np.zeros((n_frames, n_bins), dtype=np.complex128)

    for k in range(n_bins):
        freq = freqs[k]
        length = int(np.ceil(Q * sr / freq))

        # Create kernel: complex exponential * Hann window
        t = np.arange(length, dtype=np.float64)
        t = (t - (length - 1) / 2) / sr
        kernel = np.exp(2j * np.pi * freq * t)
        window = np.hanning(length)
        kernel *= window

        # L1 normalize
        kernel /= np.sum(np.abs(kernel))

        # Convolve with audio (using fftconvolve for efficiency)
        # kernel[::-1].conj() for correlation instead of convolution
        conv_result = signal.fftconvolve(y, kernel[::-1].conj(), mode='same')

        # Sample at hop intervals
        for frame in range(n_frames):
            center = frame * hop_length
            if center < len(conv_result):
                result[frame, k] = conv_result[center]

    # Normalize by sqrt(filter_length)
    lengths = np.ceil(Q * sr / freqs).astype(int)
    result /= np.sqrt(lengths)[np.newaxis, :]

    return np.abs(result)

def main():
    print("=" * 70)
    print("Simple CQT vs librosa")
    print("=" * 70)

    # Generate test audio
    duration = 2.0
    t = np.arange(int(SR * duration)) / SR
    c4, e4, g4 = 261.63, 329.63, 392.00
    y = (
        0.3 * np.sin(2 * np.pi * c4 * t) +
        0.3 * np.sin(2 * np.pi * e4 * t) +
        0.3 * np.sin(2 * np.pi * g4 * t)
    ).astype(np.float32)

    # Compute simple CQT
    print("Computing simple CQT...")
    simple_result = simple_cqt(y)
    print(f"Simple CQT shape: {simple_result.shape}")

    # Compute librosa CQT
    print("Computing librosa CQT...")
    librosa_cqt = librosa.cqt(
        y, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    librosa_result = np.abs(librosa_cqt).T
    print(f"librosa CQT shape: {librosa_result.shape}")

    # Compare
    print("\n" + "-" * 70)
    print("Comparison:")
    print("-" * 70)

    diff = np.abs(simple_result - librosa_result)
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    print(f"RMS difference: {np.sqrt((diff**2).mean()):.6f}")

    # Check specific bins
    print("\nFrame 10, bin 126 (C4):")
    print(f"  Simple: {simple_result[10, 126]:.6f}")
    print(f"  librosa: {librosa_result[10, 126]:.6f}")
    print(f"  Ratio: {simple_result[10, 126] / librosa_result[10, 126]:.6f}")

    print("\nFrame 10, bin 147 (G4):")
    print(f"  Simple: {simple_result[10, 147]:.6f}")
    print(f"  librosa: {librosa_result[10, 147]:.6f}")
    print(f"  Ratio: {simple_result[10, 147] / librosa_result[10, 147]:.6f}")

    # Find per-frame ratio for bins with signal
    print("\n" + "-" * 70)
    print("Per-bin ratios (where both > 0.1):")
    print("-" * 70)

    for k in [126, 138, 147]:
        if librosa_result[10, k] > 0.1:
            ratio = simple_result[10, k] / librosa_result[10, k]
            print(f"Bin {k}: simple={simple_result[10, k]:.6f}, librosa={librosa_result[10, k]:.6f}, ratio={ratio:.6f}")

if __name__ == "__main__":
    main()
