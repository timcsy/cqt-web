#!/usr/bin/env python3
"""
Trace librosa's hybrid_cqt step by step to understand the exact algorithm.
"""

import numpy as np
import librosa
from scipy import signal

# Parameters matching our WASM implementation
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = 23.12  # F#0

def generate_test_audio():
    """Generate C major chord."""
    duration = 2.0
    t = np.arange(int(SR * duration)) / SR
    c4, e4, g4 = 261.63, 329.63, 392.00
    audio = (
        0.3 * np.sin(2 * np.pi * c4 * t) +
        0.3 * np.sin(2 * np.pi * e4 * t) +
        0.3 * np.sin(2 * np.pi * g4 * t)
    ).astype(np.float32)
    return audio

def trace_hybrid_cqt():
    """Trace through librosa's hybrid_cqt implementation."""
    print("=" * 70)
    print("Tracing librosa hybrid_cqt")
    print("=" * 70)

    audio = generate_test_audio()
    print(f"\nAudio: {len(audio)} samples, {len(audio)/SR:.1f}s")

    # Compute CQT
    print(f"\nParameters:")
    print(f"  SR: {SR}")
    print(f"  hop_length: {HOP_LENGTH}")
    print(f"  bins_per_octave: {BINS_PER_OCTAVE}")
    print(f"  n_bins: {N_BINS}")
    print(f"  fmin: {FMIN}")

    # Q factor
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    print(f"  Q: {Q:.6f}")

    # Number of octaves
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    print(f"  n_octaves: {n_octaves}")

    # Frequencies for each bin
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    print(f"\nFrequencies:")
    print(f"  Bin 0: {freqs[0]:.4f} Hz")
    print(f"  Bin 126 (C4): {freqs[126]:.4f} Hz")
    print(f"  Bin 138 (E4): {freqs[138]:.4f} Hz")
    print(f"  Bin 147 (G4): {freqs[147]:.4f} Hz")
    print(f"  Bin 287: {freqs[287]:.4f} Hz")

    # Filter lengths at original SR
    filter_lengths = np.ceil(Q * SR / freqs).astype(int)
    print(f"\nFilter lengths at original SR:")
    print(f"  Bin 0: {filter_lengths[0]} samples")
    print(f"  Bin 126: {filter_lengths[126]} samples")
    print(f"  Bin 287: {filter_lengths[287]} samples")

    # Compute hybrid CQT
    print("\n" + "=" * 70)
    print("Computing librosa.hybrid_cqt")
    print("=" * 70)

    cqt = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_mag = np.abs(cqt).T  # (time, freq)

    print(f"\nCQT shape: {cqt_mag.shape}")
    print(f"\nFrame 10 values at expected peaks:")
    print(f"  Bin 126 (C4 261Hz): {cqt_mag[10, 126]:.6f}")
    print(f"  Bin 138 (E4 329Hz): {cqt_mag[10, 138]:.6f}")
    print(f"  Bin 147 (G4 392Hz): {cqt_mag[10, 147]:.6f}")

    # Find actual peaks
    frame10 = cqt_mag[10, :]
    top_indices = np.argsort(frame10)[::-1][:10]
    print(f"\nFrame 10 top 10 bins:")
    for idx in top_indices:
        print(f"  Bin {idx}: {freqs[idx]:.2f} Hz, value={frame10[idx]:.6f}")

    # Now trace the internal VQT processing
    print("\n" + "=" * 70)
    print("Tracing VQT internals")
    print("=" * 70)

    # librosa uses __vqt for multi-rate processing
    # The key is understanding how it processes each octave

    # Octave assignment
    print(f"\nOctave structure (bins_per_octave={BINS_PER_OCTAVE}):")
    for oct in range(n_octaves):
        bin_start = oct * BINS_PER_OCTAVE
        bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)
        print(f"  Octave {oct}: bins {bin_start}-{bin_end-1}, freqs {freqs[bin_start]:.2f}-{freqs[bin_end-1]:.2f} Hz")

    # Processing order in VQT: from highest to lowest frequency octave
    print(f"\nProcessing order (highest freq first):")
    for i in range(n_octaves):
        oct = n_octaves - 1 - i
        bin_start = oct * BINS_PER_OCTAVE
        bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)
        decimations = i
        effective_sr = SR / (2 ** decimations)
        effective_hop = HOP_LENGTH // (2 ** decimations)
        print(f"  Step {i}: Octave {oct}, bins {bin_start}-{bin_end-1}, SR={effective_sr:.0f}, hop={effective_hop}")

    # VQT scaling factor
    print(f"\nVQT scaling factors:")
    for i in range(n_octaves):
        oct = n_octaves - 1 - i
        decimations = i
        scale = np.sqrt(SR / (SR / (2 ** decimations)))
        print(f"  Octave {oct}: sqrt(2^{decimations}) = {scale:.6f}")

    return cqt_mag, freqs

if __name__ == "__main__":
    trace_hybrid_cqt()
