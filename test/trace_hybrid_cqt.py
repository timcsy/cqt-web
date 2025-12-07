#!/usr/bin/env python3
"""Trace librosa hybrid_cqt step by step."""

import numpy as np
import librosa
# from librosa.core.constantq import __cqt_filter_fft, __early_downsample_count

# Parameters
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

def main():
    print("=" * 70)
    print("librosa hybrid_cqt detailed trace")
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

    print(f"\nAudio: {len(y)} samples")

    # Key parameters from librosa
    filter_scale = 1  # default
    norm = 1  # L1 normalization
    window = 'hann'
    gamma = 0  # for constant-Q (not VQT)

    # Compute Q
    Q = float(filter_scale) / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    print(f"Q factor: {Q:.10f}")

    # Number of octaves
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    print(f"Number of octaves: {n_octaves}")

    # Frequencies
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)

    # Filter lengths at original SR
    lengths = np.ceil(Q * SR / freqs).astype(int)

    # Early downsample count
    # librosa determines how much to downsample before processing
    # Based on the longest filter length needed

    # From librosa source: __early_downsample_count
    # Returns the number of early downsampling stages to use

    # For hybrid_cqt, it processes from HIGHEST octave to LOWEST
    # Starting with the full audio

    print("\n" + "-" * 70)
    print("Processing order (from librosa hybrid_cqt source):")
    print("-" * 70)

    # librosa hybrid_cqt processes octaves from n_octaves-1 down to 0
    # BUT the bin assignment is:
    # - Octave 0 = bins 0 to bpo-1 (lowest freq)
    # - Octave n-1 = bins (n-1)*bpo to end (highest freq)

    # The audio is progressively downsampled AFTER each octave

    # Let's trace through:
    y_oct = y.copy()
    sr_oct = SR
    hop_oct = HOP_LENGTH

    print("\nTracing hybrid_cqt processing:")
    print(f"Starting with {len(y_oct)} samples at SR={sr_oct}\n")

    # Process from highest octave to lowest
    for i, oct in enumerate(range(n_octaves - 1, -1, -1)):
        bin_start = oct * BINS_PER_OCTAVE
        bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)
        n_bins_oct = bin_end - bin_start

        # Filter lengths at current SR
        lengths_oct = np.ceil(Q * sr_oct / freqs[bin_start:bin_end]).astype(int)

        # Number of frames
        n_frames = int(np.ceil(len(y_oct) / hop_oct))

        print(f"Step {i}: Processing octave {oct}")
        print(f"  Bins: {bin_start}-{bin_end-1} ({n_bins_oct} bins)")
        print(f"  Frequencies: {freqs[bin_start]:.2f}-{freqs[bin_end-1]:.2f} Hz")
        print(f"  Current SR: {sr_oct}")
        print(f"  Current hop: {hop_oct}")
        print(f"  Audio length: {len(y_oct)}")
        print(f"  Number of frames: {n_frames}")
        print(f"  Filter lengths at SR={sr_oct}: {lengths_oct.min()}-{lengths_oct.max()}")
        print()

        # Downsample for next octave (lower freq)
        if oct > 0:
            # Decimate by 2
            y_oct = librosa.resample(y_oct, orig_sr=sr_oct, target_sr=sr_oct//2, res_type='soxr_hq')
            sr_oct = sr_oct // 2
            hop_oct = hop_oct // 2
            if hop_oct < 1:
                hop_oct = 1

    # Now let's verify by computing actual CQT
    print("\n" + "=" * 70)
    print("Verification with actual CQT computation:")
    print("=" * 70)

    cqt = librosa.cqt(
        y, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_mag = np.abs(cqt).T

    print(f"\nCQT shape: {cqt_mag.shape}")
    print(f"\nFrame 10, peaks at C4, E4, G4:")
    print(f"  Bin 126 (C4={freqs[126]:.2f} Hz): {cqt_mag[10, 126]:.6f}")
    print(f"  Bin 138 (E4={freqs[138]:.2f} Hz): {cqt_mag[10, 138]:.6f}")
    print(f"  Bin 147 (G4={freqs[147]:.2f} Hz): {cqt_mag[10, 147]:.6f}")

    # Check which octave these bins belong to
    print(f"\n  Bin 126 is in octave {126 // BINS_PER_OCTAVE}")
    print(f"  Bin 138 is in octave {138 // BINS_PER_OCTAVE}")
    print(f"  Bin 147 is in octave {147 // BINS_PER_OCTAVE}")

if __name__ == "__main__":
    main()
