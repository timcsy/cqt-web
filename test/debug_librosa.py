#!/usr/bin/env python3
"""Debug librosa CQT to understand exact algorithm."""

import numpy as np
import librosa

# Parameters
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

def main():
    print("=" * 70)
    print("librosa CQT Debug")
    print("=" * 70)

    # Q factor
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    print(f"\nQ factor: {Q:.10f}")

    # Frequencies
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    print(f"Fmin: {FMIN:.10f} Hz")
    print(f"Frequency range: {freqs[0]:.10f} - {freqs[-1]:.10f} Hz")

    # Filter lengths at original SR
    lengths = np.ceil(Q * SR / freqs).astype(int)
    print(f"\nFilter lengths at SR={SR}:")
    print(f"  Bin 0 (lowest freq): {lengths[0]}")
    print(f"  Bin 287 (highest freq): {lengths[-1]}")

    # Number of octaves
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    print(f"\nNumber of octaves: {n_octaves}")

    # librosa processes octaves differently
    # Let's trace through the hybrid_cqt implementation
    print("\n" + "-" * 70)
    print("Octave-by-octave analysis (librosa style):")
    print("-" * 70)

    # In librosa hybrid_cqt:
    # - It processes from LOWEST to HIGHEST frequency
    # - For each octave, it decimates the audio BEFORE processing
    # - The filter lengths are computed at the effective (decimated) SR

    for oct in range(n_octaves):
        bin_start = oct * BINS_PER_OCTAVE
        bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)
        n_bins_oct = bin_end - bin_start

        freq_min = freqs[bin_start]
        freq_max = freqs[bin_end - 1]

        # In librosa, the audio is decimated by 2^oct for octave oct
        # (octave 0 = lowest freq, no decimation yet... wait, that's not right either)

        # Let me check the actual librosa source
        # hybrid_cqt processes from n_octaves-1 down to 0
        # But the bin mapping is different

        # Actually in librosa:
        # - octave 0 = bins 0 to bpo-1 (lowest freq)
        # - octave 1 = bins bpo to 2*bpo-1
        # etc.

        # The effective SR for octave oct is: SR / 2^oct
        effective_sr = SR / (2 ** oct)
        length_min_eff = int(np.ceil(Q * effective_sr / freq_max))
        length_max_eff = int(np.ceil(Q * effective_sr / freq_min))

        print(f"Octave {oct}: bins {bin_start:3d}-{bin_end-1:3d}, "
              f"freq {freq_min:.2f}-{freq_max:.2f} Hz, "
              f"eff SR={effective_sr:.1f}, "
              f"filter len {length_min_eff}-{length_max_eff}")

    # Generate test audio
    print("\n" + "-" * 70)
    print("CQT computation:")
    print("-" * 70)

    duration = 2.0
    t = np.arange(int(SR * duration)) / SR
    c4, e4, g4 = 261.63, 329.63, 392.00
    audio = (
        0.3 * np.sin(2 * np.pi * c4 * t) +
        0.3 * np.sin(2 * np.pi * e4 * t) +
        0.3 * np.sin(2 * np.pi * g4 * t)
    ).astype(np.float32)

    # Compute CQT
    cqt = librosa.cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_mag = np.abs(cqt).T

    print(f"CQT shape: {cqt_mag.shape}")
    print(f"\nFrame 0, bins 0-9 (lowest freq):")
    for i in range(10):
        print(f"  Bin {i} ({freqs[i]:.2f} Hz): {cqt_mag[0, i]:.10f}")

    print(f"\nFrame 0, bins 126-135 (around C4=261.63 Hz):")
    for i in range(126, 136):
        print(f"  Bin {i} ({freqs[i]:.2f} Hz): {cqt_mag[0, i]:.10f}")

    # Find peaks
    print(f"\nFrame 10, top 10 peaks:")
    frame10 = cqt_mag[10]
    peak_indices = np.argsort(frame10)[::-1][:10]
    for idx in peak_indices:
        print(f"  Bin {idx} ({freqs[idx]:.2f} Hz): {frame10[idx]:.10f}")

    # Check actual filter bank construction
    print("\n" + "-" * 70)
    print("Filter bank analysis:")
    print("-" * 70)

    # For the lowest frequency bin (bin 0), compute expected filter response
    k = 0
    freq = freqs[k]
    length = lengths[k]

    print(f"Bin 0: freq={freq:.4f} Hz, filter length at SR={SR}: {length}")

    # The filter is: exp(2j * pi * freq * t) * window(length)
    # where t is centered: t = (n - (length-1)/2) / sr
    # Then L1 normalized

    t_filt = np.arange(length, dtype=np.float64)
    t_filt -= (length - 1) / 2
    t_filt /= SR

    kernel = np.exp(2j * np.pi * freq * t_filt)
    window = np.hanning(length)
    kernel *= window

    l1_norm_before = np.sum(np.abs(kernel))
    kernel /= l1_norm_before
    l1_norm_after = np.sum(np.abs(kernel))

    print(f"  L1 norm before normalization: {l1_norm_before:.6f}")
    print(f"  L1 norm after normalization: {l1_norm_after:.6f}")
    print(f"  Kernel first 5 values: {kernel[:5]}")

if __name__ == "__main__":
    main()
