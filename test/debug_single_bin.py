#!/usr/bin/env python3
"""Debug single bin computation to match librosa exactly."""

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
    print("Single Bin Debug: Bin 126 (C4 = 261.63 Hz)")
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

    # Compute Q
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    print(f"Q factor: {Q:.10f}")

    # Frequencies
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    print(f"Bin 126 frequency: {freqs[126]:.6f} Hz")

    # Bin 126 is in octave 3 (126 // 36 = 3)
    octave = 126 // BINS_PER_OCTAVE
    n_octaves = 8
    print(f"Bin 126 is in octave {octave}")

    # Number of decimations before this octave is processed
    # Processing order: oct 7, 6, 5, 4, 3, 2, 1, 0
    # Octave 7: 0 decimations
    # Octave 6: 1 decimation
    # ...
    # Octave 3: 4 decimations
    decimations_before = n_octaves - 1 - octave
    print(f"Decimations before octave {octave}: {decimations_before}")

    # Effective SR when processing this octave
    effective_sr = SR / (2 ** decimations_before)
    print(f"Effective SR: {effective_sr:.2f}")

    # Filter length at effective SR
    filter_length = int(np.ceil(Q * effective_sr / freqs[126]))
    print(f"Filter length at effective SR: {filter_length}")

    # Filter length at original SR
    filter_length_orig = int(np.ceil(Q * SR / freqs[126]))
    print(f"Filter length at original SR: {filter_length_orig}")

    # VQT scale factor
    vqt_scale = np.sqrt(2 ** decimations_before)
    print(f"VQT scale factor: {vqt_scale:.6f}")

    print("\n" + "-" * 70)
    print("Manual computation of bin 126, frame 0:")
    print("-" * 70)

    # Decimate audio to the effective SR
    y_dec = y.copy()
    for i in range(decimations_before):
        y_dec = librosa.resample(y_dec, orig_sr=SR // (2**i), target_sr=SR // (2**(i+1)), res_type='soxr_hq')

    print(f"Decimated audio length: {len(y_dec)}")

    # Effective hop length
    effective_hop = HOP_LENGTH // (2 ** decimations_before)
    print(f"Effective hop: {effective_hop}")

    # Build the filter kernel
    freq = freqs[126]
    length = filter_length

    t_kernel = np.arange(length, dtype=np.float64)
    t_kernel -= (length - 1) / 2
    t_kernel /= effective_sr

    # Complex exponential * Hann window
    kernel = np.exp(2j * np.pi * freq * t_kernel)
    window = np.hanning(length)
    kernel *= window

    # L1 normalize
    l1_norm = np.sum(np.abs(kernel))
    kernel /= l1_norm

    print(f"Kernel L1 norm after normalization: {np.sum(np.abs(kernel)):.6f}")

    # n_fft for this computation
    n_fft = int(2 ** np.ceil(np.log2(length)))
    print(f"n_fft: {n_fft}")

    # Zero-pad kernel
    kernel_padded = np.zeros(n_fft, dtype=np.complex128)
    kernel_padded[:length] = kernel

    # FFT of kernel
    kernel_fft = np.fft.fft(kernel_padded)

    # Scale by length / n_fft
    kernel_fft *= length / n_fft

    # Frame 0: extract centered frame
    frame = 0
    center = frame * effective_hop
    frame_data = np.zeros(n_fft, dtype=np.float64)

    for i in range(n_fft):
        idx = center - n_fft // 2 + i
        if 0 <= idx < len(y_dec):
            frame_data[i] = y_dec[idx]

    print(f"Frame data first 10: {frame_data[:10]}")

    # FFT of frame (no windowing of frame - librosa doesn't window the frame)
    frame_fft = np.fft.fft(frame_data)

    # Correlation: sum(frame_fft * conj(kernel_fft))
    corr = np.sum(frame_fft * np.conj(kernel_fft))
    print(f"Correlation (raw): {corr}")
    print(f"Correlation magnitude: {np.abs(corr):.10f}")

    # Apply VQT scale
    corr_scaled = corr * vqt_scale
    print(f"After VQT scale: {np.abs(corr_scaled):.10f}")

    # Normalize by sqrt(filter_length at original SR)
    result = np.abs(corr_scaled) / np.sqrt(filter_length_orig)
    print(f"Final result: {result:.10f}")

    # Compare with librosa
    cqt = librosa.cqt(
        y, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_mag = np.abs(cqt).T

    print(f"\nlibrosa result: {cqt_mag[0, 126]:.10f}")
    print(f"Ratio (manual/librosa): {result / cqt_mag[0, 126]:.6f}")

    # Try without the final normalization
    result_no_norm = np.abs(corr_scaled)
    print(f"\nWithout final norm: {result_no_norm:.10f}")
    print(f"Ratio (no_norm/librosa): {result_no_norm / cqt_mag[0, 126]:.6f}")

    # Try with different normalization
    result_sqrt_eff = np.abs(corr_scaled) / np.sqrt(filter_length)
    print(f"\nWith sqrt(effective filter length): {result_sqrt_eff:.10f}")
    print(f"Ratio: {result_sqrt_eff / cqt_mag[0, 126]:.6f}")

if __name__ == "__main__":
    main()
