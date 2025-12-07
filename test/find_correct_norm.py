#!/usr/bin/env python3
"""Find the correct normalization to match librosa."""

import numpy as np
import librosa
from scipy import signal

# Parameters
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

def main():
    print("=" * 70)
    print("Finding Correct Normalization")
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

    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)

    # Compute librosa CQT for reference
    cqt = librosa.cqt(
        y, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_mag = np.abs(cqt).T

    # Test bin 126
    octave = 3
    n_octaves = 8
    decimations = n_octaves - 1 - octave

    # Decimate
    y_dec = y.copy()
    sr_dec = SR
    for i in range(decimations):
        y_dec = librosa.resample(y_dec, orig_sr=sr_dec, target_sr=sr_dec//2, res_type='soxr_hq')
        sr_dec //= 2

    freq = freqs[126]
    length_dec = int(np.ceil(Q * sr_dec / freq))
    length_orig = int(np.ceil(Q * SR / freq))

    # Build kernel WITHOUT L1 normalization
    t_kernel = np.arange(length_dec, dtype=np.float64)
    t_kernel = (t_kernel - (length_dec - 1) / 2) / sr_dec
    kernel = np.exp(2j * np.pi * freq * t_kernel)
    window = np.hanning(length_dec)
    kernel *= window

    # Try different normalizations
    print("\nTrying different normalization strategies:")
    print("-" * 70)

    hop_dec = HOP_LENGTH // (2 ** decimations)
    vqt_scale = np.sqrt(2 ** decimations)

    # 1. No normalization
    kernel1 = kernel.copy()
    result1 = signal.fftconvolve(y_dec, kernel1[::-1].conj(), mode='same')
    val1 = np.abs(result1[10 * hop_dec]) * vqt_scale
    print(f"1. No norm: {val1:.6f}, ratio to librosa: {val1 / cqt_mag[10, 126]:.6f}")

    # 2. L1 norm only
    kernel2 = kernel.copy() / np.sum(np.abs(kernel))
    result2 = signal.fftconvolve(y_dec, kernel2[::-1].conj(), mode='same')
    val2 = np.abs(result2[10 * hop_dec]) * vqt_scale
    print(f"2. L1 norm: {val2:.6f}, ratio to librosa: {val2 / cqt_mag[10, 126]:.6f}")

    # 3. L2 norm
    kernel3 = kernel.copy() / np.sqrt(np.sum(np.abs(kernel)**2))
    result3 = signal.fftconvolve(y_dec, kernel3[::-1].conj(), mode='same')
    val3 = np.abs(result3[10 * hop_dec]) * vqt_scale
    print(f"3. L2 norm: {val3:.6f}, ratio to librosa: {val3 / cqt_mag[10, 126]:.6f}")

    # 4. No kernel norm, divide by sqrt(length_orig)
    kernel4 = kernel.copy()
    result4 = signal.fftconvolve(y_dec, kernel4[::-1].conj(), mode='same')
    val4 = np.abs(result4[10 * hop_dec]) * vqt_scale / np.sqrt(length_orig)
    print(f"4. No kernel norm, /sqrt(len_orig): {val4:.6f}, ratio: {val4 / cqt_mag[10, 126]:.6f}")

    # 5. No kernel norm, divide by length_orig
    val5 = np.abs(result4[10 * hop_dec]) * vqt_scale / length_orig
    print(f"5. No kernel norm, /len_orig: {val5:.6f}, ratio: {val5 / cqt_mag[10, 126]:.6f}")

    # 6. No kernel norm, divide by sqrt(length_dec)
    val6 = np.abs(result4[10 * hop_dec]) * vqt_scale / np.sqrt(length_dec)
    print(f"6. No kernel norm, /sqrt(len_dec): {val6:.6f}, ratio: {val6 / cqt_mag[10, 126]:.6f}")

    # 7. L1 norm, multiply by sqrt(length_orig)
    result7 = signal.fftconvolve(y_dec, kernel2[::-1].conj(), mode='same')
    val7 = np.abs(result7[10 * hop_dec]) * vqt_scale * np.sqrt(length_orig)
    print(f"7. L1 norm, *sqrt(len_orig): {val7:.6f}, ratio: {val7 / cqt_mag[10, 126]:.6f}")

    # 8. L1 norm, multiply by length_dec
    val8 = np.abs(result7[10 * hop_dec]) * vqt_scale * length_dec
    print(f"8. L1 norm, *len_dec: {val8:.6f}, ratio: {val8 / cqt_mag[10, 126]:.6f}")

    # 9. Scale by L1 norm of original window
    l1_norm = np.sum(np.abs(kernel))
    val9 = np.abs(result2[10 * hop_dec]) * vqt_scale * l1_norm / np.sqrt(length_orig)
    print(f"9. L1 norm kernel, *l1_norm/sqrt(len_orig): {val9:.6f}, ratio: {val9 / cqt_mag[10, 126]:.6f}")

    # 10. The librosa way: length normalization (from source)
    # In librosa, the filter is L1 normalized, then the output is scaled by 1/sqrt(length)
    # But wait - let me check what length means here

    print("\n" + "-" * 70)
    print("More analysis:")
    print("-" * 70)

    # What is the actual L1 norm of the Hann-windowed complex exponential?
    print(f"L1 norm of windowed kernel: {l1_norm:.6f}")
    print(f"sqrt(length_dec): {np.sqrt(length_dec):.6f}")
    print(f"sqrt(length_orig): {np.sqrt(length_orig):.6f}")
    print(f"Ratio l1_norm / sqrt(length_dec): {l1_norm / np.sqrt(length_dec):.6f}")

    # The Hann window's L1 norm for length N is approximately N/2
    # So l1_norm ≈ length_dec / 2
    print(f"length_dec / 2: {length_dec / 2:.6f}")
    print(f"Actual l1_norm: {l1_norm:.6f}")

    # 11. Try: raw result * vqt_scale * 2 / sqrt(length_orig)
    val11 = np.abs(result4[10 * hop_dec]) * vqt_scale * 2 / np.sqrt(length_orig)
    print(f"11. No norm, *2*vqt/sqrt(len_orig): {val11:.6f}, ratio: {val11 / cqt_mag[10, 126]:.6f}")

    # 12. L1 norm, multiply by sqrt(length_dec) (undo the extra normalization)
    val12 = np.abs(result2[10 * hop_dec]) * vqt_scale * np.sqrt(length_dec)
    print(f"12. L1 norm, *sqrt(len_dec)*vqt: {val12:.6f}, ratio: {val12 / cqt_mag[10, 126]:.6f}")

    # Hmm, let me think about this differently
    # The observed ratio is about 0.000918, which is close to 1/1089 ≈ 0.000918
    # 1089 = 33^2, and sqrt(4336) ≈ 65.8, so that's not it
    # sqrt(length_orig) = 65.8
    # 1/65.8^2 = 0.000231, not 0.000918

    # Let me compute: what would we need to multiply by?
    needed_scale = cqt_mag[10, 126] / (np.abs(result2[10 * hop_dec]) * vqt_scale)
    print(f"\nNeeded scale factor: {needed_scale:.6f}")
    print(f"sqrt(length_orig): {np.sqrt(length_orig):.6f}")
    print(f"length_orig: {length_orig}")
    print(f"sqrt(length_dec): {np.sqrt(length_dec):.6f}")
    print(f"length_dec: {length_dec}")

    # Check if needed_scale equals length_dec
    print(f"\nRatio needed_scale / length_dec: {needed_scale / length_dec:.6f}")
    print(f"Ratio needed_scale / sqrt(length_dec): {needed_scale / np.sqrt(length_dec):.6f}")
    print(f"Ratio needed_scale / l1_norm: {needed_scale / l1_norm:.6f}")

if __name__ == "__main__":
    main()
