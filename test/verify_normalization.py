#!/usr/bin/env python3
"""
Verify the correct normalization for CQT.
"""

import numpy as np
import librosa
from scipy import signal

# Parameters
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = 23.12

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

def main():
    print("=" * 70)
    print("Verify normalization")
    print("=" * 70)

    audio = generate_test_audio()

    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    frequencies = FMIN * (2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE))

    # Focus on octave 3 (C4 at bin 126)
    oct = 3
    decimations = n_octaves - 1 - oct

    # Decimate audio
    current_audio = audio.astype(np.float64)
    for _ in range(decimations):
        current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)

    effective_sr = SR / (2 ** decimations)
    current_hop = HOP_LENGTH // (2 ** decimations)
    n_fft = 512

    # Filter info for bin 126
    freq = frequencies[126]
    length = int(np.ceil(Q * effective_sr / freq))
    length_orig = Q * SR / freq  # length at original SR

    print(f"Bin 126 (C4):")
    print(f"  Frequency: {freq:.2f} Hz")
    print(f"  Filter length at effective SR: {length}")
    print(f"  Filter length at original SR: {length_orig:.2f}")
    print(f"  VQT scale (sqrt(2^{decimations})): {np.sqrt(2**decimations):.4f}")

    # Build kernel
    kernel = np.zeros(n_fft, dtype=np.complex128)
    for n in range(length):
        t = (n - (length - 1) / 2.0) / effective_sr
        phase = 2.0 * np.pi * freq * t
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1))) if length > 1 else 1.0
        kernel[n] = np.exp(1j * phase) * w

    kernel /= np.sum(np.abs(kernel))
    kernel_fft = np.fft.fft(kernel)
    kernel_fft *= length / n_fft

    # Compute STFT
    stft = librosa.stft(current_audio, n_fft=n_fft, hop_length=current_hop,
                        window='hann', center=True)

    # Frame 10
    frame = 10
    frame_stft = stft[:, frame]

    # Reconstruct full FFT
    n_pos = n_fft // 2 + 1
    stft_full = np.zeros(n_fft, dtype=np.complex128)
    stft_full[:n_pos] = frame_stft
    if n_fft > 2:
        stft_full[n_pos:] = np.conj(frame_stft[-2:0:-1])

    # Dot product
    response = np.sum(stft_full * np.conj(kernel_fft))

    print(f"\n=== Normalization trace ===")
    print(f"1. Raw dot product: {np.abs(response):.6f}")

    # Apply VQT scale
    vqt_scale = np.sqrt(2 ** decimations)
    response_vqt = response * vqt_scale
    print(f"2. After VQT scale: {np.abs(response_vqt):.6f}")

    # librosa's final normalization: divide by sqrt(lengths_at_original_sr)
    response_final = response_vqt / np.sqrt(length_orig)
    print(f"3. After /sqrt(length_orig): {np.abs(response_final):.6f}")

    # Actual librosa
    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    print(f"\nActual librosa: {np.abs(cqt_librosa[126, 10]):.6f}")

    # Check what the C++ code currently does
    print("\n=== What C++ code does (currently) ===")
    # C++ multiplies by sqrt(filter_len_dec), not divides by sqrt(filter_len_orig)
    response_cpp_style = np.abs(response_vqt) * np.sqrt(length)
    print(f"C++ style (*sqrt(length_dec)): {response_cpp_style:.6f}")

    # The correct formula
    print("\n=== Correct formula ===")
    print("librosa uses: result = (dot_product * vqt_scale) / sqrt(length_orig)")
    print(f"Which gives: {np.abs(response_final):.6f}")

    # Still not matching! Let me trace more carefully...
    print("\n=== Debugging the remaining discrepancy ===")
    print(f"Our result: {np.abs(response_final):.6f}")
    print(f"librosa result: {np.abs(cqt_librosa[126, 10]):.6f}")
    print(f"Ratio: {np.abs(cqt_librosa[126, 10]) / np.abs(response_final):.6f}")

    # Let me check if librosa uses different resampling
    print("\n=== Checking librosa's resampling ===")
    # librosa uses soxr_hq by default
    current_audio_librosa = librosa.resample(audio.astype(np.float64), orig_sr=SR, target_sr=effective_sr, res_type='soxr_hq')
    print(f"scipy.signal.decimate length: {len(current_audio)}")
    print(f"librosa.resample length: {len(current_audio_librosa)}")

    # Recompute with librosa's resampling
    stft_librosa = librosa.stft(current_audio_librosa, n_fft=n_fft, hop_length=current_hop,
                                window='hann', center=True)
    frame_stft_librosa = stft_librosa[:, frame]

    stft_full_librosa = np.zeros(n_fft, dtype=np.complex128)
    stft_full_librosa[:n_pos] = frame_stft_librosa
    if n_fft > 2:
        stft_full_librosa[n_pos:] = np.conj(frame_stft_librosa[-2:0:-1])

    response_librosa_resample = np.sum(stft_full_librosa * np.conj(kernel_fft))
    response_librosa_resample_vqt = response_librosa_resample * vqt_scale
    response_librosa_resample_final = response_librosa_resample_vqt / np.sqrt(length_orig)

    print(f"\nWith librosa resampling:")
    print(f"  Our result: {np.abs(response_librosa_resample_final):.6f}")
    print(f"  librosa result: {np.abs(cqt_librosa[126, 10]):.6f}")
    print(f"  Ratio: {np.abs(cqt_librosa[126, 10]) / np.abs(response_librosa_resample_final):.6f}")

if __name__ == "__main__":
    main()
