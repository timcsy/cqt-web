#!/usr/bin/env python3
"""
Compare C++ logic with librosa step by step.
Focus on the exact computation differences.
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
    print("Debug C++ vs librosa: Step by Step")
    print("=" * 70)

    audio = generate_test_audio()

    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    frequencies = FMIN * (2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE))

    print(f"Q = {Q:.6f}")
    print(f"n_octaves = {n_octaves}")

    # Focus on octave 3 (contains C4 at bin 126)
    oct = 3
    decimations = n_octaves - 1 - oct  # 4 decimations

    bin_start = oct * BINS_PER_OCTAVE  # 108
    bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)  # 144
    oct_freqs = frequencies[bin_start:bin_end]

    print(f"\nOctave {oct}: bins {bin_start}-{bin_end-1}")
    print(f"Decimations: {decimations}")

    # Decimate audio
    current_audio = audio.astype(np.float64)
    current_hop = HOP_LENGTH
    for _ in range(decimations):
        current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)
        current_hop = current_hop // 2

    effective_sr = SR / (2 ** decimations)
    print(f"Effective SR: {effective_sr}")
    print(f"Audio length after decimation: {len(current_audio)}")
    print(f"Current hop: {current_hop}")

    # Filter lengths at effective SR
    oct_lengths = np.ceil(Q * effective_sr / oct_freqs).astype(int)
    max_length = np.max(oct_lengths)
    n_fft = int(2 ** np.ceil(np.log2(max_length)))

    print(f"Filter lengths: {oct_lengths[0]} - {oct_lengths[-1]}")
    print(f"n_fft: {n_fft}")

    # C4 is at local index 18 (126 - 108)
    local_idx = 18
    freq = oct_freqs[local_idx]
    length = oct_lengths[local_idx]

    print(f"\n--- Bin 126 (C4, local idx {local_idx}) ---")
    print(f"Frequency: {freq:.2f} Hz")
    print(f"Filter length at effective SR: {length}")

    # Method 1: C++ style (my implementation)
    print("\n=== Method 1: C++ style (frame-by-frame FFT correlation) ===")

    # Build kernel (C++ style)
    kernel_cpp = np.zeros(n_fft, dtype=np.complex128)
    for n in range(length):
        t = (n - (length - 1) / 2.0) / effective_sr
        phase = 2.0 * np.pi * freq * t
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1))) if length > 1 else 1.0
        kernel_cpp[n] = np.exp(1j * phase) * window

    l1_norm = np.sum(np.abs(kernel_cpp))
    kernel_cpp /= l1_norm
    kernel_fft_cpp = np.fft.fft(kernel_cpp)
    kernel_fft_cpp *= length / n_fft

    print(f"L1 norm of kernel: {l1_norm:.6f}")
    print(f"Kernel FFT at DC: {np.abs(kernel_fft_cpp[0]):.6f}")

    # Compute at frame 10
    frame = 10
    center = frame * current_hop

    # Extract frame data
    frame_data = np.zeros(n_fft, dtype=np.complex128)
    for j in range(n_fft):
        idx = center - n_fft // 2 + j
        if 0 <= idx < len(current_audio):
            frame_data[j] = current_audio[idx]

    frame_fft = np.fft.fft(frame_data)
    product = frame_fft * np.conj(kernel_fft_cpp)
    result_ifft = np.fft.ifft(product)
    cpp_raw = result_ifft[0]

    print(f"Frame {frame}, center={center}")
    print(f"Raw response (index 0): {np.abs(cpp_raw):.6f}")

    # Apply VQT scale
    vqt_scale = np.sqrt(2.0 ** decimations)
    cpp_vqt = cpp_raw * vqt_scale
    print(f"With VQT scale ({vqt_scale:.4f}): {np.abs(cpp_vqt):.6f}")

    # Apply final normalization (multiply by sqrt(filter_len_dec))
    filter_len_dec = length  # Already at decimated SR
    cpp_final = np.abs(cpp_vqt) * np.sqrt(filter_len_dec)
    print(f"With sqrt(filter_len={length}): {cpp_final:.6f}")

    # Method 2: librosa style (STFT-based)
    print("\n=== Method 2: librosa style (STFT-based) ===")

    # Build kernel (same as C++)
    kernel_librosa = np.zeros(n_fft, dtype=np.complex128)
    for n in range(length):
        t = (n - (length - 1) / 2.0) / effective_sr
        phase = 2.0 * np.pi * freq * t
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1))) if length > 1 else 1.0
        kernel_librosa[n] = np.exp(1j * phase) * window

    kernel_librosa /= np.sum(np.abs(kernel_librosa))
    kernel_fft_librosa = np.fft.fft(kernel_librosa)
    kernel_fft_librosa *= length / n_fft

    # Use librosa STFT
    stft = librosa.stft(current_audio, n_fft=n_fft, hop_length=current_hop,
                        window='hann', center=True)

    print(f"STFT shape: {stft.shape}")

    # Get frame column
    frame_stft = stft[:, frame]
    print(f"Frame STFT shape: {frame_stft.shape}")

    # Reconstruct full FFT from half STFT
    n_pos = n_fft // 2 + 1
    stft_full = np.zeros(n_fft, dtype=np.complex128)
    stft_full[:n_pos] = frame_stft
    if n_fft > 2:
        stft_full[n_pos:] = np.conj(frame_stft[-2:0:-1])

    # Dot product with kernel
    response = np.sum(stft_full * np.conj(kernel_fft_librosa))
    print(f"STFT-based response: {np.abs(response):.6f}")

    # Apply VQT scale
    response_vqt = response * vqt_scale
    print(f"With VQT scale: {np.abs(response_vqt):.6f}")

    # librosa's final normalization: divide by sqrt(length_at_original_sr)
    length_orig = Q * SR / freq
    response_final = response_vqt / np.sqrt(length_orig)
    print(f"With /sqrt(length_orig={length_orig:.2f}): {np.abs(response_final):.6f}")

    # Compare with actual librosa
    print("\n=== Actual librosa output ===")
    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    print(f"librosa value at bin 126, frame 10: {np.abs(cqt_librosa[126, 10]):.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"C++ style (my impl): {cpp_final:.6f}")
    print(f"librosa style: {np.abs(response_final):.6f}")
    print(f"Actual librosa: {np.abs(cqt_librosa[126, 10]):.6f}")

    print("\n=== Key differences ===")
    print("1. C++ uses frame-by-frame FFT; librosa uses STFT")
    print("2. C++ normalizes by * sqrt(filter_len_dec); librosa uses / sqrt(filter_len_orig)")
    print(f"   filter_len_dec = {length}, filter_len_orig = {length_orig:.2f}")
    print(f"   sqrt ratio = {np.sqrt(length_orig) / np.sqrt(length):.6f}")

    # Test if using librosa's normalization fixes C++ approach
    print("\n=== Testing fix for C++ ===")
    cpp_fixed = np.abs(cpp_vqt) / np.sqrt(length_orig)
    print(f"C++ with /sqrt(length_orig): {cpp_fixed:.6f}")

if __name__ == "__main__":
    main()
