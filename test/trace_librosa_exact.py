#!/usr/bin/env python3
"""
Trace librosa's exact implementation step by step.
"""

import numpy as np
import librosa
import librosa.core.constantq as cq_module
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
    print("Trace librosa's exact implementation")
    print("=" * 70)

    audio = generate_test_audio()

    # Get the filter scale (default is 1.0)
    filter_scale = 1.0

    # Compute Q
    Q = filter_scale / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    print(f"\nQ = {Q:.6f}")

    # Get frequencies
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)

    # Compute filter lengths at original SR
    lengths = Q * SR / freqs
    print(f"lengths[126] = {lengths[126]:.6f}")

    # Now trace __vqt (the internal VQT function)
    # Key parameters:
    # - sparsity factor (not used in our case)
    # - alpha for tuning (0.0)
    # - window (default 'hann')

    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    print(f"n_octaves = {n_octaves}")

    # librosa's __vqt processes octaves from highest to lowest
    # For each octave, it:
    # 1. Gets the filter bank for that octave
    # 2. Computes STFT response
    # 3. Applies VQT scaling
    # 4. Decimates audio for next octave

    # Let's trace octave 3 specifically (contains C4)
    octave = 3
    decimations = n_octaves - 1 - octave
    print(f"\nOctave {octave}, decimations = {decimations}")

    # Decimate audio
    current_audio = audio.astype(np.float64)
    current_sr = SR
    for _ in range(decimations):
        current_audio = librosa.resample(current_audio, orig_sr=current_sr,
                                         target_sr=current_sr//2, res_type='soxr_hq')
        current_sr //= 2

    print(f"After decimation: len={len(current_audio)}, sr={current_sr}")

    # Get frequencies for this octave
    bin_start = octave * BINS_PER_OCTAVE
    bin_end = min((octave + 1) * BINS_PER_OCTAVE, N_BINS)
    oct_freqs = freqs[bin_start:bin_end]
    print(f"Octave freqs: {oct_freqs[0]:.2f} - {oct_freqs[-1]:.2f} Hz")

    # Get filter lengths at current SR
    oct_lengths = Q * current_sr / oct_freqs
    print(f"Filter lengths at current SR: {oct_lengths[0]:.2f} - {oct_lengths[-1]:.2f}")

    # n_fft for this octave
    n_fft = 2 ** int(np.ceil(np.log2(np.max(oct_lengths))))
    print(f"n_fft = {n_fft}")

    # Now let's build the filter bank like librosa does
    # Using librosa's __cqt_filter_fft equivalent

    filters = []
    for i, (freq, length) in enumerate(zip(oct_freqs, oct_lengths)):
        length_int = int(np.ceil(length))

        # Time array centered
        t = np.arange(length_int, dtype=np.float64)
        t = (t - (length_int - 1) / 2) / current_sr

        # Complex exponential
        kernel = np.exp(2j * np.pi * freq * t)

        # Hann window
        window = np.hanning(length_int)
        kernel *= window

        # L1 normalize
        kernel /= np.sum(np.abs(kernel))

        # Pad to n_fft
        kernel_padded = np.zeros(n_fft, dtype=np.complex128)
        kernel_padded[:length_int] = kernel

        # FFT
        kernel_fft = np.fft.fft(kernel_padded)

        # Scale by length / n_fft
        kernel_fft *= length / n_fft  # Note: use continuous length, not int

        filters.append(kernel_fft)

    filters = np.array(filters)  # (n_bins_oct, n_fft)
    print(f"Filter bank shape: {filters.shape}")

    # Compute hop length at current SR
    current_hop = HOP_LENGTH // (2 ** decimations)
    print(f"Current hop: {current_hop}")

    # Compute STFT
    stft = librosa.stft(current_audio, n_fft=n_fft, hop_length=current_hop,
                        window='hann', center=True, pad_mode='constant')
    print(f"STFT shape: {stft.shape}")

    # The CQT response is: filters @ stft (matrix multiplication)
    # filters: (n_bins_oct, n_fft)
    # stft: (n_fft/2+1, n_frames)
    # We need full FFT for the multiplication

    n_frames = stft.shape[1]
    cqt_oct = np.zeros((len(oct_freqs), n_frames), dtype=np.complex128)

    for frame in range(n_frames):
        # Get full FFT from STFT
        frame_stft = stft[:, frame]
        n_pos = n_fft // 2 + 1
        stft_full = np.zeros(n_fft, dtype=np.complex128)
        stft_full[:n_pos] = frame_stft
        if n_fft > 2:
            stft_full[n_pos:] = np.conj(frame_stft[-2:0:-1])

        # Multiply with filter bank
        for i, kernel_fft in enumerate(filters):
            cqt_oct[i, frame] = np.sum(stft_full * np.conj(kernel_fft))

    print(f"CQT octave shape: {cqt_oct.shape}")

    # Apply VQT scaling
    vqt_scale = np.sqrt(SR / current_sr)
    cqt_oct *= vqt_scale
    print(f"VQT scale: {vqt_scale}")

    # Now check bin 126 (local index 126 - 108 = 18)
    local_idx = 126 - bin_start
    print(f"\nBin 126 (local idx {local_idx}):")
    print(f"  Raw magnitude at frame 10: {np.abs(cqt_oct[local_idx, 10]):.6f}")

    # Final normalization by sqrt(lengths) - using original SR lengths
    final_lengths = Q * SR / oct_freqs
    cqt_oct_normalized = cqt_oct / np.sqrt(final_lengths[:, np.newaxis])

    print(f"  After /sqrt(lengths): {np.abs(cqt_oct_normalized[local_idx, 10]):.6f}")

    # Compare with librosa
    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_librosa_mag = np.abs(cqt_librosa).T

    print(f"\nLibrosa value at bin 126, frame 10: {cqt_librosa_mag[10, 126]:.6f}")

    ratio = cqt_librosa_mag[10, 126] / np.abs(cqt_oct_normalized[local_idx, 10])
    print(f"Ratio: {ratio:.6f}")

if __name__ == "__main__":
    main()
