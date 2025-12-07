#!/usr/bin/env python3
"""
Debug the C++ CQT logic by replicating it in Python and comparing with librosa.
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

def replicate_cpp_logic(audio):
    """
    Replicate the C++ CQT logic exactly.
    """
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))

    print(f"Q = {Q:.6f}")
    print(f"n_octaves = {n_octaves}")

    # Compute frequencies
    frequencies = FMIN * (2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE))

    # Number of frames
    num_frames = int(np.ceil(len(audio) / HOP_LENGTH))
    print(f"num_frames = {num_frames}")

    # Result arrays
    cqt_complex = np.zeros((num_frames, N_BINS), dtype=np.complex128)

    # Process octaves from HIGHEST to LOWEST
    current_audio = audio.astype(np.float64)
    current_hop = HOP_LENGTH

    for oct in range(n_octaves - 1, -1, -1):
        decimations_done = n_octaves - 1 - oct

        bin_start = oct * BINS_PER_OCTAVE
        bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)
        n_bins_oct = bin_end - bin_start

        if n_bins_oct <= 0:
            if oct > 0:
                current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)
                current_hop = max(1, current_hop // 2)
            continue

        # Effective SR at this point
        effective_sr = SR / (2 ** decimations_done)

        # Filter lengths at effective SR
        oct_freqs = frequencies[bin_start:bin_end]
        oct_lengths = np.ceil(Q * effective_sr / oct_freqs).astype(int)

        # n_fft for this octave
        max_length = np.max(oct_lengths)
        n_fft = int(2 ** np.ceil(np.log2(max_length)))

        # Build filters and compute CQT for this octave
        oct_result = np.zeros((len(current_audio) // current_hop + 1, n_bins_oct), dtype=np.complex128)

        for i, (freq, length) in enumerate(zip(oct_freqs, oct_lengths)):
            # Create kernel: complex exponential * window
            kernel = np.zeros(n_fft, dtype=np.complex128)

            # Build centered kernel
            for n in range(length):
                t = (n - (length - 1) / 2.0) / effective_sr
                phase = 2.0 * np.pi * freq * t
                window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1))) if length > 1 else 1.0
                kernel[n] = np.exp(1j * phase) * window

            # L1 normalize
            l1_norm = np.sum(np.abs(kernel))
            if l1_norm > 0:
                kernel /= l1_norm

            # FFT of kernel
            kernel_fft = np.fft.fft(kernel)

            # Scale by length / n_fft
            kernel_fft *= length / n_fft

            # For each frame, compute correlation
            n_frames_oct = len(current_audio) // current_hop + 1
            for frame in range(n_frames_oct):
                center = frame * current_hop

                # Extract frame data centered at 'center'
                frame_data = np.zeros(n_fft, dtype=np.complex128)
                for j in range(n_fft):
                    idx = center - n_fft // 2 + j
                    if 0 <= idx < len(current_audio):
                        frame_data[j] = current_audio[idx]

                # FFT of frame
                frame_fft = np.fft.fft(frame_data)

                # Multiply by conjugate of kernel FFT
                product = frame_fft * np.conj(kernel_fft)

                # IFFT
                result = np.fft.ifft(product)

                # Result at index 0
                oct_result[frame, i] = result[0]

        # VQT scaling factor
        vqt_scale = np.sqrt(2.0 ** decimations_done)

        # Copy results
        for frame in range(num_frames):
            src_frame = frame >> decimations_done if decimations_done > 0 else frame
            if src_frame < len(oct_result):
                for i in range(n_bins_oct):
                    cqt_complex[frame, bin_start + i] = oct_result[src_frame, i] * vqt_scale

        # Print debug info for octave 3 (C4)
        if oct == 3:
            print(f"\nOctave 3 debug:")
            print(f"  decimations_done = {decimations_done}")
            print(f"  effective_sr = {effective_sr}")
            print(f"  n_fft = {n_fft}")
            print(f"  vqt_scale = {vqt_scale}")
            print(f"  Bin 126 (local 18) raw: {np.abs(oct_result[10, 18]):.6f}")
            print(f"  Bin 126 (local 18) with vqt: {np.abs(oct_result[10, 18]) * vqt_scale:.6f}")

        # Decimate for next octave
        if oct > 0:
            current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)
            current_hop = max(1, current_hop // 2)

    # Final normalization
    result = np.zeros((num_frames, N_BINS), dtype=np.float64)

    for frame in range(num_frames):
        for k in range(N_BINS):
            octave = k // BINS_PER_OCTAVE
            decimations = n_octaves - 1 - octave

            # Filter length at decimated SR
            effective_sr = SR / (2 ** decimations)
            filter_len_dec = np.ceil(Q * effective_sr / frequencies[k])

            mag = np.abs(cqt_complex[frame, k])
            # Multiply by sqrt(filter_length_at_decimated_sr)
            mag *= np.sqrt(filter_len_dec)

            result[frame, k] = mag

    return result

def main():
    print("=" * 70)
    print("Debug C++ CQT Logic")
    print("=" * 70)

    audio = generate_test_audio()

    # Compute using C++ logic
    print("\n--- Replicating C++ logic ---")
    cpp_result = replicate_cpp_logic(audio)

    # Compare with librosa
    print("\n--- Comparing with librosa ---")
    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_librosa_mag = np.abs(cqt_librosa).T

    print(f"\nFrame 10, Bin 126 (C4):")
    print(f"  C++ logic: {cpp_result[10, 126]:.6f}")
    print(f"  librosa: {cqt_librosa_mag[10, 126]:.6f}")
    print(f"  Ratio: {cqt_librosa_mag[10, 126] / cpp_result[10, 126]:.6f}")

    print(f"\nFrame 10, Bin 147 (G4):")
    print(f"  C++ logic: {cpp_result[10, 147]:.6f}")
    print(f"  librosa: {cqt_librosa_mag[10, 147]:.6f}")
    print(f"  Ratio: {cqt_librosa_mag[10, 147] / cpp_result[10, 147]:.6f}")

if __name__ == "__main__":
    main()
