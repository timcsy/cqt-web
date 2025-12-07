#!/usr/bin/env python3
"""
Debug the normalization factors in librosa's CQT.
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
    print("Debug Normalization Factors")
    print("=" * 70)

    audio = generate_test_audio()
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))

    # Focus on C4 (bin 126, octave 3)
    k = 126
    octave = 3
    freq = FMIN * (2.0 ** (k / BINS_PER_OCTAVE))
    print(f"\nAnalyzing bin {k} (C4 = {freq:.2f} Hz)")
    print(f"Octave: {octave}")

    # When processing octave 3, how many decimations have been done?
    # librosa processes from highest (oct 7) to lowest (oct 0)
    # So when processing oct 3, we've done 7-3 = 4 decimations
    decimations = n_octaves - 1 - octave
    print(f"Decimations done: {decimations}")

    # Effective SR at this point
    effective_sr = SR / (2 ** decimations)
    print(f"Effective SR: {effective_sr}")

    # Filter length at effective SR
    filter_len_eff = int(np.ceil(Q * effective_sr / freq))
    print(f"Filter length at effective SR: {filter_len_eff}")

    # Filter length at original SR
    filter_len_orig = int(np.ceil(Q * SR / freq))
    print(f"Filter length at original SR: {filter_len_orig}")

    # Now let's trace through what librosa actually does
    print("\n" + "=" * 70)
    print("Librosa's normalization factors")
    print("=" * 70)

    # librosa uses:
    # 1. L1 normalized wavelet kernels
    # 2. Kernel scaled by length/n_fft (in __vqt_filter_fft)
    # 3. VQT scaling: sqrt(sr_orig / sr_current)
    # 4. Final normalization by sqrt(lengths) - using original lengths

    # Let's compute manually
    print("\n1. L1 normalization: kernel /= sum(abs(kernel))")

    # Create kernel like librosa
    length = filter_len_eff
    t = np.arange(length, dtype=np.float64)
    t_centered = (t - (length - 1) / 2) / effective_sr
    kernel = np.exp(2j * np.pi * freq * t_centered)
    window = np.hanning(length)
    kernel *= window
    l1_norm = np.sum(np.abs(kernel))
    print(f"   L1 norm of kernel: {l1_norm:.6f}")
    kernel_l1 = kernel / l1_norm

    print("\n2. Scale by length/n_fft")
    n_fft = 2 ** int(np.ceil(np.log2(length)))
    scale_factor = length / n_fft
    print(f"   length: {length}, n_fft: {n_fft}")
    print(f"   scale_factor: {scale_factor:.6f}")

    print("\n3. VQT scaling: sqrt(sr_orig / sr_current)")
    vqt_scale = np.sqrt(SR / effective_sr)
    print(f"   sqrt({SR} / {effective_sr}) = {vqt_scale:.6f}")

    print("\n4. Final normalization by sqrt(lengths)")
    # librosa uses the lengths computed at the FULL sample rate for normalization
    # But the response itself is computed at decimated SR
    # The normalization is applied in __vqt, line:
    #   C /= np.sqrt(lengths)
    # where lengths are computed at ORIGINAL sr: lengths = Q * sr / freqs
    lengths_orig = Q * SR / freq
    print(f"   lengths_orig: {lengths_orig:.6f}")
    print(f"   sqrt(lengths_orig): {np.sqrt(lengths_orig):.6f}")

    # Total scaling from convolution to final output
    # If we have a unit-amplitude sinusoid at freq, the CQT response is:
    # response = (convolution with L1-normalized kernel) * (length/nfft) * vqt_scale / sqrt(lengths_orig)

    print("\n" + "=" * 70)
    print("Verifying with actual computation")
    print("=" * 70)

    # Decimate audio to effective SR
    current_audio = audio.astype(np.float64)
    for _ in range(decimations):
        current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)

    print(f"\nDecimated audio length: {len(current_audio)}")

    # Convolve with L1-normalized kernel
    conv_result = signal.fftconvolve(current_audio, kernel_l1[::-1].conj(), mode='same')

    # Take value at frame 10
    frame = 10
    effective_hop = HOP_LENGTH // (2 ** decimations)
    center = frame * effective_hop
    conv_val = conv_result[center]
    print(f"\nFrame {frame}, center={center}")
    print(f"Convolution value: {np.abs(conv_val):.6f}")

    # Apply scale_factor
    scaled_val = conv_val * scale_factor
    print(f"After length/nfft: {np.abs(scaled_val):.6f}")

    # Apply VQT scale
    vqt_scaled = scaled_val * vqt_scale
    print(f"After VQT scale: {np.abs(vqt_scaled):.6f}")

    # Apply final normalization
    final_val = vqt_scaled / np.sqrt(lengths_orig)
    print(f"After /sqrt(lengths): {np.abs(final_val):.6f}")

    # Compare with librosa
    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_librosa_mag = np.abs(cqt_librosa).T
    print(f"\nLibrosa value at bin {k}, frame {frame}: {cqt_librosa_mag[frame, k]:.6f}")

    # The ratio
    ratio = cqt_librosa_mag[frame, k] / np.abs(final_val)
    print(f"Ratio (librosa / our): {ratio:.6f}")

    # Let's also check the convolution without L1 normalization
    print("\n" + "=" * 70)
    print("Alternative: Without L1 normalization")
    print("=" * 70)

    # librosa's wavelet uses L2 normalization for the final length adjustment
    # Let me check the actual librosa code more carefully

    # The key is that librosa computes:
    # 1. Basis filters in frequency domain
    # 2. STFT of audio
    # 3. C = fft_basis @ stft
    # 4. C /= sqrt(lengths)

    # The fft_basis is constructed with:
    # - L1 normalized wavelets
    # - Scaled by length / n_fft
    # - Conjugated for correlation

    # But there's a missing piece: the STFT uses a window,
    # and the window sum normalization

    # Actually, looking at librosa's __cqt_response:
    # It uses STFT with window=True, which applies Hann window and proper normalization

    print("\nLet me trace through using STFT approach like librosa...")

    # STFT of audio at decimated SR
    stft = librosa.stft(current_audio, n_fft=n_fft, hop_length=effective_hop,
                        window='hann', center=True)
    print(f"STFT shape: {stft.shape}")

    # Create filter in frequency domain
    kernel_padded = np.zeros(n_fft, dtype=np.complex128)
    kernel_padded[:length] = kernel_l1
    kernel_fft = np.fft.fft(kernel_padded)
    kernel_fft *= scale_factor

    # Get CQT coefficient by dot product with STFT column
    # Note: librosa uses conj of kernel for correlation
    frame_stft = stft[:, frame]
    # Only use positive frequencies
    n_pos = n_fft // 2 + 1

    # The response is: sum(stft * conj(kernel_fft))
    # But stft is only positive frequencies, we need full FFT
    stft_full = np.zeros(n_fft, dtype=np.complex128)
    stft_full[:n_pos] = frame_stft
    if n_fft > 2:
        stft_full[n_pos:] = np.conj(frame_stft[-2:0:-1])

    response = np.sum(stft_full * np.conj(kernel_fft))
    print(f"\nSTFT dot product response: {np.abs(response):.6f}")

    # Apply VQT scale
    response_vqt = response * vqt_scale
    print(f"After VQT scale: {np.abs(response_vqt):.6f}")

    # Apply length normalization
    response_final = response_vqt / np.sqrt(lengths_orig)
    print(f"After /sqrt(lengths): {np.abs(response_final):.6f}")

    ratio2 = cqt_librosa_mag[frame, k] / np.abs(response_final)
    print(f"Ratio (librosa / stft): {ratio2:.6f}")

if __name__ == "__main__":
    main()
