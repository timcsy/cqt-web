#!/usr/bin/env python3
"""Debug librosa CQT internals."""

import numpy as np
import librosa
from scipy import signal
from scipy.fft import fft, ifft

# Parameters
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

def main():
    print("=" * 70)
    print("librosa CQT Internals Debug")
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

    # Frequencies
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)

    # Let's try to understand librosa's __cqt_response function
    # From librosa source: it uses conv1d or fft-based convolution

    print("\n" + "-" * 70)
    print("Testing librosa-style convolution for bin 126")
    print("-" * 70)

    # Bin 126 is in octave 3
    octave = 3
    n_octaves = 8
    decimations = n_octaves - 1 - octave  # 4

    # Decimate audio
    y_dec = y.copy()
    sr_dec = SR
    for i in range(decimations):
        y_dec = librosa.resample(y_dec, orig_sr=sr_dec, target_sr=sr_dec//2, res_type='soxr_hq')
        sr_dec //= 2

    print(f"Decimated SR: {sr_dec}")
    print(f"Decimated audio length: {len(y_dec)}")

    # Build filter for bin 126 at decimated SR
    freq = freqs[126]
    length = int(np.ceil(Q * sr_dec / freq))
    print(f"Filter length: {length}")

    # Create kernel (same as before)
    t_kernel = np.arange(length, dtype=np.float64)
    t_kernel = (t_kernel - (length - 1) / 2) / sr_dec
    kernel = np.exp(2j * np.pi * freq * t_kernel)
    window = np.hanning(length)
    kernel *= window

    # L1 normalize
    kernel /= np.sum(np.abs(kernel))

    # librosa uses fftconvolve - let's try that
    # The key is that librosa does: fftconvolve(y, kernel[::-1].conj())
    # This is equivalent to correlation

    # Actually, let's look at what __cqt_response does:
    # It uses scipy.signal.fftconvolve with mode='same'

    # Try direct correlation using fftconvolve
    result_conv = signal.fftconvolve(y_dec, kernel[::-1].conj(), mode='same')
    print(f"\nfftconvolve result shape: {result_conv.shape}")

    # Get frame 0 (at position 0 with center=True means center at index 0)
    # With mode='same', output has same length as input
    # Frame at position 0 corresponds to index 0

    hop_dec = HOP_LENGTH // (2 ** decimations)
    print(f"Decimated hop: {hop_dec}")

    frame_0_value = result_conv[0]
    print(f"Frame 0 value (complex): {frame_0_value}")
    print(f"Frame 0 magnitude: {np.abs(frame_0_value):.10f}")

    # Apply VQT scale
    vqt_scale = np.sqrt(2 ** decimations)
    scaled = frame_0_value * vqt_scale
    print(f"After VQT scale: {np.abs(scaled):.10f}")

    # Normalize by sqrt(filter_length at original SR)
    length_orig = int(np.ceil(Q * SR / freq))
    normalized = scaled / np.sqrt(length_orig)
    print(f"Final (normalized by sqrt(orig length)): {np.abs(normalized):.10f}")

    # Compare with librosa
    cqt = librosa.cqt(
        y, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_mag = np.abs(cqt).T

    print(f"\nlibrosa result: {cqt_mag[0, 126]:.10f}")
    print(f"Ratio: {np.abs(normalized) / cqt_mag[0, 126]:.6f}")

    # Try frame 10 instead (might have more signal)
    print("\n" + "-" * 70)
    print("Testing frame 10:")
    print("-" * 70)

    frame_10_idx = 10 * hop_dec
    if frame_10_idx < len(result_conv):
        frame_10_value = result_conv[frame_10_idx]
        print(f"Frame 10 value (complex): {frame_10_value}")
        print(f"Frame 10 magnitude: {np.abs(frame_10_value):.10f}")

        scaled_10 = frame_10_value * vqt_scale
        normalized_10 = scaled_10 / np.sqrt(length_orig)
        print(f"Final (frame 10): {np.abs(normalized_10):.10f}")
        print(f"librosa frame 10: {cqt_mag[10, 126]:.10f}")
        print(f"Ratio: {np.abs(normalized_10) / cqt_mag[10, 126]:.6f}")

    # Let's check what the actual filter response looks like
    print("\n" + "-" * 70)
    print("Filter analysis:")
    print("-" * 70)

    # The key insight might be that librosa doesn't actually do frame-by-frame processing
    # Instead, it does a full convolution and then samples at hop intervals

    # Let's get all frames by sampling at hop intervals
    frames_manual = []
    for i in range(87):
        idx = i * hop_dec
        if idx < len(result_conv):
            val = result_conv[idx] * vqt_scale / np.sqrt(length_orig)
            frames_manual.append(np.abs(val))
        else:
            frames_manual.append(0)

    frames_manual = np.array(frames_manual)
    frames_librosa = cqt_mag[:, 126]

    print(f"Manual mean: {frames_manual.mean():.6f}")
    print(f"librosa mean: {frames_librosa.mean():.6f}")
    print(f"Mean ratio: {frames_manual.mean() / frames_librosa.mean():.6f}")

    # Check if there's a constant scale factor
    nonzero = frames_librosa > 0.01
    if np.any(nonzero):
        ratios = frames_manual[nonzero] / frames_librosa[nonzero]
        print(f"Per-frame ratios (where librosa > 0.01): min={ratios.min():.6f}, max={ratios.max():.6f}, mean={ratios.mean():.6f}")

if __name__ == "__main__":
    main()
