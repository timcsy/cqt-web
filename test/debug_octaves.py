#!/usr/bin/env python3
"""
Debug each octave's CQT computation step by step.
Compare with what the C++ code should produce.
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

def simple_cqt_octave(audio, sr, hop_length, freqs, Q):
    """
    Compute CQT for one octave using simple convolution.
    This is what the C++ code attempts to do.
    """
    n_bins = len(freqs)
    n_frames = int(np.ceil(len(audio) / hop_length))
    result = np.zeros((n_frames, n_bins), dtype=np.complex128)

    for k, freq in enumerate(freqs):
        # Filter length
        length = int(np.ceil(Q * sr / freq))

        # Create kernel: complex exponential * Hann window
        t = np.arange(length, dtype=np.float64)
        t_centered = (t - (length - 1) / 2) / sr
        kernel = np.exp(2j * np.pi * freq * t_centered)
        window = np.hanning(length)
        kernel *= window

        # L1 normalize
        kernel /= np.sum(np.abs(kernel))

        # Convolve with audio
        # Using scipy.signal.fftconvolve with mode='same'
        conv_result = signal.fftconvolve(audio, kernel[::-1].conj(), mode='same')

        # Sample at hop intervals
        for frame in range(n_frames):
            center = frame * hop_length
            if center < len(conv_result):
                result[frame, k] = conv_result[center]

    return result

def debug_single_octave(octave, audio, sr, hop_length):
    """Debug a single octave's computation."""
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))

    bin_start = octave * BINS_PER_OCTAVE
    bin_end = min((octave + 1) * BINS_PER_OCTAVE, N_BINS)
    n_bins_oct = bin_end - bin_start

    # Frequencies for this octave
    freqs = FMIN * (2.0 ** (np.arange(bin_start, bin_end) / BINS_PER_OCTAVE))

    print(f"\nOctave {octave}: bins {bin_start}-{bin_end-1}")
    print(f"  Frequencies: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"  Audio length: {len(audio)}, SR: {sr}, hop: {hop_length}")

    # Compute CQT for this octave
    cqt_oct = simple_cqt_octave(audio, sr, hop_length, freqs, Q)

    # Filter lengths
    filter_lengths = np.ceil(Q * sr / freqs).astype(int)
    print(f"  Filter lengths: {filter_lengths[0]} - {filter_lengths[-1]}")

    # Take magnitude and normalize
    # The normalization factor should be sqrt(filter_length)
    cqt_mag = np.abs(cqt_oct)
    cqt_normalized = cqt_mag / np.sqrt(filter_lengths)[np.newaxis, :]

    # VQT scaling
    decimations = n_octaves - 1 - octave
    vqt_scale = np.sqrt(2.0 ** decimations)
    print(f"  Decimations: {decimations}, VQT scale: {vqt_scale:.4f}")

    # Check frame 10
    if cqt_normalized.shape[0] > 10:
        print(f"  Frame 10, first 5 bins: {cqt_normalized[10, :5]}")
        print(f"  Frame 10, max bin: {np.argmax(cqt_normalized[10])}, value: {np.max(cqt_normalized[10]):.6f}")

    return cqt_normalized, vqt_scale

def main():
    print("=" * 70)
    print("Debug CQT Octave by Octave")
    print("=" * 70)

    audio = generate_test_audio()
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))

    print(f"\nParameters:")
    print(f"  Q: {Q:.6f}")
    print(f"  n_octaves: {n_octaves}")
    print(f"  Audio length: {len(audio)}")

    # Expected peaks: C4 (261.63), E4 (329.63), G4 (392)
    # C4 is in octave 3 (bins 108-143), bin 126
    # E4 is in octave 3 (bins 108-143), bin 138
    # G4 is in octave 4 (bins 144-179), bin 147

    print("\nExpected peak bins:")
    print("  C4 (261.63 Hz): bin 126 (octave 3)")
    print("  E4 (329.63 Hz): bin 138 (octave 3)")
    print("  G4 (392 Hz): bin 147 (octave 4)")

    # Process each octave like librosa does: highest to lowest
    print("\n" + "=" * 70)
    print("Processing octaves (highest to lowest)")
    print("=" * 70)

    current_audio = audio.astype(np.float64)
    current_sr = SR
    current_hop = HOP_LENGTH

    all_results = {}

    for i in range(n_octaves):
        oct = n_octaves - 1 - i

        # Debug this octave
        cqt_oct, vqt_scale = debug_single_octave(oct, current_audio, current_sr, current_hop)
        all_results[oct] = (cqt_oct, vqt_scale, current_sr, current_hop)

        # Decimate for next octave
        if oct > 0:
            # Apply low-pass filter and downsample by 2
            current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)
            current_sr //= 2
            current_hop = max(1, current_hop // 2)
            print(f"  After decimation: audio_len={len(current_audio)}, sr={current_sr}, hop={current_hop}")

    # Now compare with librosa
    print("\n" + "=" * 70)
    print("Comparison with librosa.hybrid_cqt")
    print("=" * 70)

    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    cqt_librosa_mag = np.abs(cqt_librosa).T

    print(f"\nLibrosa CQT shape: {cqt_librosa_mag.shape}")
    print(f"\nFrame 10 at expected peaks:")
    print(f"  Bin 126 (C4): librosa={cqt_librosa_mag[10, 126]:.6f}")
    print(f"  Bin 138 (E4): librosa={cqt_librosa_mag[10, 138]:.6f}")
    print(f"  Bin 147 (G4): librosa={cqt_librosa_mag[10, 147]:.6f}")

    # Check what our simple implementation produces for octave 3 (C4, E4) and octave 4 (G4)
    print("\n\nOctave 3 (bins 108-143, contains C4 and E4):")
    if 3 in all_results:
        cqt_oct3, vqt_scale3, sr3, hop3 = all_results[3]
        # Bin 126 is at local index 126 - 108 = 18
        # Bin 138 is at local index 138 - 108 = 30
        if cqt_oct3.shape[0] > 10:
            print(f"  Local bin 18 (C4): {cqt_oct3[10, 18]:.6f} * {vqt_scale3:.4f} = {cqt_oct3[10, 18] * vqt_scale3:.6f}")
            print(f"  Local bin 30 (E4): {cqt_oct3[10, 30]:.6f} * {vqt_scale3:.4f} = {cqt_oct3[10, 30] * vqt_scale3:.6f}")

    print("\nOctave 4 (bins 144-179, contains G4):")
    if 4 in all_results:
        cqt_oct4, vqt_scale4, sr4, hop4 = all_results[4]
        # Bin 147 is at local index 147 - 144 = 3
        if cqt_oct4.shape[0] > 10:
            print(f"  Local bin 3 (G4): {cqt_oct4[10, 3]:.6f} * {vqt_scale4:.4f} = {cqt_oct4[10, 3] * vqt_scale4:.6f}")

if __name__ == "__main__":
    main()
