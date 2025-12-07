#!/usr/bin/env python3
"""Debug octave 4 specifically to find the source of the 37% difference."""

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
    print("Debug Octave 4 (bin 147 = G4)")
    print("=" * 70)

    audio = generate_test_audio()
    
    Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    frequencies = FMIN * (2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE))
    
    print(f"Q = {Q:.6f}")
    print(f"n_octaves = {n_octaves}")
    
    # Bin 147 is in octave 4 (147 // 36 = 4)
    bin_idx = 147
    oct = bin_idx // BINS_PER_OCTAVE
    
    print(f"\nBin {bin_idx} is in octave {oct}")
    print(f"  Frequency: {frequencies[bin_idx]:.2f} Hz (should be ~G4=392)")
    
    # C++ processing order: from highest octave (n-1) down to 0
    # Decimations done before processing octave k = (n_octaves - 1 - k)
    decimations = n_octaves - 1 - oct
    print(f"  Decimations done before processing octave {oct}: {decimations}")
    
    effective_sr = SR / (2 ** decimations)
    print(f"  Effective SR: {effective_sr}")
    
    # Filter length at effective SR
    freq = frequencies[bin_idx]
    length_at_effective = int(np.ceil(Q * effective_sr / freq))
    length_at_original = Q * SR / freq
    
    print(f"  Filter length at effective SR: {length_at_effective}")
    print(f"  Filter length at original SR: {length_at_original:.2f}")
    
    # VQT scale
    vqt_scale = np.sqrt(2 ** decimations)
    print(f"  VQT scale: {vqt_scale:.6f}")
    
    # Now let's trace the actual values
    # First, decimate the audio as C++ would
    current_audio = audio.astype(np.float64)
    
    # For octave 4, decimations = 8 - 1 - 4 = 3
    print(f"\nApplying {decimations} decimations with sqrt(2) scale...")
    for i in range(decimations):
        # Apply half-band filter and downsample
        current_audio = signal.decimate(current_audio, 2, ftype='fir', zero_phase=True)
        current_audio *= np.sqrt(2)  # Energy-preserving scale
        print(f"  After decimation {i+1}: length={len(current_audio)}, max={np.max(np.abs(current_audio)):.4f}")
    
    current_hop = HOP_LENGTH // (2 ** decimations)
    print(f"  Current hop length: {current_hop}")
    
    # Determine FFT size for this octave
    # In C++, we find max filter length for the octave
    bin_start = oct * BINS_PER_OCTAVE
    bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)
    
    max_length = 0
    for k in range(bin_start, bin_end):
        length = int(np.ceil(Q * effective_sr / frequencies[k]))
        max_length = max(max_length, length)
    
    n_fft = int(2 ** np.ceil(np.log2(max_length)))
    print(f"\n  Octave {oct} bins: {bin_start} to {bin_end-1}")
    print(f"  Max filter length: {max_length}")
    print(f"  FFT size: {n_fft}")
    
    # Build kernel for bin 147
    length = length_at_effective
    kernel = np.zeros(n_fft, dtype=np.complex128)
    
    # Center the kernel
    start_idx = (n_fft - length) // 2
    
    for n in range(length):
        t = (n - (length - 1) / 2.0) / effective_sr
        phase = 2.0 * np.pi * freq * t
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1))) if length > 1 else 1.0
        kernel[start_idx + n] = np.exp(1j * phase) * w
    
    # L1 normalize
    l1_norm = np.sum(np.abs(kernel))
    kernel /= l1_norm
    
    # FFT and scale
    kernel_fft = np.fft.fft(kernel)
    kernel_fft *= length / n_fft * vqt_scale
    
    print(f"\n  Kernel L1 norm: {l1_norm:.6f}")
    print(f"  Kernel scale: {length / n_fft * vqt_scale:.6f}")
    
    # Compute frame 10
    frame = 10
    center = frame * current_hop
    
    # Extract frame (no window, as per librosa's window='ones')
    frame_data = np.zeros(n_fft, dtype=np.complex128)
    for j in range(n_fft):
        idx = center - n_fft // 2 + j
        if 0 <= idx < len(current_audio):
            frame_data[j] = current_audio[idx]
    
    # FFT of frame
    frame_fft = np.fft.fft(frame_data)
    
    # Dot product
    response = np.sum(frame_fft * np.conj(kernel_fft))
    
    # Final normalization: divide by sqrt(length at original SR)
    response_normalized = response / np.sqrt(length_at_original)
    
    print(f"\n  Frame center: {center}")
    print(f"  Raw dot product: {np.abs(response):.6f}")
    print(f"  After /sqrt(length_orig): {np.abs(response_normalized):.6f}")
    
    # Get librosa reference
    cqt_librosa = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    
    print(f"\n  librosa value: {np.abs(cqt_librosa[bin_idx, frame]):.6f}")
    print(f"  Ratio (our/librosa): {np.abs(response_normalized) / np.abs(cqt_librosa[bin_idx, frame]):.6f}")
    
    # Let's also check the C++ output
    import json
    try:
        with open('/Users/timcsy/Documents/Projects/test/ChordMini/cqt/build/cqt_cpp_output.json', 'r') as f:
            cpp_data = json.load(f)
        cpp_cqt = np.array(cpp_data['cqt']).reshape(cpp_data['numFrames'], cpp_data['nBins'])
        print(f"  C++ value: {cpp_cqt[frame, bin_idx]:.6f}")
        print(f"  C++/librosa ratio: {cpp_cqt[frame, bin_idx] / np.abs(cqt_librosa[bin_idx, frame]):.6f}")
    except Exception as e:
        print(f"  Could not load C++ output: {e}")

if __name__ == "__main__":
    main()
