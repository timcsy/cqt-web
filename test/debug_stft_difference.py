#!/usr/bin/env python3
"""
Debug the exact difference between frame-by-frame FFT and librosa STFT.
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
    print("Debug STFT difference")
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

    print(f"Audio length: {len(current_audio)}")
    print(f"Effective SR: {effective_sr}")
    print(f"Current hop: {current_hop}")

    # Filter info for bin 126
    freq = frequencies[126]
    length = int(np.ceil(Q * effective_sr / freq))
    n_fft = 512

    print(f"\nFilter length: {length}")
    print(f"n_fft: {n_fft}")

    # Frame 10
    frame = 10
    center = frame * current_hop

    print(f"\nFrame {frame}, center={center}")

    # Method 1: My frame extraction (C++ style)
    print("\n=== Method 1: C++ style frame extraction ===")
    frame_data_cpp = np.zeros(n_fft, dtype=np.float64)
    for j in range(n_fft):
        idx = center - n_fft // 2 + j
        if 0 <= idx < len(current_audio):
            frame_data_cpp[j] = current_audio[idx]

    print(f"Frame extraction: indices {center - n_fft // 2} to {center + n_fft // 2 - 1}")
    print(f"Frame sum: {np.sum(np.abs(frame_data_cpp)):.6f}")
    print(f"Frame energy: {np.sum(frame_data_cpp ** 2):.6f}")

    # Method 2: librosa STFT style
    print("\n=== Method 2: librosa STFT style ===")
    stft = librosa.stft(current_audio, n_fft=n_fft, hop_length=current_hop,
                        window='hann', center=True)
    frame_stft = stft[:, frame]

    # Reconstruct the windowed frame that librosa uses
    # librosa pads with n_fft//2 zeros at beginning and end when center=True
    padded_audio = np.pad(current_audio, (n_fft // 2, n_fft // 2), mode='constant')

    # Extract frame with Hann window
    frame_start = frame * current_hop
    frame_end = frame_start + n_fft
    frame_data_librosa = padded_audio[frame_start:frame_end]
    window = np.hanning(n_fft)
    frame_data_windowed = frame_data_librosa * window

    print(f"Padded audio length: {len(padded_audio)}")
    print(f"Frame start: {frame_start}, Frame end: {frame_end}")
    print(f"Frame sum (before window): {np.sum(np.abs(frame_data_librosa)):.6f}")
    print(f"Frame sum (after window): {np.sum(np.abs(frame_data_windowed)):.6f}")
    print(f"Frame energy (after window): {np.sum(frame_data_windowed ** 2):.6f}")

    # Check if they are at the same position
    print("\n=== Frame comparison ===")
    print(f"C++ center index in original audio: {center}")
    print(f"librosa center index in original audio: {frame * current_hop}")

    # Check the actual audio values
    print(f"\nC++ frame first 5: {frame_data_cpp[:5]}")
    print(f"librosa frame first 5 (no window): {frame_data_librosa[:5]}")

    # Key insight: librosa's STFT applies a window, but our kernel already has a window!
    # So we're double-windowing in the correlation?

    # Let's compute the correlation correctly
    print("\n=== Correlation analysis ===")

    # Build CQT kernel (same for both)
    kernel = np.zeros(n_fft, dtype=np.complex128)
    for n in range(length):
        t = (n - (length - 1) / 2.0) / effective_sr
        phase = 2.0 * np.pi * freq * t
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1))) if length > 1 else 1.0
        kernel[n] = np.exp(1j * phase) * w

    kernel /= np.sum(np.abs(kernel))
    kernel_fft = np.fft.fft(kernel)
    kernel_fft *= length / n_fft

    # C++ style: use raw frame (no window)
    frame_fft_cpp = np.fft.fft(frame_data_cpp)
    response_cpp = np.sum(frame_fft_cpp * np.conj(kernel_fft))
    # Wait, this is wrong. We want the value at the correlation center, not sum.
    # Actually, for circular correlation via FFT:
    # ifft(fft(a) * conj(fft(b))) gives correlation
    # The value at index 0 is the correlation when b starts at a[0]

    product_cpp = frame_fft_cpp * np.conj(kernel_fft)
    corr_cpp = np.fft.ifft(product_cpp)

    print(f"\nC++ style correlation (index 0): {np.abs(corr_cpp[0]):.6f}")
    print(f"C++ style correlation (max): {np.abs(corr_cpp).max():.6f}")
    print(f"C++ style correlation (max index): {np.argmax(np.abs(corr_cpp))}")

    # librosa style: use STFT (windowed frame)
    # Reconstruct full FFT
    n_pos = n_fft // 2 + 1
    stft_full = np.zeros(n_fft, dtype=np.complex128)
    stft_full[:n_pos] = frame_stft
    if n_fft > 2:
        stft_full[n_pos:] = np.conj(frame_stft[-2:0:-1])

    response_librosa = np.sum(stft_full * np.conj(kernel_fft))
    print(f"\nlibrosa style response (dot product): {np.abs(response_librosa):.6f}")

    # The key difference: librosa uses a DOT PRODUCT, not a full correlation
    # In librosa's __cqt_response, it computes: C = fft_basis @ stft
    # This is a matrix multiplication, which is equivalent to dot product for each row

    # Let's verify by computing the same way
    print("\n=== Understanding librosa's method ===")
    print("librosa uses: fft_basis @ stft, which is a dot product for each frequency bin")
    print("This is NOT a correlation, it's a direct multiplication in frequency domain")
    print(f"\nSo the correct operation is: sum(stft_full * conj(kernel_fft))")
    print(f"Result: {np.abs(response_librosa):.6f}")

    # Now, the question is: why does C++ give different results?
    # The C++ code does: ifft(frame_fft * conj(kernel_fft))[0]
    # This is equivalent to: sum(frame * kernel*) / n_fft  ... NO!
    # Actually, ifft(X * Y)[0] = mean(ifft(X * Y)) ... NO, that's wrong too.
    # ifft(X * Y)[0] = (1/n_fft) * sum(X * Y)

    # Wait, ifft(X)[0] = (1/n_fft) * sum(X)
    # So ifft(frame_fft * conj(kernel_fft))[0] = (1/n_fft) * sum(frame_fft * conj(kernel_fft))

    # But we want sum(frame_fft * conj(kernel_fft)), which is n_fft times ifft(...)[0]!

    print("\n=== Found the bug! ===")
    print("ifft(X * Y)[0] = (1/n_fft) * sum(X * Y)")
    print("We want: sum(X * Y)")
    print("So we need to multiply by n_fft!")

    cpp_corrected = corr_cpp[0] * n_fft
    print(f"\nC++ corrected (multiply by n_fft): {np.abs(cpp_corrected):.6f}")
    print(f"librosa response: {np.abs(response_librosa):.6f}")
    print(f"Ratio: {np.abs(response_librosa) / np.abs(cpp_corrected):.6f}")

    # Still not matching... let's check if it's the window
    print("\n=== Window effect ===")
    print("librosa STFT applies Hann window to the frame BEFORE FFT")
    print("But C++ uses raw frame (no window)")

    # Apply window to C++ frame
    window = np.hanning(n_fft)
    frame_data_cpp_windowed = frame_data_cpp * window
    frame_fft_cpp_windowed = np.fft.fft(frame_data_cpp_windowed)
    response_cpp_windowed = np.sum(frame_fft_cpp_windowed * np.conj(kernel_fft))

    print(f"\nC++ with Hann window: {np.abs(response_cpp_windowed):.6f}")
    print(f"librosa: {np.abs(response_librosa):.6f}")
    print(f"Ratio: {np.abs(response_librosa) / np.abs(response_cpp_windowed):.6f}")

    # Check frame alignment
    print("\n=== Frame alignment check ===")
    print(f"Are frame positions the same?")
    print(f"C++ frame center in original: {center}")
    print(f"After padding (librosa): frame_start={frame_start}, so center in original = {frame_start - n_fft//2 + n_fft//2} = {frame_start}")
    print(f"Wait, librosa's center in original = frame_start - n_fft//2 = {frame_start - n_fft//2}")
    print(f"But with center=True, librosa pads, so center in original = frame * hop = {frame * current_hop}")

    # So both are centered at the same position
    # Let's check the actual frame data
    print(f"\n=== Checking actual frame data ===")
    # Extract the same data as librosa
    cpp_frame_correct = np.zeros(n_fft, dtype=np.float64)
    for j in range(n_fft):
        idx = frame * current_hop - n_fft // 2 + j
        if 0 <= idx < len(current_audio):
            cpp_frame_correct[j] = current_audio[idx]

    print(f"C++ frame sum: {np.sum(np.abs(cpp_frame_correct)):.6f}")
    print(f"librosa frame sum (no window): {np.sum(np.abs(frame_data_librosa)):.6f}")
    print(f"Difference: {np.sum(np.abs(cpp_frame_correct - frame_data_librosa)):.6f}")

if __name__ == "__main__":
    main()
