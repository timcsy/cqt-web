#!/usr/bin/env python3
"""Compare FFT correlation methods."""

import numpy as np
from scipy import signal

def main():
    print("=" * 70)
    print("Comparing FFT Correlation Methods")
    print("=" * 70)

    # Simple test case
    n = 16
    signal_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    kernel = np.array([0.5, 1.0, 0.5], dtype=np.complex128)

    print("\nInput signal:", signal_data)
    print("Kernel:", kernel)

    # Method 1: scipy fftconvolve with kernel reversed and conjugated
    result1 = signal.fftconvolve(signal_data, kernel[::-1].conj(), mode='same')
    print(f"\n1. scipy.fftconvolve(signal, kernel[::-1].conj()): {result1[:8]}")

    # Method 2: Manual FFT correlation (like in C++)
    nfft = 16
    signal_fft = np.fft.fft(signal_data, nfft)
    kernel_padded = np.zeros(nfft, dtype=np.complex128)
    kernel_padded[:len(kernel)] = kernel
    kernel_fft = np.fft.fft(kernel_padded)

    # Inner product: sum(signal_fft * conj(kernel_fft))
    corr_scalar = np.sum(signal_fft * np.conj(kernel_fft))
    print(f"2. sum(signal_fft * conj(kernel_fft)): {corr_scalar}")

    # Method 3: FFT multiplication then IFFT
    result3_fft = signal_fft * np.conj(kernel_fft)
    result3 = np.fft.ifft(result3_fft).real
    print(f"3. ifft(signal_fft * conj(kernel_fft)): {result3[:8]}")

    # Method 4: Direct convolution
    result4 = np.convolve(signal_data, kernel[::-1].conj(), mode='same')
    print(f"4. np.convolve(signal, kernel[::-1].conj()): {result4[:8]}")

    # The key insight: fftconvolve gives an array (convolution at each position)
    # while sum(fft*conj(fft)) gives a scalar (sum of the circular convolution)

    print("\n" + "-" * 70)
    print("Analysis:")
    print("-" * 70)
    print("- fftconvolve returns a convolution at each position")
    print("- sum(signal_fft * conj(kernel_fft)) returns the correlation at position 0")
    print("- To get the full correlation, we need ifft(signal_fft * conj(kernel_fft))")
    print()

    # The C++ code computes: sum(frame_fft * conj(kernel_fft))
    # This is equivalent to: N * ifft(frame_fft * conj(kernel_fft))[0]
    # where N is the FFT size

    print(f"sum(signal_fft * conj(kernel_fft)): {corr_scalar}")
    print(f"N * ifft(...)[0]: {nfft * result3[0]}")

    # So the C++ code is computing correlation at position 0 only,
    # but librosa computes correlation at all positions and then samples

    # For centered convolution (librosa's mode='same'), the correlation at frame center
    # is at a different index

    # Let's verify with the kernel scale factor issue
    print("\n" + "-" * 70)
    print("Scale factor analysis:")
    print("-" * 70)

    # In C++, the kernel is L1 normalized and then scaled by length/nfft
    kernel_l1 = kernel / np.sum(np.abs(kernel))
    kernel_scaled = kernel_l1 * len(kernel) / nfft
    print(f"L1 norm of kernel: {np.sum(np.abs(kernel)):.6f}")
    print(f"Kernel after L1 norm: {kernel_l1}")
    print(f"Kernel after scale by len/nfft: {kernel_scaled}")

    # Now compute with scaled kernel
    kernel_scaled_padded = np.zeros(nfft, dtype=np.complex128)
    kernel_scaled_padded[:len(kernel)] = kernel_scaled
    kernel_scaled_fft = np.fft.fft(kernel_scaled_padded)

    corr_scaled = np.sum(signal_fft * np.conj(kernel_scaled_fft))
    print(f"\nsum(signal_fft * conj(kernel_scaled_fft)): {corr_scaled}")

    # Compare with fftconvolve using L1 normalized kernel
    result_l1 = signal.fftconvolve(signal_data, kernel_l1[::-1].conj(), mode='same')
    print(f"fftconvolve with L1 norm kernel: {result_l1[:8]}")

if __name__ == "__main__":
    main()
