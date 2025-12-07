#!/usr/bin/env python3
"""
Match librosa's exact computation by using its internal functions.
"""

import numpy as np
import librosa
import librosa.core.constantq as cq
from librosa.core.audio import resample

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
    print("Match librosa's exact hybrid_cqt")
    print("=" * 70)

    audio = generate_test_audio()

    # Call hybrid_cqt and examine its internal workings
    # hybrid_cqt calls __vqt internally

    # Let's look at what hybrid_cqt returns and trace the scale

    # First, get the reference
    cqt_ref = librosa.hybrid_cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    print(f"Reference CQT shape: {cqt_ref.shape}")
    print(f"Reference CQT at bin 126, frame 10: {np.abs(cqt_ref[126, 10]):.6f}")

    # Now let's look at the VQT function
    # In librosa, hybrid_cqt calls:
    # 1. __vqt for the CQT computation
    # 2. Applies gamma parameter for pseudo-CQT (which we don't use)

    # The key is in __vqt:
    # - It builds filter banks using __vqt_filter_fft
    # - Computes STFT response using __cqt_response
    # - Applies VQT scaling

    # Let me check the actual lengths used
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    filter_scale = 1.0
    Q = filter_scale / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
    lengths = Q * SR / freqs

    print(f"\nQ = {Q:.6f}")
    print(f"lengths[126] = {lengths[126]:.6f}")

    # Check if librosa uses different normalization
    # Looking at librosa source code:
    # In __vqt_filter_fft, the basis is:
    # 1. Created with wavelet function
    # 2. L1 normalized
    # 3. Scaled by lengths / n_fft
    # 4. FFT'd

    # In __cqt_response, it does:
    # 1. STFT of audio
    # 2. Dot product: C = fft_basis @ stft
    # 3. Returns C

    # In __vqt, it does:
    # 1. C = __cqt_response(...)
    # 2. C *= sqrt(sr_ratio)  # VQT scaling
    # 3. Continue for next octave

    # Finally, in cqt/hybrid_cqt:
    # C /= sqrt(lengths)

    # So the issue might be in how STFT is computed or the basis construction

    # Let me try to use librosa's actual functions
    print("\n" + "=" * 70)
    print("Using librosa's internal functions")
    print("=" * 70)

    # Get the filter basis for octave containing bin 126
    octave = 3
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    decimations = n_octaves - 1 - octave

    bin_start = octave * BINS_PER_OCTAVE
    bin_end = min((octave + 1) * BINS_PER_OCTAVE, N_BINS)
    oct_freqs = freqs[bin_start:bin_end]

    # Resample like librosa
    current_audio = audio.astype(np.float64)
    current_sr = SR
    for _ in range(decimations):
        current_audio = resample(current_audio, orig_sr=current_sr,
                                target_sr=current_sr//2, res_type='soxr_hq')
        current_sr //= 2

    print(f"After resampling: len={len(current_audio)}, sr={current_sr}")

    # Get filter lengths at current SR
    oct_lengths = Q * current_sr / oct_freqs
    n_fft = 2 ** int(np.ceil(np.log2(np.max(oct_lengths))))
    print(f"n_fft = {n_fft}")

    # Let me try a different approach: compute the actual ratio by
    # examining the transformation at each step

    print("\n" + "=" * 70)
    print("Examining the exact scale factor")
    print("=" * 70)

    # For a pure sinusoid at frequency f with amplitude A,
    # the CQT response at bin k (with center frequency f_k = f) should be:
    # A * sqrt(length_k) approximately (for ideal kernel)

    # For our C4 = 261.63 Hz with amplitude 0.3:
    # Expected response ≈ 0.3 * sqrt(4336) ≈ 0.3 * 65.85 ≈ 19.75

    # But librosa gives 9.92, which is about half of this
    # This suggests there's an additional factor of ~2

    # Actually, for complex sinusoid, the L1 norm of kernel is approximately length/2
    # (since Hann window integrates to length/2)
    # So L1 normalized kernel has magnitude ~2/length

    # Convolution of sin(2*pi*f*t) with kernel gives:
    # If kernel is L1 normalized, the response is ~0.5 (since only half matches)

    # Let me compute the expected response more carefully
    local_idx = 126 - bin_start
    f = oct_freqs[local_idx]
    length = oct_lengths[local_idx]

    # The sinusoid amplitude is 0.3
    # The kernel L1 norm is ~length/2 for Hann window
    # So L1 normalized kernel has scale 2/length
    # The convolution response is ~0.3 * 0.5 = 0.15 (half matches complex exponential)
    # After L1 normalization: 0.15 (unchanged since we're looking at the matched component)

    # Wait, let me think more carefully about this...
    # The input is: 0.3 * sin(2*pi*f*t)
    # The kernel is: exp(2*pi*j*f*t) * window / L1_norm
    # Convolution gives: integral of input * conj(kernel)
    # = integral of 0.3 * sin(2*pi*f*t) * exp(-2*pi*j*f*t) * window / L1_norm
    # = integral of 0.3 * (exp(j*2*pi*f*t) - exp(-j*2*pi*f*t))/(2j) * exp(-j*2*pi*f*t) * window / L1_norm
    # = integral of 0.3 * (1 - exp(-j*4*pi*f*t))/(2j) * window / L1_norm
    # ≈ 0.3 * 0.5 / L1_norm * integral(window)  (high freq term averages to 0)
    # = 0.3 * 0.5 / L1_norm * (length/2)  (Hann window integrates to length/2)
    # = 0.3 * 0.5 * (length/2) / L1_norm

    # L1_norm = sum(abs(exp(j*...) * window)) = sum(window) = length/2
    # So: 0.3 * 0.5 * (length/2) / (length/2) = 0.15

    # Hmm, this gives 0.15 before scale factors...

    # Actually, the issue is the STFT. librosa's STFT applies a window and
    # normalizes by sum(window). Let me check this.

    # Looking at librosa.stft:
    # - Applies window to each frame
    # - No normalization by window sum in the forward transform
    # - The response magnitude depends on the window's energy

    # For Hann window of length n_fft, sum(window^2) ≈ n_fft * 3/8

    # Let me just compute what librosa actually produces and match it

    # I'll compute the ratio between my manual calculation and librosa's output
    # to find the missing scale factor

    # Create a pure sine wave at C4 frequency
    c4_freq = freqs[126]
    t = np.arange(len(audio)) / SR
    pure_c4 = 0.3 * np.sin(2 * np.pi * c4_freq * t).astype(np.float32)

    # Compute CQT of pure C4
    cqt_c4 = librosa.hybrid_cqt(
        pure_c4, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )

    print(f"\nPure C4 (0.3 amplitude) CQT at bin 126:")
    print(f"  Frame 10: {np.abs(cqt_c4[126, 10]):.6f}")
    print(f"  Max: {np.max(np.abs(cqt_c4[126, :])):.6f}")

    # What's the relationship between input amplitude and CQT output?
    # For amplitude A, librosa gives A * some_factor
    # Let's find that factor

    factor = np.abs(cqt_c4[126, 10]) / 0.3
    print(f"  Factor (output/input_amplitude): {factor:.6f}")

    # Now, what should this factor be according to the math?
    # According to librosa's normalization:
    # CQT = (STFT_response * kernel) * VQT_scale / sqrt(length)
    # For matched sinusoid, STFT_response * kernel ≈ some constant depending on window

    # Let me check the theoretical maximum response
    # For a sinusoid of amplitude A, the STFT magnitude at the matching bin is:
    # A * sum(window) / 2 (the /2 is because sin = (exp - exp*)/2j)

    # For Hann window of length n_fft:
    # sum(window) = n_fft / 2

    # So STFT magnitude = A * n_fft / 4

    # Then the CQT kernel has magnitude length/n_fft after L1 norm and scaling
    # Response = A * n_fft / 4 * length / n_fft = A * length / 4

    # After VQT scale (sqrt(16) = 4 for octave 3):
    # = A * length / 4 * 4 = A * length

    # After /sqrt(length):
    # = A * sqrt(length)

    # For our case: 0.3 * sqrt(4336) = 0.3 * 65.85 = 19.75

    # But librosa gives ~10, which is about half!

    # The missing factor of 2 might be because:
    # 1. The kernel uses L1 norm which includes the complex part
    # 2. Or the convolution only captures half the energy

    # Looking more carefully: for a real sinusoid sin(2*pi*f*t),
    # the positive and negative frequency components each have amplitude A/2
    # So the CQT (which correlates with exp(j*2*pi*f*t)) captures A/2

    # Expected: 0.3/2 * sqrt(4336) = 0.15 * 65.85 = 9.88

    # That matches! The factor should be 0.5 * sqrt(length)

    expected = 0.5 * np.sqrt(lengths[126])
    print(f"\nExpected factor: 0.5 * sqrt({lengths[126]:.2f}) = {expected:.6f}")
    print(f"Actual factor: {factor:.6f}")
    print(f"Ratio: {factor/expected:.6f}")

if __name__ == "__main__":
    main()
