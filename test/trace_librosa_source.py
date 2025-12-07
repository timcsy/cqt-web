#!/usr/bin/env python3
"""Trace librosa source code to understand exact algorithm."""

import numpy as np
import librosa
import librosa.core.constantq

# Check librosa version
print(f"librosa version: {librosa.__version__}")

# Look at the actual cqt function signature and docstring
print("\n" + "=" * 70)
print("librosa.cqt parameters:")
print("=" * 70)
import inspect
sig = inspect.signature(librosa.cqt)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.default}")

# The key insight is that librosa.cqt calls __cqt with different arguments
# depending on the scale parameter

# Let's trace through manually
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

print("\n" + "=" * 70)
print("Manual trace of librosa CQT:")
print("=" * 70)

# From librosa source, cqt() with default scale='vqt' calls hybrid_cqt
# hybrid_cqt does:
# 1. Compute n_octaves
# 2. Compute freqs_octave for each octave
# 3. For each octave, compute the CQT at appropriate resolution

n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
print(f"n_octaves: {n_octaves}")

# librosa uses early downsampling to reduce computation
# The key is understanding how it maps octaves to bins

# From the source code analysis:
# - hybrid_cqt processes octaves from 0 to n_octaves-1
# - Octave 0 processes the LOWEST frequency bins (bins 0 to bpo-1)
# - The audio is progressively downsampled as octaves increase
# - BUT the filter bank is built at the EFFECTIVE sample rate

# The confusion is about what "effective sample rate" means:
# For octave 0 (lowest freq): no downsampling yet, sr_oct = sr
# For octave 1: audio has been downsampled once, sr_oct = sr/2
# etc.

# Let's verify by computing filter lengths manually
Q = 1.0 / (2.0 ** (1.0 / BINS_PER_OCTAVE) - 1)
freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)

print("\n" + "-" * 70)
print("librosa octave processing (from source analysis):")
print("-" * 70)

# In librosa's __cqt_response:
# For each octave oct, the effective sample rate is sr / 2^oct
# The filter lengths are: ceil(Q * sr_oct / freq)

for oct in range(n_octaves):
    bin_start = oct * BINS_PER_OCTAVE
    bin_end = min((oct + 1) * BINS_PER_OCTAVE, N_BINS)

    # Effective SR for this octave
    # librosa uses early_downsample_count to determine initial downsampling
    # Then progressively downsamples

    sr_oct = SR / (2 ** oct)

    # Filter lengths at effective SR
    lengths_oct = np.ceil(Q * sr_oct / freqs[bin_start:bin_end]).astype(int)

    print(f"Octave {oct}: bins {bin_start:3d}-{bin_end-1:3d}, "
          f"sr_oct={sr_oct:.1f}, "
          f"filter lengths {lengths_oct.min()}-{lengths_oct.max()}")

# Now let's compute the normalization factor
# librosa applies: C /= sqrt(lengths) at the ORIGINAL sample rate
print("\n" + "-" * 70)
print("Normalization (filter lengths at original SR):")
print("-" * 70)

lengths_orig = np.ceil(Q * SR / freqs).astype(int)
print(f"Filter lengths at SR={SR}:")
print(f"  Bin 0: {lengths_orig[0]}")
print(f"  Bin 126 (C4): {lengths_orig[126]}")
print(f"  Bin 287: {lengths_orig[-1]}")

# The VQT scaling factor
print("\n" + "-" * 70)
print("VQT scaling factor:")
print("-" * 70)

# In librosa hybrid_cqt, each octave's result is scaled by sqrt(sr / sr_oct)
# This is equivalent to sqrt(2^oct)
for oct in range(n_octaves):
    scale = np.sqrt(2 ** oct)
    print(f"Octave {oct}: VQT scale = {scale:.6f}")

# Let's compute what the expected ratio should be
print("\n" + "-" * 70)
print("Expected correction factor:")
print("-" * 70)

# For C++ implementation, I was using:
# 1. Octave 0 = highest freq (wrong!)
# 2. VQT scale = sqrt(2^oct) based on wrong octave mapping

# The correct mapping should be:
# - Octave 0 = lowest freq bins (0-35), no decimation, VQT scale = 1
# - Octave 7 = highest freq bins (252-287), decimated 7 times, VQT scale = sqrt(128) ≈ 11.3

# But wait, if octave 7 has the highest frequencies and smallest filter lengths,
# and it's processed at sr/128 = 172 Hz, then the filter length at 172 Hz would be:
# length = ceil(Q * 172 / 5807) = ceil(51.4 * 172 / 5807) = ceil(1.52) = 2

# This matches the debug output!

print("\nActual librosa processing order:")
print("Octave 0 (lowest freq bins 0-35): processed at full SR, VQT scale = 1")
print("Octave 7 (highest freq bins 252-287): processed at SR/128, VQT scale = sqrt(128)")

print("\nThis means for high frequencies, librosa uses very short filters")
print("but compensates with the VQT scaling factor.")
