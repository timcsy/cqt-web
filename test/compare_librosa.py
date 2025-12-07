#!/usr/bin/env python3
"""Compare C++ CQT output with librosa."""

import numpy as np
import librosa
import json
import sys

# Parameters matching CNN-LSTM model
SR = 22050
HOP_LENGTH = 512
BINS_PER_OCTAVE = 36
N_BINS = 288
FMIN = librosa.note_to_hz('F#0')

def generate_test_audio():
    """Generate C major chord test audio."""
    duration = 2.0
    t = np.arange(int(SR * duration)) / SR
    c4, e4, g4 = 261.63, 329.63, 392.00
    audio = (
        0.3 * np.sin(2 * np.pi * c4 * t) +
        0.3 * np.sin(2 * np.pi * e4 * t) +
        0.3 * np.sin(2 * np.pi * g4 * t)
    ).astype(np.float32)
    return audio

def compute_librosa_cqt(audio):
    """Compute CQT using librosa."""
    cqt = librosa.cqt(
        audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, tuning=0.0,
    )
    return np.abs(cqt).T

def main():
    print("=" * 70)
    print("CQT Comparison: C++ vs Python (librosa)")
    print("=" * 70)

    # Load C++ output
    try:
        with open('cqt_cpp_output.json', 'r') as f:
            cpp_data = json.load(f)
    except FileNotFoundError:
        print("Error: cqt_cpp_output.json not found.")
        print("Please run the C++ test first.")
        sys.exit(1)

    print("\nC++ parameters:")
    print(f"  sr: {cpp_data['params']['sr']}")
    print(f"  hop_length: {cpp_data['params']['hop_length']}")
    print(f"  bins_per_octave: {cpp_data['params']['bins_per_octave']}")
    print(f"  n_bins: {cpp_data['params']['n_bins']}")
    print(f"  fmin: {cpp_data['params']['fmin']:.4f} Hz")
    print(f"  Q: {cpp_data['params']['Q']:.6f}")

    # Generate same audio
    audio = generate_test_audio()
    print(f"\nGenerated audio: {len(audio)} samples")

    # Verify audio matches
    cpp_audio = np.array(cpp_data['audio_first_100'])
    audio_diff = np.abs(audio[:100] - cpp_audio).max()
    print(f"Audio difference (first 100): {audio_diff:.10f}")

    # Compute librosa CQT
    print("\nComputing librosa CQT...")
    librosa_cqt = compute_librosa_cqt(audio)
    print(f"librosa CQT shape: {librosa_cqt.shape}")

    # C++ CQT
    cpp_frame0 = np.array(cpp_data['frame0_full'])
    cpp_frame0_sliced = np.array(cpp_data['frame0_sliced'])
    print(f"C++ frame 0 length: {len(cpp_frame0)}")
    print(f"C++ frame 0 sliced length: {len(cpp_frame0_sliced)}")

    # Compare full frame 0
    print("\n" + "-" * 70)
    print("Full Frame 0 Comparison (288 bins):")
    print("-" * 70)

    librosa_frame0 = librosa_cqt[0]
    diff = np.abs(cpp_frame0 - librosa_frame0)
    print(f"Max difference: {diff.max():.10f}")
    print(f"Mean difference: {diff.mean():.10f}")
    print(f"RMS difference: {np.sqrt((diff**2).mean()):.10f}")

    # Mean values
    print(f"\nC++ mean: {np.mean(cpp_frame0):.10f}")
    print(f"librosa mean: {np.mean(librosa_frame0):.10f}")
    if np.mean(librosa_frame0) != 0:
        print(f"Mean ratio: {np.mean(cpp_frame0) / np.mean(librosa_frame0):.6f}")

    # Compare first 10 bins
    print("\nFirst 10 bins comparison:")
    freqs = cpp_data['freqs']
    for i in range(10):
        print(f"  Bin {i} ({freqs[i]:.2f} Hz): C++={cpp_frame0[i]:.8f}, "
              f"librosa={librosa_frame0[i]:.8f}, diff={diff[i]:.8f}")

    # Compare sliced (bins 18-269)
    print("\n" + "-" * 70)
    print("Sliced Frame 0 Comparison (252 bins, bins 18-269):")
    print("-" * 70)

    librosa_sliced = librosa_cqt[0, 18:270]
    diff_sliced = np.abs(cpp_frame0_sliced - librosa_sliced)
    print(f"Max difference: {diff_sliced.max():.10f}")
    print(f"Mean difference: {diff_sliced.mean():.10f}")
    print(f"RMS difference: {np.sqrt((diff_sliced**2).mean()):.10f}")

    # Find worst bins
    print("\nWorst 10 bins (largest differences):")
    worst_indices = np.argsort(diff_sliced)[::-1][:10]
    for idx in worst_indices:
        orig_bin = idx + 18
        print(f"  Sliced bin {idx} (orig {orig_bin}, {freqs[orig_bin]:.2f} Hz): "
              f"diff={diff_sliced[idx]:.8f}, C++={cpp_frame0_sliced[idx]:.8f}, "
              f"librosa={librosa_sliced[idx]:.8f}")

    # Export librosa reference for debugging
    output = {
        "librosa_frame0_full": librosa_cqt[0].tolist(),
        "librosa_frame0_sliced": librosa_cqt[0, 18:270].tolist(),
        "librosa_shape": list(librosa_cqt.shape),
        "freqs": freqs,
    }

    with open('librosa_reference.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved librosa reference to librosa_reference.json")

    # Check if acceptable
    rms_diff = np.sqrt((diff_sliced**2).mean())
    threshold = 0.01  # 1% RMS error
    if rms_diff < threshold:
        print(f"\n✓ PASS: RMS difference ({rms_diff:.6f}) < threshold ({threshold})")
        return 0
    else:
        print(f"\n✗ FAIL: RMS difference ({rms_diff:.6f}) >= threshold ({threshold})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
