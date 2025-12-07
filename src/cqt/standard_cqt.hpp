#ifndef CQT_STANDARD_CQT_HPP
#define CQT_STANDARD_CQT_HPP

#include "cqt_base.hpp"
#include <vector>
#include <complex>

namespace cqt {

/**
 * Standard CQT implementation matching librosa.cqt
 *
 * Unlike HybridCQT, this does NOT use early downsampling.
 * All frequencies are processed at the original sample rate.
 * This is more accurate but slower than HybridCQT.
 *
 * Used by BTC chord recognition model.
 *
 * Default parameters (BTC):
 *   - sampleRate: 22050
 *   - hopLength: 2048
 *   - binsPerOctave: 24
 *   - nBins: 144
 *   - fmin: 32.7 (C1)
 */
class StandardCQT : public CQTBase {
public:
    // Constructor with default parameters for BTC model
    StandardCQT();

    // Constructor with custom parameters
    StandardCQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin);

    // Initialize the CQT (build filter banks)
    void initialize() override;

    // Compute CQT magnitude spectrogram
    std::vector<float> compute(const float* audio, size_t length) override;

    // Compute with progress callback
    std::vector<float> computeWithProgress(
        const float* audio,
        size_t length,
        ProgressCallback callback
    ) override;

    // Get frequencies for each bin
    std::vector<double> getFrequencies() const override;

    // Get filter lengths for each bin
    std::vector<int> getFilterLengths() const override;

private:
    // Filter bank (frequency domain) for all bins
    std::vector<std::vector<Complex>> filterBank_;

    // FFT size (single size for all bins since no downsampling)
    int fftSize_;

    // Filter lengths for each bin
    std::vector<int> filterLengths_;

    // Frequencies for each bin
    std::vector<double> frequencies_;

    // Build filter bank
    void buildFilterBank();

    // Core computation with optional progress callback
    std::vector<float> computeInternal(
        const float* audio,
        size_t length,
        ProgressCallback callback
    );
};

} // namespace cqt

#endif // CQT_STANDARD_CQT_HPP
