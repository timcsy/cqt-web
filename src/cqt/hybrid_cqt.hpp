#ifndef CQT_HYBRID_CQT_HPP
#define CQT_HYBRID_CQT_HPP

#include "cqt_base.hpp"
#include <vector>
#include <complex>

namespace cqt {

/**
 * Hybrid CQT implementation matching librosa.hybrid_cqt
 *
 * Uses early downsampling for lower octaves to improve efficiency.
 * This is the default for CNN-LSTM chord recognition model.
 *
 * Default parameters (CNN-LSTM):
 *   - sampleRate: 22050
 *   - hopLength: 512
 *   - binsPerOctave: 36
 *   - nBins: 288
 *   - fmin: 23.12 (F#0)
 */
class HybridCQT : public CQTBase {
public:
    // Constructor with default parameters for CNN-LSTM model
    HybridCQT();

    // Constructor with custom parameters
    HybridCQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin);

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
    // Filter banks for each octave (frequency domain)
    std::vector<std::vector<std::vector<Complex>>> filterBanks_;

    // FFT sizes for each octave
    std::vector<int> fftSizes_;

    // Filter lengths for each bin
    std::vector<int> filterLengths_;

    // Frequencies for each bin
    std::vector<double> frequencies_;

    // Build filter bank for a specific octave
    void buildFilterBank(int octave);

    // Compute a single octave's CQT
    std::vector<std::vector<Complex>> computeOctave(
        const std::vector<double>& audio,
        int sr,
        int hopLength,
        int octave
    );

    // Core computation with optional progress callback
    std::vector<float> computeInternal(
        const float* audio,
        size_t length,
        ProgressCallback callback
    );
};

} // namespace cqt

#endif // CQT_HYBRID_CQT_HPP
