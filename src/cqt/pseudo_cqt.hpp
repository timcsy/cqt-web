#ifndef CQT_PSEUDO_CQT_HPP
#define CQT_PSEUDO_CQT_HPP

#include "cqt_base.hpp"
#include <vector>
#include <complex>

namespace cqt {

/**
 * Pseudo CQT implementation matching librosa.pseudo_cqt
 *
 * Uses STFT-based approach with frequency mapping.
 * Faster than standard CQT but less accurate at low frequencies.
 *
 * The pseudo CQT computes an STFT first, then maps the frequency bins
 * to CQT frequency bins using interpolation.
 */
class PseudoCQT : public CQTBase {
public:
    // Constructor with default parameters
    PseudoCQT();

    // Constructor with custom parameters
    PseudoCQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin);

    // Initialize (build frequency mapping)
    void initialize() override;

    // Compute pseudo-CQT magnitude spectrogram
    std::vector<float> compute(const float* audio, size_t length) override;

    // Compute with progress callback
    std::vector<float> computeWithProgress(
        const float* audio,
        size_t length,
        ProgressCallback callback
    ) override;

    // Get frequencies for each bin
    std::vector<double> getFrequencies() const override;

    // Get filter lengths for each bin (approximation based on Q)
    std::vector<int> getFilterLengths() const override;

private:
    // Frequencies for each CQT bin
    std::vector<double> frequencies_;

    // Approximate filter lengths
    std::vector<int> filterLengths_;

    // FFT size for STFT
    int fftSize_;

    // Frequency mapping from STFT bins to CQT bins
    // For each CQT bin, stores the STFT bin indices and weights for interpolation
    struct FreqMapping {
        int lowBin;
        int highBin;
        double weight;  // Weight for interpolation (0 = all low, 1 = all high)
    };
    std::vector<FreqMapping> freqMapping_;

    // Core computation
    std::vector<float> computeInternal(
        const float* audio,
        size_t length,
        ProgressCallback callback
    );
};

} // namespace cqt

#endif // CQT_PSEUDO_CQT_HPP
