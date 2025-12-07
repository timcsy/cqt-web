#ifndef CQT_VQT_HPP
#define CQT_VQT_HPP

#include "cqt_base.hpp"
#include <vector>
#include <complex>

namespace cqt {

/**
 * Variable-Q Transform (VQT) implementation matching librosa.vqt
 *
 * Unlike CQT which has constant Q across all frequencies,
 * VQT allows the Q factor to vary with gamma parameter.
 *
 * Q(k) = Q0 * (1 + gamma / freq(k))
 *
 * When gamma = 0, VQT is equivalent to CQT.
 * Higher gamma values give more frequency resolution at low frequencies.
 */
class VQT : public CQTBase {
public:
    // Constructor with default parameters
    VQT();

    // Constructor with custom parameters
    VQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin, double gamma = 0.0);

    // Initialize the VQT
    void initialize() override;

    // Compute VQT magnitude spectrogram
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

    // Get gamma parameter
    double gamma() const { return gamma_; }

    // Get Q factor for a specific bin
    double Q(int bin) const;

private:
    // Gamma parameter for variable Q
    double gamma_;

    // Filter banks for each octave (frequency domain)
    std::vector<std::vector<std::vector<Complex>>> filterBanks_;

    // FFT sizes for each octave
    std::vector<int> fftSizes_;

    // Filter lengths for each bin (variable due to gamma)
    std::vector<int> filterLengths_;

    // Frequencies for each bin
    std::vector<double> frequencies_;

    // Build filter bank for a specific octave
    void buildFilterBank(int octave);

    // Compute a single octave's VQT
    std::vector<std::vector<Complex>> computeOctave(
        const std::vector<double>& audio,
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

#endif // CQT_VQT_HPP
