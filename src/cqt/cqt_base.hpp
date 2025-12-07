#ifndef CQT_CQT_BASE_HPP
#define CQT_CQT_BASE_HPP

#include <vector>
#include <complex>
#include <cmath>
#include "../progress.hpp"

namespace cqt {

using Complex = std::complex<double>;

/**
 * CQT parameters structure
 */
struct CQTParams {
    int sampleRate = 22050;
    int hopLength = 512;
    int binsPerOctave = 36;
    int nBins = 288;
    double fmin = 23.12;  // F#0

    // Computed parameters
    double Q() const {
        return 1.0 / (std::pow(2.0, 1.0 / binsPerOctave) - 1);
    }

    int nOctaves() const {
        return static_cast<int>(std::ceil(static_cast<double>(nBins) / binsPerOctave));
    }

    double fmax() const {
        return fmin * std::pow(2.0, static_cast<double>(nBins - 1) / binsPerOctave);
    }
};

/**
 * Abstract base class for CQT implementations
 */
class CQTBase {
public:
    virtual ~CQTBase() = default;

    // Initialize the CQT (build filter banks)
    virtual void initialize() = 0;

    // Compute CQT magnitude spectrogram
    // Returns: [numFrames * nBins] flattened array (row-major, frames first)
    virtual std::vector<float> compute(const float* audio, size_t length) = 0;

    std::vector<float> compute(const std::vector<float>& audio) {
        return compute(audio.data(), audio.size());
    }

    // Compute with progress callback
    virtual std::vector<float> computeWithProgress(
        const float* audio,
        size_t length,
        ProgressCallback callback
    ) {
        // Default implementation: just call compute without progress
        return compute(audio, length);
    }

    std::vector<float> computeWithProgress(
        const std::vector<float>& audio,
        ProgressCallback callback
    ) {
        return computeWithProgress(audio.data(), audio.size(), callback);
    }

    // Get parameters
    const CQTParams& params() const { return params_; }

    // Get frequencies for each bin
    virtual std::vector<double> getFrequencies() const = 0;

    // Get filter lengths for each bin
    virtual std::vector<int> getFilterLengths() const = 0;

    // Get output shape [numFrames, nBins] for given audio length
    int getNumFrames(size_t audioLength) const {
        return getNumFrames(audioLength, params_.hopLength);
    }

protected:
    CQTParams params_;
    bool initialized_ = false;

    // Get number of frames for given audio length
    int getNumFrames(size_t audioLength, int hopLength) const {
        // librosa uses center=True by default
        return static_cast<int>(std::ceil(static_cast<double>(audioLength) / hopLength));
    }
};

} // namespace cqt

#endif // CQT_CQT_BASE_HPP
