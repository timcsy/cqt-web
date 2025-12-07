#ifndef CQT_HPP
#define CQT_HPP

#include <vector>
#include <complex>
#include <cmath>

namespace cqt {

using Complex = std::complex<double>;

// CQT parameters structure
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
};

class CQT {
public:
    // Constructor with default parameters for CNN-LSTM model
    CQT();

    // Constructor with custom parameters
    CQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin);

    // Initialize the CQT (build filter banks)
    void initialize();

    // Compute CQT magnitude spectrogram
    // Returns: [numFrames * nBins] flattened array (row-major, frames first)
    std::vector<float> compute(const float* audio, size_t length);
    std::vector<float> compute(const std::vector<float>& audio);

    // Get parameters
    const CQTParams& params() const { return params_; }

    // Get frequencies for each bin
    std::vector<double> getFrequencies() const;

    // Get filter lengths for each bin
    std::vector<int> getFilterLengths() const;

private:
    CQTParams params_;
    bool initialized_ = false;

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
    // Returns complex CQT coefficients [numFrames x binsInOctave]
    std::vector<std::vector<Complex>> computeOctave(
        const std::vector<double>& audio,
        int sr,
        int hopLength,
        int octave
    );

    // Apply early downsampling if needed
    std::vector<double> earlyDownsample(const std::vector<double>& audio, int targetSr);

    // Get number of frames for given audio length
    int getNumFrames(size_t audioLength, int hopLength) const;
};

} // namespace cqt

#endif // CQT_HPP
