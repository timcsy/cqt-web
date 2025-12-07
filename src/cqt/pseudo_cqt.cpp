#include "pseudo_cqt.hpp"
#include "../core/fft.hpp"
#include <algorithm>
#include <cstring>

namespace cqt {

PseudoCQT::PseudoCQT() {
    // Default parameters (same as HybridCQT)
    params_.sampleRate = 22050;
    params_.hopLength = 512;
    params_.binsPerOctave = 36;
    params_.nBins = 288;
    params_.fmin = 23.12;  // F#0 in Hz
}

PseudoCQT::PseudoCQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin) {
    params_.sampleRate = sampleRate;
    params_.hopLength = hopLength;
    params_.binsPerOctave = binsPerOctave;
    params_.nBins = nBins;
    params_.fmin = fmin;
}

void PseudoCQT::initialize() {
    if (initialized_) return;

    double Q = params_.Q();

    // Compute frequencies for each CQT bin
    frequencies_.resize(params_.nBins);
    filterLengths_.resize(params_.nBins);

    int maxLength = 0;
    for (int k = 0; k < params_.nBins; k++) {
        frequencies_[k] = params_.fmin * std::pow(2.0, static_cast<double>(k) / params_.binsPerOctave);
        filterLengths_[k] = static_cast<int>(std::ceil(Q * params_.sampleRate / frequencies_[k]));
        maxLength = std::max(maxLength, filterLengths_[k]);
    }

    // Determine FFT size based on maximum filter length
    // Use a larger FFT for better frequency resolution at low frequencies
    fftSize_ = static_cast<int>(FFT::nextPow2(maxLength * 2));

    // Build frequency mapping from STFT bins to CQT bins
    freqMapping_.resize(params_.nBins);

    double freqPerBin = static_cast<double>(params_.sampleRate) / fftSize_;

    for (int k = 0; k < params_.nBins; k++) {
        double targetFreq = frequencies_[k];

        // Find the two nearest STFT bins
        double binPos = targetFreq / freqPerBin;
        int lowBin = static_cast<int>(std::floor(binPos));
        int highBin = lowBin + 1;

        // Clamp to valid range
        lowBin = std::max(0, std::min(lowBin, fftSize_ / 2));
        highBin = std::max(0, std::min(highBin, fftSize_ / 2));

        // Calculate interpolation weight
        double weight = binPos - lowBin;

        freqMapping_[k] = { lowBin, highBin, weight };
    }

    initialized_ = true;
}

std::vector<double> PseudoCQT::getFrequencies() const {
    return frequencies_;
}

std::vector<int> PseudoCQT::getFilterLengths() const {
    return filterLengths_;
}

std::vector<float> PseudoCQT::compute(const float* audio, size_t length) {
    return computeInternal(audio, length, nullptr);
}

std::vector<float> PseudoCQT::computeWithProgress(
    const float* audio,
    size_t length,
    ProgressCallback callback
) {
    return computeInternal(audio, length, callback);
}

std::vector<float> PseudoCQT::computeInternal(
    const float* audio,
    size_t length,
    ProgressCallback callback
) {
    if (!initialized_) {
        initialize();
    }

    int numFrames = getNumFrames(length, params_.hopLength);

    // Result
    std::vector<float> result(numFrames * params_.nBins);

    // Report initial progress
    if (callback) {
        callback(0.0f, "Starting Pseudo-CQT computation");
    }

    // Hann window
    std::vector<double> window(fftSize_);
    for (int n = 0; n < fftSize_; n++) {
        window[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (fftSize_ - 1)));
    }

    // For each frame
    for (int frame = 0; frame < numFrames; frame++) {
        int center = frame * params_.hopLength;

        // Extract and window frame data
        std::vector<Complex> frameData(fftSize_, Complex(0, 0));

        for (int j = 0; j < fftSize_; j++) {
            int idx = center - fftSize_ / 2 + j;
            if (idx >= 0 && idx < static_cast<int>(length)) {
                frameData[j] = Complex(static_cast<double>(audio[idx]) * window[j], 0);
            }
        }

        // FFT
        FFT::forward(frameData);

        // Map STFT bins to CQT bins using interpolation
        for (int k = 0; k < params_.nBins; k++) {
            const auto& mapping = freqMapping_[k];

            // Get magnitudes of neighboring STFT bins
            double magLow = std::abs(frameData[mapping.lowBin]);
            double magHigh = std::abs(frameData[mapping.highBin]);

            // Linear interpolation of magnitudes
            double mag = magLow * (1.0 - mapping.weight) + magHigh * mapping.weight;

            // Normalize (similar to standard CQT normalization)
            mag /= std::sqrt(static_cast<double>(filterLengths_[k]));

            result[frame * params_.nBins + k] = static_cast<float>(mag);
        }

        // Report progress
        if (callback && (frame % std::max(1, numFrames / 10) == 0 || frame == numFrames - 1)) {
            float progress = static_cast<float>(frame + 1) / numFrames;
            callback(progress, "Processing frames");
        }
    }

    // Report completion
    if (callback) {
        callback(1.0f, "Pseudo-CQT computation complete");
    }

    return result;
}

} // namespace cqt
