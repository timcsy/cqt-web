#include "standard_cqt.hpp"
#include "../core/fft.hpp"
#include <algorithm>
#include <cstring>

namespace cqt {

StandardCQT::StandardCQT() {
    // Default parameters for BTC model
    params_.sampleRate = 22050;
    params_.hopLength = 2048;
    params_.binsPerOctave = 24;
    params_.nBins = 144;
    params_.fmin = 32.7;  // C1 in Hz
}

StandardCQT::StandardCQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin) {
    params_.sampleRate = sampleRate;
    params_.hopLength = hopLength;
    params_.binsPerOctave = binsPerOctave;
    params_.nBins = nBins;
    params_.fmin = fmin;
}

void StandardCQT::initialize() {
    if (initialized_) return;

    double Q = params_.Q();

    // Compute frequencies and filter lengths for each bin
    frequencies_.resize(params_.nBins);
    filterLengths_.resize(params_.nBins);

    int maxLength = 0;
    for (int k = 0; k < params_.nBins; k++) {
        frequencies_[k] = params_.fmin * std::pow(2.0, static_cast<double>(k) / params_.binsPerOctave);
        filterLengths_[k] = static_cast<int>(std::ceil(Q * params_.sampleRate / frequencies_[k]));
        maxLength = std::max(maxLength, filterLengths_[k]);
    }

    // FFT size (single size for all - use next power of 2 of longest filter)
    fftSize_ = static_cast<int>(FFT::nextPow2(maxLength));

    // Build filter bank
    buildFilterBank();

    initialized_ = true;
}

void StandardCQT::buildFilterBank() {
    double Q = params_.Q();

    filterBank_.resize(params_.nBins);

    for (int k = 0; k < params_.nBins; k++) {
        double freq = frequencies_[k];
        int length = filterLengths_[k];

        // Create time-domain filter: complex exponential * window
        std::vector<Complex> kernel(fftSize_, Complex(0, 0));

        // Calculate start index to center the kernel
        int startIdx = (fftSize_ - length) / 2;

        // Build centered kernel
        double l1Norm = 0.0;

        for (int n = 0; n < length; n++) {
            // Centered time
            double t = (n - (length - 1) / 2.0) / params_.sampleRate;

            // Complex exponential
            double phase = 2.0 * M_PI * freq * t;
            Complex expVal(std::cos(phase), std::sin(phase));

            // Hann window
            double window;
            if (length > 1) {
                window = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (length - 1)));
            } else {
                window = 1.0;
            }

            kernel[startIdx + n] = expVal * window;
            l1Norm += std::abs(kernel[startIdx + n]);
        }

        // L1 normalize
        if (l1Norm > 0) {
            for (int n = 0; n < length; n++) {
                kernel[startIdx + n] /= l1Norm;
            }
        }

        // FFT of kernel
        FFT::forward(kernel);

        // Scale by length / nfft (librosa convention)
        double scale = static_cast<double>(length) / fftSize_;
        for (auto& c : kernel) {
            c *= scale;
        }

        filterBank_[k] = std::move(kernel);
    }
}

std::vector<double> StandardCQT::getFrequencies() const {
    return frequencies_;
}

std::vector<int> StandardCQT::getFilterLengths() const {
    return filterLengths_;
}

std::vector<float> StandardCQT::compute(const float* audio, size_t length) {
    return computeInternal(audio, length, nullptr);
}

std::vector<float> StandardCQT::computeWithProgress(
    const float* audio,
    size_t length,
    ProgressCallback callback
) {
    return computeInternal(audio, length, callback);
}

std::vector<float> StandardCQT::computeInternal(
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
        callback(0.0f, "Starting CQT computation");
    }

    // For each frame
    for (int frame = 0; frame < numFrames; frame++) {
        int center = frame * params_.hopLength;

        // Extract frame data centered at 'center'
        std::vector<Complex> frameData(fftSize_, Complex(0, 0));

        for (int j = 0; j < fftSize_; j++) {
            int idx = center - fftSize_ / 2 + j;
            if (idx >= 0 && idx < static_cast<int>(length)) {
                frameData[j] = Complex(static_cast<double>(audio[idx]), 0);
            }
        }

        // FFT of frame
        FFT::forward(frameData);

        // For each CQT bin
        for (int k = 0; k < params_.nBins; k++) {
            const auto& kernelFft = filterBank_[k];

            // Compute dot product: sum(frameData * conj(kernelFft))
            Complex dotProduct(0, 0);
            for (int j = 0; j < fftSize_; j++) {
                dotProduct += frameData[j] * std::conj(kernelFft[j]);
            }

            // Take magnitude and normalize
            double mag = std::abs(dotProduct);
            mag /= std::sqrt(static_cast<double>(filterLengths_[k]));

            result[frame * params_.nBins + k] = static_cast<float>(mag);
        }

        // Report progress (frame-based, but only every 10% to reduce overhead)
        if (callback && (frame % std::max(1, numFrames / 10) == 0 || frame == numFrames - 1)) {
            float progress = static_cast<float>(frame + 1) / numFrames;
            callback(progress, "Processing frames");
        }
    }

    // Report completion
    if (callback) {
        callback(1.0f, "CQT computation complete");
    }

    return result;
}

} // namespace cqt
