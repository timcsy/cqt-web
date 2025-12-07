#include "hybrid_cqt.hpp"
#include "../core/fft.hpp"
#include "../core/resample.hpp"
#include <algorithm>
#include <cstring>

namespace cqt {

HybridCQT::HybridCQT() {
    // Default parameters for CNN-LSTM model
    params_.sampleRate = 22050;
    params_.hopLength = 512;
    params_.binsPerOctave = 36;
    params_.nBins = 288;
    params_.fmin = 23.12;  // F#0 in Hz
}

HybridCQT::HybridCQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin) {
    params_.sampleRate = sampleRate;
    params_.hopLength = hopLength;
    params_.binsPerOctave = binsPerOctave;
    params_.nBins = nBins;
    params_.fmin = fmin;
}

void HybridCQT::initialize() {
    if (initialized_) return;

    double Q = params_.Q();
    int nOctaves = params_.nOctaves();

    // Compute frequencies and filter lengths for each bin at original sample rate
    frequencies_.resize(params_.nBins);
    filterLengths_.resize(params_.nBins);

    for (int k = 0; k < params_.nBins; k++) {
        frequencies_[k] = params_.fmin * std::pow(2.0, static_cast<double>(k) / params_.binsPerOctave);
        filterLengths_[k] = static_cast<int>(std::ceil(Q * params_.sampleRate / frequencies_[k]));
    }

    // Build filter banks for each octave
    filterBanks_.resize(nOctaves);
    fftSizes_.resize(nOctaves);

    for (int oct = 0; oct < nOctaves; oct++) {
        buildFilterBank(oct);
    }

    initialized_ = true;
}

void HybridCQT::buildFilterBank(int octave) {
    double Q = params_.Q();
    int bpo = params_.binsPerOctave;
    int nOctaves = params_.nOctaves();

    // Bins for this octave
    int binStart = octave * bpo;
    int binEnd = std::min((octave + 1) * bpo, params_.nBins);
    int nBinsOctave = binEnd - binStart;

    if (nBinsOctave <= 0) {
        filterBanks_[octave] = {};
        fftSizes_[octave] = 0;
        return;
    }

    // Processing order: octave n-1 first at full SR, then decimate
    int decimationCount = nOctaves - 1 - octave;
    double effectiveSr = params_.sampleRate / std::pow(2.0, decimationCount);

    // Compute filter lengths at effective sample rate
    std::vector<int> octaveLengths(nBinsOctave);
    int maxLength = 0;

    for (int i = 0; i < nBinsOctave; i++) {
        int k = binStart + i;
        int length = static_cast<int>(std::ceil(Q * effectiveSr / frequencies_[k]));
        octaveLengths[i] = std::max(length, 2);
        maxLength = std::max(maxLength, octaveLengths[i]);
    }

    // FFT size (next power of 2)
    int nfft = static_cast<int>(FFT::nextPow2(maxLength));
    fftSizes_[octave] = nfft;

    // Build filter for each bin in this octave
    filterBanks_[octave].resize(nBinsOctave);

    // VQT scale for this octave: sqrt(original_sr / effective_sr)
    double vqtScale = std::sqrt(params_.sampleRate / effectiveSr);

    for (int i = 0; i < nBinsOctave; i++) {
        int k = binStart + i;
        double freq = frequencies_[k];
        int length = octaveLengths[i];

        // Create time-domain filter: complex exponential * window
        // Center the kernel in the FFT buffer (like librosa's wavelet)
        std::vector<Complex> kernel(nfft, Complex(0, 0));

        // Calculate start index to center the kernel
        int startIdx = (nfft - length) / 2;

        // Build centered kernel
        double l1Norm = 0.0;

        for (int n = 0; n < length; n++) {
            // Centered time: t = (n - (length-1)/2) / sr
            double t = (n - (length - 1) / 2.0) / effectiveSr;

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

            // Place kernel at centered position
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
        // Also apply VQT scale here
        double scale = static_cast<double>(length) / nfft * vqtScale;
        for (auto& c : kernel) {
            c *= scale;
        }

        filterBanks_[octave][i] = std::move(kernel);
    }
}

std::vector<double> HybridCQT::getFrequencies() const {
    return frequencies_;
}

std::vector<int> HybridCQT::getFilterLengths() const {
    return filterLengths_;
}

std::vector<std::vector<Complex>> HybridCQT::computeOctave(
    const std::vector<double>& audio,
    int sr,
    int hopLength,
    int octave
) {
    int bpo = params_.binsPerOctave;

    // Bins for this octave
    int binStart = octave * bpo;
    int binEnd = std::min((octave + 1) * bpo, params_.nBins);
    int nBinsOctave = binEnd - binStart;

    if (nBinsOctave <= 0 || filterBanks_[octave].empty()) {
        return {};
    }

    int nfft = fftSizes_[octave];
    if (nfft == 0) return {};

    int numFrames = getNumFrames(audio.size(), hopLength);

    std::vector<std::vector<Complex>> result(numFrames, std::vector<Complex>(nBinsOctave));

    // For each frame, compute the CQT response
    for (int frame = 0; frame < numFrames; frame++) {
        int center = frame * hopLength;

        // Extract frame data centered at 'center'
        // NO window applied to audio frames (librosa uses window='ones')
        std::vector<Complex> frameData(nfft, Complex(0, 0));

        for (int j = 0; j < nfft; j++) {
            int idx = center - nfft / 2 + j;
            if (idx >= 0 && idx < static_cast<int>(audio.size())) {
                frameData[j] = Complex(audio[idx], 0);
            }
        }

        // FFT of frame
        FFT::forward(frameData);

        // For each CQT filter, compute dot product with frame FFT
        for (int i = 0; i < nBinsOctave; i++) {
            const auto& kernelFft = filterBanks_[octave][i];

            // Compute dot product: sum(frameData * conj(kernelFft))
            Complex dotProduct(0, 0);
            for (int j = 0; j < nfft; j++) {
                dotProduct += frameData[j] * std::conj(kernelFft[j]);
            }

            result[frame][i] = dotProduct;
        }
    }

    return result;
}

std::vector<float> HybridCQT::compute(const float* audio, size_t length) {
    return computeInternal(audio, length, nullptr);
}

std::vector<float> HybridCQT::computeWithProgress(
    const float* audio,
    size_t length,
    ProgressCallback callback
) {
    return computeInternal(audio, length, callback);
}

std::vector<float> HybridCQT::computeInternal(
    const float* audio,
    size_t length,
    ProgressCallback callback
) {
    if (!initialized_) {
        initialize();
    }

    // Convert to double
    std::vector<double> audioDouble(length);
    for (size_t i = 0; i < length; i++) {
        audioDouble[i] = static_cast<double>(audio[i]);
    }

    int nOctaves = params_.nOctaves();
    int bpo = params_.binsPerOctave;
    int numFrames = getNumFrames(length, params_.hopLength);

    // Result: complex CQT coefficients [numFrames x nBins]
    std::vector<std::vector<Complex>> cqtComplex(numFrames, std::vector<Complex>(params_.nBins, Complex(0, 0)));

    // Process octaves from HIGHEST to LOWEST frequency (librosa order)
    std::vector<double> currentAudio = audioDouble;
    int currentHop = params_.hopLength;

    // Report initial progress
    if (callback) {
        callback(0.0f, "Starting CQT computation");
    }

    // Process from octave n-1 (highest freq) down to octave 0 (lowest freq)
    for (int oct = nOctaves - 1; oct >= 0; oct--) {
        // Compute this octave's CQT
        auto octaveResult = computeOctave(currentAudio, 0, currentHop, oct);

        if (octaveResult.empty()) {
            // Decimate for next octave anyway
            if (oct > 0) {
                currentAudio = Resampler::decimate2(currentAudio);
                currentHop = std::max(1, currentHop / 2);
            }
            continue;
        }

        // Bins for this octave
        int binStart = oct * bpo;
        int binEnd = std::min((oct + 1) * bpo, params_.nBins);
        int nBinsOctave = binEnd - binStart;

        // Copy results directly
        int octaveFrames = static_cast<int>(octaveResult.size());
        for (int frame = 0; frame < std::min(numFrames, octaveFrames); frame++) {
            for (int i = 0; i < nBinsOctave; i++) {
                cqtComplex[frame][binStart + i] = octaveResult[frame][i];
            }
        }

        // Report progress (octave-based)
        if (callback) {
            float progress = static_cast<float>(nOctaves - oct) / nOctaves;
            callback(progress, "Processing octave");
        }

        // Decimate audio for next (lower frequency) octave
        if (oct > 0) {
            currentAudio = Resampler::decimate2(currentAudio);
            currentHop = std::max(1, currentHop / 2);
        }
    }

    // Apply final normalization and take magnitude
    std::vector<float> result(numFrames * params_.nBins);

    for (int frame = 0; frame < numFrames; frame++) {
        for (int k = 0; k < params_.nBins; k++) {
            double mag = std::abs(cqtComplex[frame][k]);
            // Divide by sqrt(filter_length_at_original_sr)
            mag /= std::sqrt(static_cast<double>(filterLengths_[k]));
            result[frame * params_.nBins + k] = static_cast<float>(mag);
        }
    }

    // Report completion
    if (callback) {
        callback(1.0f, "CQT computation complete");
    }

    return result;
}

} // namespace cqt
