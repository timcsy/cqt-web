#include "cqt.hpp"
#include "fft.hpp"
#include "resample.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cqt {

CQT::CQT() {
    // Default parameters for CNN-LSTM model
    params_.sampleRate = 22050;
    params_.hopLength = 512;
    params_.binsPerOctave = 36;
    params_.nBins = 288;
    params_.fmin = 23.12;  // F#0 in Hz
}

CQT::CQT(int sampleRate, int hopLength, int binsPerOctave, int nBins, double fmin) {
    params_.sampleRate = sampleRate;
    params_.hopLength = hopLength;
    params_.binsPerOctave = binsPerOctave;
    params_.nBins = nBins;
    params_.fmin = fmin;
}

void CQT::initialize() {
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
    // librosa processes from HIGHEST octave to LOWEST:
    // - First: octave n-1 (highest freq bins) at full SR
    // - Then: decimate audio, process octave n-2 at SR/2
    // - etc.
    // But the filter lengths are computed at each octave's effective SR
    filterBanks_.resize(nOctaves);
    fftSizes_.resize(nOctaves);

    for (int oct = 0; oct < nOctaves; oct++) {
        buildFilterBank(oct);
    }

    initialized_ = true;
}

void CQT::buildFilterBank(int octave) {
    // In librosa's processing:
    // - Octave n-1 (highest freq) is processed FIRST at full SR
    // - Octave n-2 is processed at SR/2 (after first decimation)
    // - ...
    // - Octave 0 (lowest freq) is processed LAST at SR/2^(n-1)

    // This means:
    // - Octave 0: effective SR = SR / 2^(n-1)
    // - Octave k: effective SR = SR / 2^(n-1-k)

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
    // Octave n-1: processed at SR (0 decimations done)
    // Octave n-2: processed at SR/2 (1 decimation done)
    // Octave k: processed at SR / 2^(n-1-k)
    int decimationCount = nOctaves - 1 - octave;
    double effectiveSr = params_.sampleRate / std::pow(2.0, decimationCount);

    // Compute filter lengths at effective sample rate
    std::vector<int> octaveLengths(nBinsOctave);
    int maxLength = 0;

    for (int i = 0; i < nBinsOctave; i++) {
        int k = binStart + i;
        int length = static_cast<int>(std::ceil(Q * effectiveSr / frequencies_[k]));
        octaveLengths[i] = std::max(length, 2);  // At least 2 samples
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
        // IMPORTANT: Center the kernel in the FFT buffer (like librosa's wavelet)
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
        // Also apply VQT scale here (as librosa does)
        double scale = static_cast<double>(length) / nfft * vqtScale;
        for (auto& c : kernel) {
            c *= scale;
        }

        filterBanks_[octave][i] = std::move(kernel);
    }
}

std::vector<double> CQT::getFrequencies() const {
    return frequencies_;
}

std::vector<int> CQT::getFilterLengths() const {
    return filterLengths_;
}

int CQT::getNumFrames(size_t audioLength, int hopLength) const {
    // librosa uses center=True by default
    return static_cast<int>(std::ceil(static_cast<double>(audioLength) / hopLength));
}

std::vector<double> CQT::earlyDownsample(const std::vector<double>& audio, int targetSr) {
    int currentSr = params_.sampleRate;
    std::vector<double> result = audio;

    while (currentSr > targetSr * 2) {
        result = Resampler::decimate2(result);
        currentSr /= 2;
    }

    return result;
}

std::vector<std::vector<Complex>> CQT::computeOctave(
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

    // NOTE: librosa's CQT uses window='ones' (no window) in __cqt_response
    // The window is already baked into the CQT kernel

    // For each frame, compute the CQT response using STFT-style computation
    // This matches librosa's approach: fft_basis @ stft
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

        // FFT of frame (no window)
        FFT::forward(frameData);

        // For each CQT filter, compute dot product with frame FFT
        // This is: sum(stft_full * conj(kernel_fft))
        // In librosa, this is done as: fft_basis @ stft (matrix multiplication)
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

std::vector<float> CQT::compute(const std::vector<float>& audio) {
    return compute(audio.data(), audio.size());
}

std::vector<float> CQT::compute(const float* audio, size_t length) {
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
    // Start with full audio, decimate after each octave
    std::vector<double> currentAudio = audioDouble;
    int currentHop = params_.hopLength;

    // Process from octave n-1 (highest freq) down to octave 0 (lowest freq)
    for (int oct = nOctaves - 1; oct >= 0; oct--) {
        // Compute this octave's CQT
        // Note: VQT scale is already applied in the filter bank during buildFilterBank()
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

        // Copy results directly - no frame mapping needed!
        // Both audio and hop_length are decimated together, so each octave
        // produces the same number of frames.
        // VQT scale is already applied in the filter bank, so no additional scaling needed
        int octaveFrames = static_cast<int>(octaveResult.size());
        for (int frame = 0; frame < std::min(numFrames, octaveFrames); frame++) {
            for (int i = 0; i < nBinsOctave; i++) {
                cqtComplex[frame][binStart + i] = octaveResult[frame][i];
            }
        }

        // Decimate audio for next (lower frequency) octave
        if (oct > 0) {
            currentAudio = Resampler::decimate2(currentAudio);
            currentHop = std::max(1, currentHop / 2);
        }
    }

    // Apply final normalization and take magnitude
    // librosa normalizes by dividing by sqrt(filter_length_at_original_sr)
    // filterLengths_[k] is already computed at original SR

    std::vector<float> result(numFrames * params_.nBins);

    for (int frame = 0; frame < numFrames; frame++) {
        for (int k = 0; k < params_.nBins; k++) {
            double mag = std::abs(cqtComplex[frame][k]);
            // Divide by sqrt(filter_length_at_original_sr) - librosa's normalization
            mag /= std::sqrt(static_cast<double>(filterLengths_[k]));

            result[frame * params_.nBins + k] = static_cast<float>(mag);
        }
    }

    return result;
}

} // namespace cqt
