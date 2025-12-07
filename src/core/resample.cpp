#include "resample.hpp"
#include <algorithm>

namespace cqt {

// 31-tap half-band lowpass filter designed for decimation by 2
// Cutoff at 0.25 (Nyquist = 0.5), stopband attenuation > 60dB
// Generated using Parks-McClellan algorithm
// Coefficients normalized to have DC gain = 1.0
const std::vector<double>& Resampler::getHalfBandFilter() {
    // Original sum was 1.191372, divide by this to normalize
    static const double norm = 1.191372;
    static const std::vector<double> coeffs = {
        -0.000152738389867 / norm,
         0.000000000000000,
         0.001295605969418 / norm,
         0.000000000000000,
        -0.005070021918710 / norm,
         0.000000000000000,
         0.014326227968250 / norm,
         0.000000000000000,
        -0.033268038533270 / norm,
         0.000000000000000,
         0.068185409802030 / norm,
         0.000000000000000,
        -0.136770943049600 / norm,
         0.000000000000000,
         0.437140576168100 / norm,
         0.500000000000000 / norm,
         0.437140576168100 / norm,
         0.000000000000000,
        -0.136770943049600 / norm,
         0.000000000000000,
         0.068185409802030 / norm,
         0.000000000000000,
        -0.033268038533270 / norm,
         0.000000000000000,
         0.014326227968250 / norm,
         0.000000000000000,
        -0.005070021918710 / norm,
         0.000000000000000,
         0.001295605969418 / norm,
         0.000000000000000,
        -0.000152738389867 / norm
    };
    return coeffs;
}

std::vector<double> Resampler::applyFIR(const std::vector<double>& input,
                                         const std::vector<double>& coeffs) {
    size_t inputLen = input.size();
    size_t filterLen = coeffs.size();
    size_t outputLen = inputLen;

    std::vector<double> output(outputLen, 0.0);
    int halfFilter = static_cast<int>(filterLen) / 2;

    for (size_t i = 0; i < outputLen; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < filterLen; j++) {
            int idx = static_cast<int>(i) - halfFilter + static_cast<int>(j);
            if (idx >= 0 && idx < static_cast<int>(inputLen)) {
                sum += input[idx] * coeffs[j];
            }
        }
        output[i] = sum;
    }

    return output;
}

std::vector<double> Resampler::decimate2(const std::vector<double>& input) {
    return decimate2(input.data(), input.size());
}

std::vector<double> Resampler::decimate2(const double* input, size_t length) {
    // Convert to vector for FIR processing
    std::vector<double> inputVec(input, input + length);

    // Apply anti-aliasing filter
    const auto& filter = getHalfBandFilter();
    std::vector<double> filtered = applyFIR(inputVec, filter);

    // Downsample by 2
    size_t outputLen = (length + 1) / 2;
    std::vector<double> output(outputLen);

    // Apply energy-preserving scale factor (like librosa with scale=True)
    // When downsampling by 2, multiply by sqrt(2) to preserve energy
    const double scaleFactor = std::sqrt(2.0);

    for (size_t i = 0; i < outputLen; i++) {
        output[i] = filtered[i * 2] * scaleFactor;
    }

    return output;
}

std::vector<double> Resampler::resample(const std::vector<double>& input,
                                         int origSr, int targetSr) {
    if (origSr == targetSr) {
        return input;
    }

    double ratio = static_cast<double>(targetSr) / origSr;
    size_t outputLen = static_cast<size_t>(std::ceil(input.size() * ratio));
    std::vector<double> output(outputLen);

    for (size_t i = 0; i < outputLen; i++) {
        double srcPos = i / ratio;
        size_t srcIdx = static_cast<size_t>(srcPos);
        double frac = srcPos - srcIdx;

        if (srcIdx + 1 < input.size()) {
            // Linear interpolation
            output[i] = input[srcIdx] * (1.0 - frac) + input[srcIdx + 1] * frac;
        } else if (srcIdx < input.size()) {
            output[i] = input[srcIdx];
        } else {
            output[i] = 0.0;
        }
    }

    return output;
}

} // namespace cqt
