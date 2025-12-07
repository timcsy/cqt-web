#include "fft.hpp"
#include <stdexcept>

namespace cqt {

size_t FFT::nextPow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

size_t FFT::reverseBits(size_t x, int log2n) {
    size_t result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void FFT::bitReverse(std::vector<Complex>& data) {
    size_t n = data.size();
    int log2n = 0;
    while ((1u << log2n) < n) log2n++;

    for (size_t i = 0; i < n; i++) {
        size_t j = reverseBits(i, log2n);
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

void FFT::fftRadix2(std::vector<Complex>& data, bool inverse) {
    size_t n = data.size();

    // Bit-reverse permutation
    bitReverse(data);

    // Cooley-Tukey iterative FFT
    for (size_t len = 2; len <= n; len *= 2) {
        double angle = 2.0 * M_PI / len;
        if (inverse) angle = -angle;

        Complex wn(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; j++) {
                Complex u = data[i + j];
                Complex t = w * data[i + j + len / 2];
                data[i + j] = u + t;
                data[i + j + len / 2] = u - t;
                w *= wn;
            }
        }
    }

    // Scale for inverse FFT
    if (inverse) {
        for (auto& x : data) {
            x /= static_cast<double>(n);
        }
    }
}

void FFT::forward(std::vector<Complex>& data) {
    size_t n = data.size();
    if (n == 0) return;

    // Ensure power of 2
    size_t n2 = nextPow2(n);
    if (n2 != n) {
        data.resize(n2, Complex(0, 0));
    }

    fftRadix2(data, false);
}

void FFT::inverse(std::vector<Complex>& data) {
    size_t n = data.size();
    if (n == 0) return;

    // Ensure power of 2
    size_t n2 = nextPow2(n);
    if (n2 != n) {
        data.resize(n2, Complex(0, 0));
    }

    fftRadix2(data, true);
}

std::vector<Complex> FFT::rfft(const std::vector<double>& real) {
    return rfft(real.data(), real.size());
}

std::vector<Complex> FFT::rfft(const double* real, size_t length) {
    size_t n = nextPow2(length);
    std::vector<Complex> data(n, Complex(0, 0));

    for (size_t i = 0; i < length; i++) {
        data[i] = Complex(real[i], 0);
    }

    forward(data);
    return data;
}

std::vector<Complex> FFT::rfft(const float* real, size_t length) {
    size_t n = nextPow2(length);
    std::vector<Complex> data(n, Complex(0, 0));

    for (size_t i = 0; i < length; i++) {
        data[i] = Complex(static_cast<double>(real[i]), 0);
    }

    forward(data);
    return data;
}

std::vector<double> FFT::forwardReal(const std::vector<double>& real) {
    auto complexResult = rfft(real);
    // Return interleaved [real0, imag0, real1, imag1, ...]
    std::vector<double> result(complexResult.size() * 2);
    for (size_t i = 0; i < complexResult.size(); i++) {
        result[i * 2] = complexResult[i].real();
        result[i * 2 + 1] = complexResult[i].imag();
    }
    return result;
}

std::vector<double> FFT::inverseComplex(const std::vector<double>& realPart,
                                         const std::vector<double>& imagPart) {
    if (realPart.size() != imagPart.size()) {
        return {};
    }

    std::vector<Complex> data(realPart.size());
    for (size_t i = 0; i < realPart.size(); i++) {
        data[i] = Complex(realPart[i], imagPart[i]);
    }

    inverse(data);

    std::vector<double> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i].real();
    }
    return result;
}

} // namespace cqt
