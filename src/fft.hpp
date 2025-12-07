#ifndef FFT_HPP
#define FFT_HPP

#include <complex>
#include <vector>
#include <cmath>

namespace cqt {

using Complex = std::complex<double>;

class FFT {
public:
    // Compute forward FFT (in-place)
    static void forward(std::vector<Complex>& data);

    // Compute inverse FFT (in-place)
    static void inverse(std::vector<Complex>& data);

    // Compute forward FFT of real data, returns complex result
    static std::vector<Complex> rfft(const std::vector<double>& real);
    static std::vector<Complex> rfft(const double* real, size_t length);
    static std::vector<Complex> rfft(const float* real, size_t length);

    // Helper: compute next power of 2
    static size_t nextPow2(size_t n);

private:
    // Cooley-Tukey radix-2 FFT
    static void fftRadix2(std::vector<Complex>& data, bool inverse);

    // Bit reversal permutation
    static void bitReverse(std::vector<Complex>& data);

    // Get bit-reversed index
    static size_t reverseBits(size_t x, int log2n);
};

} // namespace cqt

#endif // FFT_HPP
