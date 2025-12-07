#ifndef CQT_CORE_RESAMPLE_HPP
#define CQT_CORE_RESAMPLE_HPP

#include <vector>
#include <cmath>

namespace cqt {

class Resampler {
public:
    // Decimate by factor of 2 with anti-aliasing filter
    // Uses a high-quality FIR lowpass filter
    static std::vector<double> decimate2(const std::vector<double>& input);
    static std::vector<double> decimate2(const double* input, size_t length);

    // Resample by arbitrary ratio (simple linear interpolation for upsampling)
    static std::vector<double> resample(const std::vector<double>& input,
                                         int origSr, int targetSr);

private:
    // Half-band FIR filter coefficients for decimation
    // This is a 31-tap half-band filter with good stopband attenuation
    static const std::vector<double>& getHalfBandFilter();

    // Apply FIR filter
    static std::vector<double> applyFIR(const std::vector<double>& input,
                                         const std::vector<double>& coeffs);
};

} // namespace cqt

#endif // CQT_CORE_RESAMPLE_HPP
