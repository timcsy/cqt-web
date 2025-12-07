#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <functional>
#include <string>

namespace cqt {

/**
 * Progress callback function type
 * @param progress Progress value between 0.0 and 1.0
 * @param stage Current processing stage description
 */
using ProgressCallback = std::function<void(float progress, const char* stage)>;

/**
 * Progress reporting mode
 */
enum class ProgressMode {
    None = 0,      // No progress reporting (default, fastest)
    Octave = 1,    // Report once per octave (8-10 reports typically)
    Percentage = 2 // Report at percentage intervals
};

/**
 * CQT computation options
 */
struct CQTOptions {
    // Progress reporting
    bool enableProgress = false;
    ProgressCallback onProgress = nullptr;
    ProgressMode progressMode = ProgressMode::Octave;
    int percentageStep = 10;  // Only used when progressMode == Percentage
};

} // namespace cqt

#endif // PROGRESS_HPP
