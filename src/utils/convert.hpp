#ifndef CQT_UTILS_CONVERT_HPP
#define CQT_UTILS_CONVERT_HPP

#include <string>
#include <cmath>

namespace cqt {
namespace utils {

/**
 * Convert a note name to frequency in Hz
 * Format: "C4", "A#3", "Bb5", "F#0"
 * A4 = 440 Hz (standard tuning)
 *
 * @param note Note name string (e.g., "A4", "C#3", "Bb2")
 * @return Frequency in Hz, or -1 if invalid note
 */
double noteToHz(const std::string& note);

/**
 * Convert MIDI note number to frequency in Hz
 * MIDI 69 = A4 = 440 Hz
 *
 * @param midi MIDI note number (0-127)
 * @return Frequency in Hz
 */
inline double midiToHz(int midi) {
    return 440.0 * std::pow(2.0, (midi - 69) / 12.0);
}

/**
 * Convert frequency in Hz to MIDI note number
 * A4 = 440 Hz = MIDI 69
 *
 * @param hz Frequency in Hz
 * @return MIDI note number (fractional)
 */
inline double hzToMidi(double hz) {
    return 69.0 + 12.0 * std::log2(hz / 440.0);
}

/**
 * Convert MIDI note number to nearest integer
 *
 * @param hz Frequency in Hz
 * @return Nearest MIDI note number
 */
inline int hzToMidiRounded(double hz) {
    return static_cast<int>(std::round(hzToMidi(hz)));
}

/**
 * Convert frame indices to time in seconds
 *
 * @param frames Frame index
 * @param sr Sample rate
 * @param hopLength Hop length in samples
 * @return Time in seconds
 */
inline double framesToTime(int frames, int sr, int hopLength) {
    return static_cast<double>(frames * hopLength) / sr;
}

/**
 * Convert time in seconds to frame index
 *
 * @param time Time in seconds
 * @param sr Sample rate
 * @param hopLength Hop length in samples
 * @return Frame index
 */
inline int timeToFrames(double time, int sr, int hopLength) {
    return static_cast<int>(std::floor(time * sr / hopLength));
}

/**
 * Convert samples to time in seconds
 *
 * @param samples Number of samples
 * @param sr Sample rate
 * @return Time in seconds
 */
inline double samplesToTime(int samples, int sr) {
    return static_cast<double>(samples) / sr;
}

/**
 * Convert time in seconds to samples
 *
 * @param time Time in seconds
 * @param sr Sample rate
 * @return Number of samples
 */
inline int timeToSamples(double time, int sr) {
    return static_cast<int>(std::round(time * sr));
}

/**
 * Get MIDI note name from MIDI number
 *
 * @param midi MIDI note number
 * @param useSharps Use sharps (true) or flats (false)
 * @return Note name (e.g., "C4", "F#3", "Bb5")
 */
std::string midiToNote(int midi, bool useSharps = true);

} // namespace utils
} // namespace cqt

#endif // CQT_UTILS_CONVERT_HPP
