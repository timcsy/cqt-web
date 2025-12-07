#include "convert.hpp"
#include <cctype>
#include <stdexcept>

namespace cqt {
namespace utils {

// Note names with sharps
static const char* NOTE_NAMES_SHARP[] = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};

// Note names with flats
static const char* NOTE_NAMES_FLAT[] = {
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"
};

double noteToHz(const std::string& note) {
    if (note.empty()) {
        return -1.0;
    }

    // Parse note name
    size_t pos = 0;
    char noteLetter = std::toupper(note[pos++]);

    // Check for valid note letter (A-G)
    if (noteLetter < 'A' || noteLetter > 'G') {
        return -1.0;
    }

    // Map note letter to pitch class (C = 0)
    // C=0, D=2, E=4, F=5, G=7, A=9, B=11
    static const int noteOffsets[] = { 9, 11, 0, 2, 4, 5, 7 }; // A, B, C, D, E, F, G
    int pitchClass = noteOffsets[noteLetter - 'A'];

    // Check for accidentals (# or b)
    int accidental = 0;
    if (pos < note.length()) {
        if (note[pos] == '#') {
            accidental = 1;
            pos++;
        } else if (note[pos] == 'b') {
            accidental = -1;
            pos++;
        }
    }

    pitchClass = (pitchClass + accidental + 12) % 12;

    // Parse octave number
    if (pos >= note.length()) {
        return -1.0;  // No octave specified
    }

    int octave = 0;
    bool negative = false;
    if (note[pos] == '-') {
        negative = true;
        pos++;
    }

    while (pos < note.length() && std::isdigit(note[pos])) {
        octave = octave * 10 + (note[pos] - '0');
        pos++;
    }

    if (negative) {
        octave = -octave;
    }

    // Calculate MIDI note number
    // C4 = 60, A4 = 69
    int midi = (octave + 1) * 12 + pitchClass;

    // Convert to Hz
    return midiToHz(midi);
}

std::string midiToNote(int midi, bool useSharps) {
    if (midi < 0 || midi > 127) {
        return "";
    }

    int octave = (midi / 12) - 1;
    int pitchClass = midi % 12;

    const char* noteName = useSharps ? NOTE_NAMES_SHARP[pitchClass] : NOTE_NAMES_FLAT[pitchClass];

    return std::string(noteName) + std::to_string(octave);
}

} // namespace utils
} // namespace cqt
