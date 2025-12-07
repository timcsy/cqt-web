#include "../src/cqt/hybrid_cqt.hpp"
#include "../src/cqt/standard_cqt.hpp"
#include "../src/cqt/pseudo_cqt.hpp"
#include "../src/cqt/vqt.hpp"
#include "../src/core/fft.hpp"
#include "../src/core/resample.hpp"
#include "../src/utils/convert.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace cqt;

// Generate C major chord test audio
std::vector<float> generateTestAudio(int sr, double duration) {
    int numSamples = static_cast<int>(sr * duration);
    std::vector<float> audio(numSamples);

    const double c4 = 261.63;
    const double e4 = 329.63;
    const double g4 = 392.00;

    for (int i = 0; i < numSamples; i++) {
        double t = static_cast<double>(i) / sr;
        audio[i] = static_cast<float>(
            0.3 * std::sin(2.0 * M_PI * c4 * t) +
            0.3 * std::sin(2.0 * M_PI * e4 * t) +
            0.3 * std::sin(2.0 * M_PI * g4 * t)
        );
    }

    return audio;
}

void testFFT() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing FFT" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test with simple sine wave
    int n = 256;
    std::vector<Complex> data(n);
    for (int i = 0; i < n; i++) {
        // 10 Hz sine wave at 256 Hz sample rate -> peak at bin 10
        data[i] = Complex(std::sin(2.0 * M_PI * 10 * i / n), 0);
    }

    FFT::forward(data);

    // Find peak
    int peakBin = 0;
    double peakMag = 0;
    for (int i = 0; i < n / 2; i++) {
        double mag = std::abs(data[i]);
        if (mag > peakMag) {
            peakMag = mag;
            peakBin = i;
        }
    }

    std::cout << "Expected peak at bin 10, found at bin " << peakBin << std::endl;
    std::cout << "Peak magnitude: " << peakMag << std::endl;

    // Verify inverse FFT
    FFT::inverse(data);
    double error = 0;
    for (int i = 0; i < n; i++) {
        double expected = std::sin(2.0 * M_PI * 10 * i / n);
        error += std::abs(data[i].real() - expected);
    }
    std::cout << "Inverse FFT total error: " << error << std::endl;
    std::cout << "FFT test " << (peakBin == 10 && error < 1e-10 ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;
}

void testUtilityFunctions() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing Utility Functions" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test noteToHz
    std::cout << "noteToHz tests:" << std::endl;
    std::cout << "  A4 = " << utils::noteToHz("A4") << " Hz (expected 440.0)" << std::endl;
    std::cout << "  C4 = " << utils::noteToHz("C4") << " Hz (expected ~261.63)" << std::endl;
    std::cout << "  F#0 = " << utils::noteToHz("F#0") << " Hz (expected ~23.12)" << std::endl;
    std::cout << "  Bb3 = " << utils::noteToHz("Bb3") << " Hz (expected ~233.08)" << std::endl;

    // Test midiToHz
    std::cout << "\nmidiToHz tests:" << std::endl;
    std::cout << "  MIDI 69 = " << utils::midiToHz(69) << " Hz (expected 440.0)" << std::endl;
    std::cout << "  MIDI 60 = " << utils::midiToHz(60) << " Hz (expected ~261.63)" << std::endl;

    // Test hzToMidi
    std::cout << "\nhzToMidi tests:" << std::endl;
    std::cout << "  440 Hz = MIDI " << utils::hzToMidi(440.0) << " (expected 69.0)" << std::endl;
    std::cout << "  261.63 Hz = MIDI " << utils::hzToMidi(261.63) << " (expected ~60.0)" << std::endl;

    // Test midiToNote
    std::cout << "\nmidiToNote tests:" << std::endl;
    std::cout << "  MIDI 69 = " << utils::midiToNote(69, true) << " (expected A4)" << std::endl;
    std::cout << "  MIDI 60 = " << utils::midiToNote(60, true) << " (expected C4)" << std::endl;
    std::cout << "  MIDI 61 = " << utils::midiToNote(61, true) << " (expected C#4)" << std::endl;
    std::cout << "  MIDI 61 = " << utils::midiToNote(61, false) << " (expected Db4)" << std::endl;

    // Test time/frame conversions
    std::cout << "\nTime/frame conversion tests:" << std::endl;
    std::cout << "  framesToTime(10, 22050, 512) = " << utils::framesToTime(10, 22050, 512) << " s" << std::endl;
    std::cout << "  timeToFrames(0.232, 22050, 512) = " << utils::timeToFrames(0.232, 22050, 512) << std::endl;

    std::cout << std::endl;
}

void testHybridCQT() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing HybridCQT" << std::endl;
    std::cout << "========================================" << std::endl;

    HybridCQT cqt;
    cqt.initialize();

    auto params = cqt.params();
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Sample rate: " << params.sampleRate << std::endl;
    std::cout << "  Hop length: " << params.hopLength << std::endl;
    std::cout << "  Bins per octave: " << params.binsPerOctave << std::endl;
    std::cout << "  Total bins: " << params.nBins << std::endl;
    std::cout << "  Fmin: " << params.fmin << " Hz" << std::endl;
    std::cout << "  Q factor: " << params.Q() << std::endl;
    std::cout << "  Octaves: " << params.nOctaves() << std::endl;

    // Get frequencies
    auto freqs = cqt.getFrequencies();
    std::cout << "\nFrequency range: " << freqs[0] << " - " << freqs.back() << " Hz" << std::endl;

    // Generate test audio
    auto audio = generateTestAudio(params.sampleRate, 2.0);
    std::cout << "\nGenerated " << audio.size() << " samples of C major chord" << std::endl;

    // Compute CQT
    auto result = cqt.compute(audio);
    int numFrames = static_cast<int>(result.size()) / params.nBins;
    std::cout << "CQT output: " << numFrames << " frames x " << params.nBins << " bins" << std::endl;

    // Test with progress callback
    std::cout << "\nTesting with progress callback:" << std::endl;
    auto progressResult = cqt.computeWithProgress(audio, [](float progress, const char* stage) {
        std::cout << "  Progress: " << std::fixed << std::setprecision(1) << (progress * 100) << "% - " << stage << std::endl;
    });

    std::cout << "\nHybridCQT test completed." << std::endl;
    std::cout << std::endl;
}

void testStandardCQT() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing StandardCQT" << std::endl;
    std::cout << "========================================" << std::endl;

    StandardCQT cqt;
    cqt.initialize();

    auto params = cqt.params();
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Sample rate: " << params.sampleRate << std::endl;
    std::cout << "  Hop length: " << params.hopLength << std::endl;
    std::cout << "  Bins per octave: " << params.binsPerOctave << std::endl;
    std::cout << "  Total bins: " << params.nBins << std::endl;
    std::cout << "  Fmin: " << params.fmin << " Hz" << std::endl;
    std::cout << "  Q factor: " << params.Q() << std::endl;

    // Generate test audio
    auto audio = generateTestAudio(params.sampleRate, 2.0);

    // Compute CQT
    auto result = cqt.compute(audio);
    int numFrames = static_cast<int>(result.size()) / params.nBins;
    std::cout << "CQT output: " << numFrames << " frames x " << params.nBins << " bins" << std::endl;

    std::cout << "\nStandardCQT test completed." << std::endl;
    std::cout << std::endl;
}

void testPseudoCQT() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing PseudoCQT" << std::endl;
    std::cout << "========================================" << std::endl;

    PseudoCQT cqt;
    cqt.initialize();

    auto params = cqt.params();
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Sample rate: " << params.sampleRate << std::endl;
    std::cout << "  Hop length: " << params.hopLength << std::endl;
    std::cout << "  Bins per octave: " << params.binsPerOctave << std::endl;
    std::cout << "  Total bins: " << params.nBins << std::endl;

    // Generate test audio
    auto audio = generateTestAudio(params.sampleRate, 2.0);

    // Compute CQT
    auto result = cqt.compute(audio);
    int numFrames = static_cast<int>(result.size()) / params.nBins;
    std::cout << "CQT output: " << numFrames << " frames x " << params.nBins << " bins" << std::endl;

    std::cout << "\nPseudoCQT test completed." << std::endl;
    std::cout << std::endl;
}

void testVQT() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing VQT" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test with gamma = 0 (should be equivalent to CQT)
    VQT vqt0(22050, 512, 36, 288, 23.12, 0.0);
    vqt0.initialize();
    std::cout << "VQT with gamma=0:" << std::endl;
    std::cout << "  Gamma: " << vqt0.gamma() << std::endl;

    // Test with gamma = 10
    VQT vqt10(22050, 512, 36, 288, 23.12, 10.0);
    vqt10.initialize();
    std::cout << "\nVQT with gamma=10:" << std::endl;
    std::cout << "  Gamma: " << vqt10.gamma() << std::endl;

    // Generate test audio
    auto audio = generateTestAudio(22050, 2.0);

    // Compute VQT
    auto result = vqt10.compute(audio);
    int numFrames = static_cast<int>(result.size()) / vqt10.params().nBins;
    std::cout << "VQT output: " << numFrames << " frames x " << vqt10.params().nBins << " bins" << std::endl;

    std::cout << "\nVQT test completed." << std::endl;
    std::cout << std::endl;
}

void exportForComparison() {
    std::cout << "========================================" << std::endl;
    std::cout << "Exporting CQT for Python comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    HybridCQT cqt;
    cqt.initialize();
    auto params = cqt.params();

    // Generate test audio
    auto audio = generateTestAudio(params.sampleRate, 2.0);

    // Compute CQT
    auto result = cqt.compute(audio);
    int numFrames = static_cast<int>(result.size()) / params.nBins;

    // Export to JSON for Python comparison
    std::ofstream file("cqt_cpp_output.json");
    file << "{" << std::endl;
    file << "  \"params\": {" << std::endl;
    file << "    \"sr\": " << params.sampleRate << "," << std::endl;
    file << "    \"hop_length\": " << params.hopLength << "," << std::endl;
    file << "    \"bins_per_octave\": " << params.binsPerOctave << "," << std::endl;
    file << "    \"n_bins\": " << params.nBins << "," << std::endl;
    file << "    \"fmin\": " << std::setprecision(10) << params.fmin << "," << std::endl;
    file << "    \"Q\": " << params.Q() << std::endl;
    file << "  }," << std::endl;

    // Export frequencies
    auto freqs = cqt.getFrequencies();
    file << "  \"freqs\": [";
    for (size_t i = 0; i < freqs.size(); i++) {
        file << std::setprecision(10) << freqs[i];
        if (i < freqs.size() - 1) file << ", ";
    }
    file << "]," << std::endl;

    // Export number of frames
    file << "  \"num_frames\": " << numFrames << "," << std::endl;
    file << "  \"n_bins\": " << params.nBins << "," << std::endl;

    // Export full CQT matrix
    file << "  \"cqt\": [";
    for (size_t i = 0; i < result.size(); i++) {
        file << std::setprecision(10) << result[i];
        if (i < result.size() - 1) file << ", ";
    }
    file << "]," << std::endl;

    // Export specific bins at frame 10
    int frame10Start = 10 * params.nBins;
    file << "  \"frame10_bins\": {" << std::endl;
    file << "    \"126\": " << std::setprecision(10) << result[frame10Start + 126] << "," << std::endl;
    file << "    \"138\": " << std::setprecision(10) << result[frame10Start + 138] << "," << std::endl;
    file << "    \"147\": " << std::setprecision(10) << result[frame10Start + 147] << std::endl;
    file << "  }," << std::endl;

    // Export audio first 100 samples
    file << "  \"audio_first_100\": [";
    for (int i = 0; i < 100; i++) {
        file << std::setprecision(10) << audio[i];
        if (i < 99) file << ", ";
    }
    file << "]" << std::endl;

    file << "}" << std::endl;
    file.close();

    std::cout << "Exported to cqt_cpp_output.json" << std::endl;
    std::cout << "Number of frames: " << numFrames << std::endl;

    // Print key values for quick check
    std::cout << "\nFrame 10 key bins:" << std::endl;
    std::cout << "  Bin 126 (C4): " << std::setprecision(6) << result[frame10Start + 126] << std::endl;
    std::cout << "  Bin 138 (E4): " << std::setprecision(6) << result[frame10Start + 138] << std::endl;
    std::cout << "  Bin 147 (G4): " << std::setprecision(6) << result[frame10Start + 147] << std::endl;
}

int main() {
    testFFT();
    testUtilityFunctions();
    testHybridCQT();
    testStandardCQT();
    testPseudoCQT();
    testVQT();
    exportForComparison();

    return 0;
}
