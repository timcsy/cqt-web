#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "cqt/hybrid_cqt.hpp"
#include "cqt/standard_cqt.hpp"
#include "cqt/pseudo_cqt.hpp"
#include "cqt/vqt.hpp"
#include "core/fft.hpp"
#include "utils/convert.hpp"
#include <vector>

using namespace emscripten;
using namespace cqt;

// Helper to convert JS Float32Array to std::vector<float>
std::vector<float> jsArrayToVector(val audioArray) {
    unsigned int length = audioArray["length"].as<unsigned int>();
    std::vector<float> audio(length);
    for (unsigned int i = 0; i < length; i++) {
        audio[i] = audioArray[i].as<float>();
    }
    return audio;
}

// Helper to convert std::vector<float> to JS Float32Array
val vectorToJsArray(const std::vector<float>& vec) {
    val jsArray = val::global("Float32Array").new_(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        jsArray.set(i, vec[i]);
    }
    return jsArray;
}

// Helper to convert std::vector<double> to JS Float64Array
val vectorToJsArrayDouble(const std::vector<double>& vec) {
    val jsArray = val::global("Float64Array").new_(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        jsArray.set(i, vec[i]);
    }
    return jsArray;
}

// Helper to convert std::vector<int> to JS Int32Array
val vectorToJsArrayInt(const std::vector<int>& vec) {
    val jsArray = val::global("Int32Array").new_(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        jsArray.set(i, vec[i]);
    }
    return jsArray;
}

//------------------------------------------------------------------------------
// HybridCQT Wrapper
//------------------------------------------------------------------------------
class HybridCQTWrapper {
public:
    HybridCQTWrapper() {
        cqt_.initialize();
    }

    HybridCQTWrapper(int sampleRate, int hopLength, int binsPerOctave, int nBins, float fmin)
        : cqt_(sampleRate, hopLength, binsPerOctave, nBins, fmin) {
        cqt_.initialize();
    }

    val compute(val audioArray) {
        auto audio = jsArrayToVector(audioArray);
        auto result = cqt_.compute(audio.data(), audio.size());
        return vectorToJsArray(result);
    }

    val computeWithProgress(val audioArray, val callback) {
        auto audio = jsArrayToVector(audioArray);

        ProgressCallback cppCallback = nullptr;
        if (!callback.isNull() && !callback.isUndefined()) {
            cppCallback = [callback](float progress, const char* stage) {
                callback(val(progress), val(std::string(stage)));
            };
        }

        auto result = cqt_.computeWithProgress(audio.data(), audio.size(), cppCallback);
        return vectorToJsArray(result);
    }

    int getSampleRate() const { return cqt_.params().sampleRate; }
    int getHopLength() const { return cqt_.params().hopLength; }
    int getBinsPerOctave() const { return cqt_.params().binsPerOctave; }
    int getNBins() const { return cqt_.params().nBins; }
    float getFmin() const { return static_cast<float>(cqt_.params().fmin); }
    float getQ() const { return static_cast<float>(cqt_.params().Q()); }

    val getFrequencies() {
        return vectorToJsArrayDouble(cqt_.getFrequencies());
    }

    val getFilterLengths() {
        return vectorToJsArrayInt(cqt_.getFilterLengths());
    }

    val getOutputShape(int audioLength) {
        int numFrames = cqt_.getNumFrames(audioLength);
        val shape = val::array();
        shape.call<void>("push", numFrames);
        shape.call<void>("push", cqt_.params().nBins);
        return shape;
    }

private:
    HybridCQT cqt_;
};

//------------------------------------------------------------------------------
// StandardCQT Wrapper
//------------------------------------------------------------------------------
class StandardCQTWrapper {
public:
    StandardCQTWrapper() {
        cqt_.initialize();
    }

    StandardCQTWrapper(int sampleRate, int hopLength, int binsPerOctave, int nBins, float fmin)
        : cqt_(sampleRate, hopLength, binsPerOctave, nBins, fmin) {
        cqt_.initialize();
    }

    val compute(val audioArray) {
        auto audio = jsArrayToVector(audioArray);
        auto result = cqt_.compute(audio.data(), audio.size());
        return vectorToJsArray(result);
    }

    val computeWithProgress(val audioArray, val callback) {
        auto audio = jsArrayToVector(audioArray);

        ProgressCallback cppCallback = nullptr;
        if (!callback.isNull() && !callback.isUndefined()) {
            cppCallback = [callback](float progress, const char* stage) {
                callback(val(progress), val(std::string(stage)));
            };
        }

        auto result = cqt_.computeWithProgress(audio.data(), audio.size(), cppCallback);
        return vectorToJsArray(result);
    }

    int getSampleRate() const { return cqt_.params().sampleRate; }
    int getHopLength() const { return cqt_.params().hopLength; }
    int getBinsPerOctave() const { return cqt_.params().binsPerOctave; }
    int getNBins() const { return cqt_.params().nBins; }
    float getFmin() const { return static_cast<float>(cqt_.params().fmin); }
    float getQ() const { return static_cast<float>(cqt_.params().Q()); }

    val getFrequencies() {
        return vectorToJsArrayDouble(cqt_.getFrequencies());
    }

    val getFilterLengths() {
        return vectorToJsArrayInt(cqt_.getFilterLengths());
    }

    val getOutputShape(int audioLength) {
        int numFrames = cqt_.getNumFrames(audioLength);
        val shape = val::array();
        shape.call<void>("push", numFrames);
        shape.call<void>("push", cqt_.params().nBins);
        return shape;
    }

private:
    StandardCQT cqt_;
};

//------------------------------------------------------------------------------
// PseudoCQT Wrapper
//------------------------------------------------------------------------------
class PseudoCQTWrapper {
public:
    PseudoCQTWrapper() {
        cqt_.initialize();
    }

    PseudoCQTWrapper(int sampleRate, int hopLength, int binsPerOctave, int nBins, float fmin)
        : cqt_(sampleRate, hopLength, binsPerOctave, nBins, fmin) {
        cqt_.initialize();
    }

    val compute(val audioArray) {
        auto audio = jsArrayToVector(audioArray);
        auto result = cqt_.compute(audio.data(), audio.size());
        return vectorToJsArray(result);
    }

    val computeWithProgress(val audioArray, val callback) {
        auto audio = jsArrayToVector(audioArray);

        ProgressCallback cppCallback = nullptr;
        if (!callback.isNull() && !callback.isUndefined()) {
            cppCallback = [callback](float progress, const char* stage) {
                callback(val(progress), val(std::string(stage)));
            };
        }

        auto result = cqt_.computeWithProgress(audio.data(), audio.size(), cppCallback);
        return vectorToJsArray(result);
    }

    int getSampleRate() const { return cqt_.params().sampleRate; }
    int getHopLength() const { return cqt_.params().hopLength; }
    int getBinsPerOctave() const { return cqt_.params().binsPerOctave; }
    int getNBins() const { return cqt_.params().nBins; }
    float getFmin() const { return static_cast<float>(cqt_.params().fmin); }
    float getQ() const { return static_cast<float>(cqt_.params().Q()); }

    val getFrequencies() {
        return vectorToJsArrayDouble(cqt_.getFrequencies());
    }

    val getFilterLengths() {
        return vectorToJsArrayInt(cqt_.getFilterLengths());
    }

    val getOutputShape(int audioLength) {
        int numFrames = cqt_.getNumFrames(audioLength);
        val shape = val::array();
        shape.call<void>("push", numFrames);
        shape.call<void>("push", cqt_.params().nBins);
        return shape;
    }

private:
    PseudoCQT cqt_;
};

//------------------------------------------------------------------------------
// VQT Wrapper
//------------------------------------------------------------------------------
class VQTWrapper {
public:
    VQTWrapper() {
        vqt_.initialize();
    }

    VQTWrapper(int sampleRate, int hopLength, int binsPerOctave, int nBins, float fmin, float gamma = 0.0f)
        : vqt_(sampleRate, hopLength, binsPerOctave, nBins, fmin, gamma) {
        vqt_.initialize();
    }

    val compute(val audioArray) {
        auto audio = jsArrayToVector(audioArray);
        auto result = vqt_.compute(audio.data(), audio.size());
        return vectorToJsArray(result);
    }

    val computeWithProgress(val audioArray, val callback) {
        auto audio = jsArrayToVector(audioArray);

        ProgressCallback cppCallback = nullptr;
        if (!callback.isNull() && !callback.isUndefined()) {
            cppCallback = [callback](float progress, const char* stage) {
                callback(val(progress), val(std::string(stage)));
            };
        }

        auto result = vqt_.computeWithProgress(audio.data(), audio.size(), cppCallback);
        return vectorToJsArray(result);
    }

    int getSampleRate() const { return vqt_.params().sampleRate; }
    int getHopLength() const { return vqt_.params().hopLength; }
    int getBinsPerOctave() const { return vqt_.params().binsPerOctave; }
    int getNBins() const { return vqt_.params().nBins; }
    float getFmin() const { return static_cast<float>(vqt_.params().fmin); }
    float getQ() const { return static_cast<float>(vqt_.params().Q()); }
    float getGamma() const { return static_cast<float>(vqt_.gamma()); }

    val getFrequencies() {
        return vectorToJsArrayDouble(vqt_.getFrequencies());
    }

    val getFilterLengths() {
        return vectorToJsArrayInt(vqt_.getFilterLengths());
    }

    val getOutputShape(int audioLength) {
        int numFrames = vqt_.getNumFrames(audioLength);
        val shape = val::array();
        shape.call<void>("push", numFrames);
        shape.call<void>("push", vqt_.params().nBins);
        return shape;
    }

private:
    VQT vqt_;
};

//------------------------------------------------------------------------------
// FFT Wrapper
//------------------------------------------------------------------------------
class FFTWrapper {
public:
    static val forward(val realArray) {
        unsigned int length = realArray["length"].as<unsigned int>();
        std::vector<double> real(length);
        for (unsigned int i = 0; i < length; i++) {
            real[i] = realArray[i].as<double>();
        }

        auto result = FFT::forwardReal(real);

        // Return object with real and imag arrays
        size_t n = result.size() / 2;
        val realOut = val::global("Float64Array").new_(n);
        val imagOut = val::global("Float64Array").new_(n);

        for (size_t i = 0; i < n; i++) {
            realOut.set(i, result[i * 2]);
            imagOut.set(i, result[i * 2 + 1]);
        }

        val resultObj = val::object();
        resultObj.set("real", realOut);
        resultObj.set("imag", imagOut);
        return resultObj;
    }

    static val inverse(val realArray, val imagArray) {
        unsigned int length = realArray["length"].as<unsigned int>();
        std::vector<double> real(length);
        std::vector<double> imag(length);

        for (unsigned int i = 0; i < length; i++) {
            real[i] = realArray[i].as<double>();
            imag[i] = imagArray[i].as<double>();
        }

        auto result = FFT::inverseComplex(real, imag);

        val jsArray = val::global("Float64Array").new_(result.size());
        for (size_t i = 0; i < result.size(); i++) {
            jsArray.set(i, result[i]);
        }
        return jsArray;
    }
};

//------------------------------------------------------------------------------
// Utility function wrappers
//------------------------------------------------------------------------------
double noteToHzWrapper(const std::string& note) {
    return utils::noteToHz(note);
}

double midiToHzWrapper(int midi) {
    return utils::midiToHz(midi);
}

double hzToMidiWrapper(double hz) {
    return utils::hzToMidi(hz);
}

int hzToMidiRoundedWrapper(double hz) {
    return utils::hzToMidiRounded(hz);
}

double framesToTimeWrapper(int frames, int sr, int hopLength) {
    return utils::framesToTime(frames, sr, hopLength);
}

int timeToFramesWrapper(double time, int sr, int hopLength) {
    return utils::timeToFrames(time, sr, hopLength);
}

double samplesToTimeWrapper(int samples, int sr) {
    return utils::samplesToTime(samples, sr);
}

int timeToSamplesWrapper(double time, int sr) {
    return utils::timeToSamples(time, sr);
}

std::string midiToNoteWrapper(int midi, bool useSharps) {
    return utils::midiToNote(midi, useSharps);
}

//------------------------------------------------------------------------------
// Emscripten Bindings
//------------------------------------------------------------------------------
EMSCRIPTEN_BINDINGS(cqt_module) {
    // HybridCQT (default, for CNN-LSTM model)
    class_<HybridCQTWrapper>("HybridCQT")
        .constructor<>()
        .constructor<int, int, int, int, float>()
        .function("compute", &HybridCQTWrapper::compute)
        .function("computeWithProgress", &HybridCQTWrapper::computeWithProgress)
        .function("getSampleRate", &HybridCQTWrapper::getSampleRate)
        .function("getHopLength", &HybridCQTWrapper::getHopLength)
        .function("getBinsPerOctave", &HybridCQTWrapper::getBinsPerOctave)
        .function("getNBins", &HybridCQTWrapper::getNBins)
        .function("getFmin", &HybridCQTWrapper::getFmin)
        .function("getQ", &HybridCQTWrapper::getQ)
        .function("getFrequencies", &HybridCQTWrapper::getFrequencies)
        .function("getFilterLengths", &HybridCQTWrapper::getFilterLengths)
        .function("getOutputShape", &HybridCQTWrapper::getOutputShape);

    // StandardCQT (for BTC model)
    class_<StandardCQTWrapper>("StandardCQT")
        .constructor<>()
        .constructor<int, int, int, int, float>()
        .function("compute", &StandardCQTWrapper::compute)
        .function("computeWithProgress", &StandardCQTWrapper::computeWithProgress)
        .function("getSampleRate", &StandardCQTWrapper::getSampleRate)
        .function("getHopLength", &StandardCQTWrapper::getHopLength)
        .function("getBinsPerOctave", &StandardCQTWrapper::getBinsPerOctave)
        .function("getNBins", &StandardCQTWrapper::getNBins)
        .function("getFmin", &StandardCQTWrapper::getFmin)
        .function("getQ", &StandardCQTWrapper::getQ)
        .function("getFrequencies", &StandardCQTWrapper::getFrequencies)
        .function("getFilterLengths", &StandardCQTWrapper::getFilterLengths)
        .function("getOutputShape", &StandardCQTWrapper::getOutputShape);

    // PseudoCQT (fast approximation)
    class_<PseudoCQTWrapper>("PseudoCQT")
        .constructor<>()
        .constructor<int, int, int, int, float>()
        .function("compute", &PseudoCQTWrapper::compute)
        .function("computeWithProgress", &PseudoCQTWrapper::computeWithProgress)
        .function("getSampleRate", &PseudoCQTWrapper::getSampleRate)
        .function("getHopLength", &PseudoCQTWrapper::getHopLength)
        .function("getBinsPerOctave", &PseudoCQTWrapper::getBinsPerOctave)
        .function("getNBins", &PseudoCQTWrapper::getNBins)
        .function("getFmin", &PseudoCQTWrapper::getFmin)
        .function("getQ", &PseudoCQTWrapper::getQ)
        .function("getFrequencies", &PseudoCQTWrapper::getFrequencies)
        .function("getFilterLengths", &PseudoCQTWrapper::getFilterLengths)
        .function("getOutputShape", &PseudoCQTWrapper::getOutputShape);

    // VQT (Variable-Q Transform)
    class_<VQTWrapper>("VQT")
        .constructor<>()
        .constructor<int, int, int, int, float, float>()
        .function("compute", &VQTWrapper::compute)
        .function("computeWithProgress", &VQTWrapper::computeWithProgress)
        .function("getSampleRate", &VQTWrapper::getSampleRate)
        .function("getHopLength", &VQTWrapper::getHopLength)
        .function("getBinsPerOctave", &VQTWrapper::getBinsPerOctave)
        .function("getNBins", &VQTWrapper::getNBins)
        .function("getFmin", &VQTWrapper::getFmin)
        .function("getQ", &VQTWrapper::getQ)
        .function("getGamma", &VQTWrapper::getGamma)
        .function("getFrequencies", &VQTWrapper::getFrequencies)
        .function("getFilterLengths", &VQTWrapper::getFilterLengths)
        .function("getOutputShape", &VQTWrapper::getOutputShape);

    // Backwards compatibility: CQT alias for HybridCQT
    class_<HybridCQTWrapper>("CQT")
        .constructor<>()
        .constructor<int, int, int, int, float>()
        .function("compute", &HybridCQTWrapper::compute)
        .function("computeWithProgress", &HybridCQTWrapper::computeWithProgress)
        .function("getSampleRate", &HybridCQTWrapper::getSampleRate)
        .function("getHopLength", &HybridCQTWrapper::getHopLength)
        .function("getBinsPerOctave", &HybridCQTWrapper::getBinsPerOctave)
        .function("getNBins", &HybridCQTWrapper::getNBins)
        .function("getFmin", &HybridCQTWrapper::getFmin)
        .function("getQ", &HybridCQTWrapper::getQ)
        .function("getFrequencies", &HybridCQTWrapper::getFrequencies)
        .function("getFilterLengths", &HybridCQTWrapper::getFilterLengths)
        .function("getOutputShape", &HybridCQTWrapper::getOutputShape);

    // FFT utilities
    class_<FFTWrapper>("FFT")
        .class_function("forward", &FFTWrapper::forward)
        .class_function("inverse", &FFTWrapper::inverse);

    // Utility functions
    function("noteToHz", &noteToHzWrapper);
    function("midiToHz", &midiToHzWrapper);
    function("hzToMidi", &hzToMidiWrapper);
    function("hzToMidiRounded", &hzToMidiRoundedWrapper);
    function("framesToTime", &framesToTimeWrapper);
    function("timeToFrames", &timeToFramesWrapper);
    function("samplesToTime", &samplesToTimeWrapper);
    function("timeToSamples", &timeToSamplesWrapper);
    function("midiToNote", &midiToNoteWrapper);
}
