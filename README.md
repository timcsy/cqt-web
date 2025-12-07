# cqt-web

A librosa-compatible Constant-Q Transform (CQT) implementation in WebAssembly for browser-based audio analysis.

## Features

- **Multiple CQT Variants**: HybridCQT, StandardCQT, PseudoCQT, and VQT
- **librosa Compatible**: Output matches librosa's CQT implementation
- **WebAssembly**: Fast, near-native performance in the browser
- **Progress Reporting**: Optional callbacks for long audio processing
- **TypeScript Support**: Full type definitions included

## Installation

```bash
npm install cqt-web
```

## Quick Start

```javascript
import createCQTModule from 'cqt-web';

async function analyzeCQT(audioData) {
  const Module = await createCQTModule();

  // Create CQT instance (default: HybridCQT)
  const cqt = new Module.HybridCQT(22050, 512, 36, 252, 32.7);

  // Compute CQT
  const result = cqt.compute(audioData);

  // Get output shape
  const [numFrames, nBins] = cqt.getOutputShape(audioData.length);

  // Clean up
  cqt.delete();

  return { result, numFrames, nBins };
}
```

## CQT Variants

### HybridCQT (Default)

Uses early downsampling for efficiency. Recommended for most use cases.

```javascript
const cqt = new Module.HybridCQT(sampleRate, hopLength, binsPerOctave, nBins, fmin);
```

### StandardCQT

No downsampling, higher precision. Used by BTC chord recognition model.

```javascript
const cqt = new Module.StandardCQT(22050, 2048, 24, 144, 32.7);
```

### PseudoCQT

STFT-based approximation, fastest computation.

```javascript
const cqt = new Module.PseudoCQT(22050, 512, 36, 252, 32.7);
```

### VQT (Variable-Q Transform)

Variable Q-factor with gamma parameter.

```javascript
const vqt = new Module.VQT(22050, 512, 36, 252, 32.7, 20.0);
console.log(vqt.getGamma()); // 20.0
```

## Progress Reporting

For long audio files, use progress callbacks:

```javascript
const cqt = new Module.HybridCQT(22050, 512, 36, 252, 32.7);

const result = cqt.computeWithProgress(audioData, (progress, stage) => {
  console.log(`${stage}: ${(progress * 100).toFixed(1)}%`);
});
```

## Utility Functions

```javascript
const Module = await createCQTModule();

// Note/frequency conversion
Module.noteToHz('A4');        // 440.0
Module.midiToHz(69);          // 440.0
Module.hzToMidi(440.0);       // 69.0
Module.hzToMidiRounded(440);  // 69

// Time/frame conversion
Module.framesToTime(100, 22050, 512);  // Convert frames to seconds
Module.timeToFrames(2.5, 22050, 512);  // Convert seconds to frames

// Sample conversion
Module.samplesToTime(22050, 22050);    // 1.0
Module.timeToSamples(1.0, 22050);      // 22050

// MIDI to note name
Module.midiToNote(69);         // "A4"
Module.midiToNote(70, false);  // "Bb4" (use flats)
```

## FFT

Direct FFT access for custom signal processing:

```javascript
const fft = Module.FFT;

// Forward FFT
const { real, imag } = fft.forward(signal);

// Inverse FFT
const reconstructed = fft.inverse(real, imag);
```

## API Reference

### CQT Instance Methods

| Method | Description |
|--------|-------------|
| `compute(audio: Float32Array)` | Compute CQT magnitude spectrogram |
| `computeWithProgress(audio, callback)` | Compute with progress reporting |
| `getSampleRate()` | Get sample rate |
| `getHopLength()` | Get hop length |
| `getBinsPerOctave()` | Get bins per octave |
| `getNBins()` | Get total number of frequency bins |
| `getFmin()` | Get minimum frequency (Hz) |
| `getQ()` | Get Q factor |
| `getFrequencies()` | Get center frequency for each bin |
| `getFilterLengths()` | Get filter length for each bin |
| `getOutputShape(audioLength)` | Get output shape [numFrames, nBins] |
| `delete()` | Free WASM memory |

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sampleRate` | number | Audio sample rate (e.g., 22050) |
| `hopLength` | number | Hop length in samples (e.g., 512) |
| `binsPerOctave` | number | Frequency bins per octave (e.g., 36) |
| `nBins` | number | Total number of frequency bins |
| `fmin` | number | Minimum frequency in Hz (e.g., 32.7 for C1) |
| `gamma` | number | (VQT only) Variable Q parameter |

## Output Format

The `compute()` method returns a `Float32Array` containing the magnitude spectrogram in row-major order (frames × bins). To reshape:

```javascript
const result = cqt.compute(audio);
const [numFrames, nBins] = cqt.getOutputShape(audio.length);

// Access specific frame and bin
function getValue(frame, bin) {
  return result[frame * nBins + bin];
}
```

## Browser Usage

```html
<script type="module">
  import createCQTModule from './node_modules/cqt-web/dist/cqt.js';

  const Module = await createCQTModule();
  // Use Module.HybridCQT, etc.
</script>
```

## Comparison with librosa

This implementation aims for numerical compatibility with librosa:

| Feature | librosa | cqt-web |
|---------|---------|---------|
| CQT | `librosa.cqt()` | `StandardCQT` |
| Hybrid CQT | `librosa.hybrid_cqt()` | `HybridCQT` |
| Pseudo CQT | `librosa.pseudo_cqt()` | `PseudoCQT` |
| VQT | `librosa.vqt()` | `VQT` |

## Performance Tips

1. **Reuse CQT instances**: Creating a CQT object builds the filter bank, which is expensive. Reuse instances for multiple audio files with the same parameters.

2. **Choose the right variant**:
   - `HybridCQT`: Best balance of speed and accuracy
   - `PseudoCQT`: Fastest, slight accuracy trade-off
   - `StandardCQT`: Highest precision, slower

3. **Clean up**: Always call `delete()` when done to free WASM memory.

## License

MIT

## Author

timcsy
