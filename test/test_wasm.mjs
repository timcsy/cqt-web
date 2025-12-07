#!/usr/bin/env node
/**
 * Test the WASM CQT module
 */

import createCQTModule from '../dist/cqt.js';

async function main() {
  console.log('=' .repeat(70));
  console.log('Testing WASM CQT Module');
  console.log('=' .repeat(70));

  // Load the WASM module
  console.log('\nLoading WASM module...');
  const Module = await createCQTModule();
  console.log('WASM module loaded!');

  // Create CQT instance with default parameters
  console.log('\nCreating CQT instance...');
  const cqt = new Module.CQT();
  console.log('CQT instance created!');

  // Print parameters
  console.log('\nParameters:');
  console.log(`  Sample rate: ${cqt.getSampleRate()}`);
  console.log(`  Hop length: ${cqt.getHopLength()}`);
  console.log(`  Bins per octave: ${cqt.getBinsPerOctave()}`);
  console.log(`  Number of bins: ${cqt.getNBins()}`);
  console.log(`  Fmin: ${cqt.getFmin().toFixed(4)} Hz`);
  console.log(`  Q factor: ${cqt.getQ().toFixed(6)}`);

  // Generate test audio (C major chord)
  const sr = cqt.getSampleRate();
  const duration = 2.0;
  const numSamples = Math.floor(sr * duration);
  const audio = new Float32Array(numSamples);

  const c4 = 261.63, e4 = 329.63, g4 = 392.00;
  for (let i = 0; i < numSamples; i++) {
    const t = i / sr;
    audio[i] = 0.3 * Math.sin(2 * Math.PI * c4 * t) +
               0.3 * Math.sin(2 * Math.PI * e4 * t) +
               0.3 * Math.sin(2 * Math.PI * g4 * t);
  }

  console.log(`\nGenerated audio: ${numSamples} samples, ${duration}s`);

  // Get expected output shape
  const shape = cqt.getOutputShape(numSamples);
  console.log(`Expected output shape: [${shape[0]}, ${shape[1]}]`);

  // Compute CQT
  console.log('\nComputing CQT...');
  const startTime = Date.now();
  const result = cqt.compute(audio);
  const endTime = Date.now();
  console.log(`CQT computed in ${endTime - startTime}ms`);

  // Print some results
  const numFrames = shape[0];
  const nBins = shape[1];
  console.log(`\nResult length: ${result.length}`);
  console.log(`Expected length: ${numFrames * nBins}`);

  // Get frequencies
  const freqs = cqt.getFrequencies();
  console.log(`\nFrequency range: ${freqs[0].toFixed(2)} - ${freqs[nBins-1].toFixed(2)} Hz`);

  // Print frame 10
  console.log('\nFrame 10 (first 10 bins):');
  for (let i = 0; i < 10; i++) {
    const idx = 10 * nBins + i;
    console.log(`  Bin ${i} (${freqs[i].toFixed(2)} Hz): ${result[idx].toFixed(8)}`);
  }

  // Find peaks in frame 10
  console.log('\nFinding peaks in frame 10...');
  const frame10 = [];
  for (let i = 0; i < nBins; i++) {
    frame10.push({ bin: i, value: result[10 * nBins + i] });
  }
  frame10.sort((a, b) => b.value - a.value);

  console.log('Top 5 peaks:');
  for (let i = 0; i < 5; i++) {
    const peak = frame10[i];
    console.log(`  Bin ${peak.bin} (${freqs[peak.bin].toFixed(2)} Hz): ${peak.value.toFixed(6)}`);
  }

  // Clean up
  cqt.delete();

  console.log('\n' + '=' .repeat(70));
  console.log('WASM test completed successfully!');
  console.log('=' .repeat(70));
}

main().catch(console.error);
