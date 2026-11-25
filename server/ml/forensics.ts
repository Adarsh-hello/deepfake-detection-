import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { DATA_ROOT, TARGET_IMAGE_SIZE } from './dataset.ts';
import {
  compute2DFFTSpectrum,
  computeELA,
  computeSha256,
  computeTextureNoiseAnalysis,
  decodeImageBuffer,
  encodePngDataUrl,
  extractFrequencyFeatureVector,
  renderHeatmapRgba,
  resizeRgbaBilinear,
  rgbaToNormalizedRgb,
} from './imageUtils.ts';
import { getModelBlueprint, loadModelFromDisk } from './models.ts';
import { loadBenchmarks } from './trainer.ts';
import type {
  DetectionHistoryEntry,
  ForensicAnalysisResult,
  ForensicIndicator,
  ModelId,
} from './types.ts';

const HISTORY_PATH = path.join(DATA_ROOT, 'history.json');
const modelCache = new Map<ModelId, tf.LayersModel>();

export function clearModelCache(): void {
  for (const m of modelCache.values()) {
    m.dispose();
  }
  modelCache.clear();
}

async function getLoadedModel(modelId: ModelId): Promise<tf.LayersModel> {
  const cached = modelCache.get(modelId);
  if (cached) return cached;
  const loaded = await loadModelFromDisk(modelId);
  if (!loaded) {
    throw new Error(`Trained weights for ${modelId} were not found on disk. Train models first.`);
  }
  modelCache.set(modelId, loaded);
  return loaded;
}

/**
 * Computes real model-attributed spatial saliency combining input gradients (tf.grad)
 * and an 8x8 spatial occlusion sensitivity pass on the trained model.
 */
async function computeModelExplainabilityHeatmap(
  model: tf.LayersModel,
  modelId: ModelId,
  normRgb: Float32Array,
  freqVec: Float32Array,
  baseProb: number,
  outSize = 192
): Promise<string> {
  const size = TARGET_IMAGE_SIZE;
  const blueprint = getModelBlueprint(modelId);
  const grid = new Float32Array(size * size);

  // 1. Input-Gradient Attribution via tf.grad
  try {
    const spatialTensor = tf.tensor4d(normRgb, [1, size, size, 3]);
    const freqTensor = blueprint.usesFrequencyBranch
      ? tf.tensor2d(freqVec, [1, 16])
      : null;

    const gradFn = tf.grad((x: tf.Tensor) => {
      const out = freqTensor
        ? (model.apply([x, freqTensor]) as tf.Tensor)
        : (model.apply(x) as tf.Tensor);
      return out.sum();
    });

    const grads = gradFn(spatialTensor);
    const gradData = await grads.data();
    grads.dispose();
    spatialTensor.dispose();
    if (freqTensor) freqTensor.dispose();

    for (let i = 0; i < size * size; i++) {
      const g0 = Math.abs(gradData[i * 3]);
      const g1 = Math.abs(gradData[i * 3 + 1]);
      const g2 = Math.abs(gradData[i * 3 + 2]);
      grid[i] = Math.max(g0, g1, g2);
    }
  } catch {
    // Fallback gracefully to occlusion pass below if layer gradient isn't supported
  }

  // 2. 4x4 Spatial Occlusion Sensitivity Pass on the trained model
  const patchCount = 4;
  const patchDim = size / patchCount;
  const occlusionGrid = new Float32Array(size * size);

  const batchSpatial = new Float32Array(patchCount * patchCount * size * size * 3);
  const batchFreq = new Float32Array(patchCount * patchCount * 16);

  for (let py = 0; py < patchCount; py++) {
    for (let px = 0; px < patchCount; px++) {
      const bIdx = py * patchCount + px;
      const offset = bIdx * size * size * 3;
      batchSpatial.set(normRgb, offset);
      if (blueprint.usesFrequencyBranch) {
        batchFreq.set(freqVec, bIdx * 16);
      }
      // Neutralize occluded patch to zero mean
      for (let y = py * patchDim; y < (py + 1) * patchDim; y++) {
        for (let x = px * patchDim; x < (px + 1) * patchDim; x++) {
          const pOff = offset + (y * size + x) * 3;
          batchSpatial[pOff] = 0;
          batchSpatial[pOff + 1] = 0;
          batchSpatial[pOff + 2] = 0;
        }
      }
    }
  }

  const bSpatialT = tf.tensor4d(batchSpatial, [patchCount * patchCount, size, size, 3]);
  const bFreqT = blueprint.usesFrequencyBranch
    ? tf.tensor2d(batchFreq, [patchCount * patchCount, 16])
    : null;
  const occPreds = bFreqT
    ? (model.predict([bSpatialT, bFreqT]) as tf.Tensor)
    : (model.predict(bSpatialT) as tf.Tensor);
  const occProbs = await occPreds.data();
  occPreds.dispose();
  bSpatialT.dispose();
  if (bFreqT) bFreqT.dispose();

  for (let py = 0; py < patchCount; py++) {
    for (let px = 0; px < patchCount; px++) {
      const delta = Math.abs(baseProb - occProbs[py * patchCount + px]);
      for (let y = py * patchDim; y < (py + 1) * patchDim; y++) {
        for (let x = px * patchDim; x < (px + 1) * patchDim; x++) {
          occlusionGrid[y * size + x] = delta;
        }
      }
    }
  }

  // Normalize & blend input-gradient saliency (65%) + occlusion sensitivity (35%)
  let maxGrad = 1e-8;
  let maxOcc = 1e-8;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > maxGrad) maxGrad = grid[i];
    if (occlusionGrid[i] > maxOcc) maxOcc = occlusionGrid[i];
  }

  const combinedSmall = new Float32Array(size * size);
  for (let i = 0; i < combinedSmall.length; i++) {
    combinedSmall[i] = 0.65 * (grid[i] / maxGrad) + 0.35 * (occlusionGrid[i] / maxOcc);
  }

  // Smooth bilinear upsample to 192x192 for crisp visual overlay
  const upsampled = new Float32Array(outSize * outSize);
  const scale = size / outSize;
  for (let y = 0; y < outSize; y++) {
    const sy = Math.min(size - 1, Math.max(0, (y + 0.5) * scale - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(size - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < outSize; x++) {
      const sx = Math.min(size - 1, Math.max(0, (x + 0.5) * scale - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(size - 1, x0 + 1);
      const fx = sx - x0;
      const top =
        combinedSmall[y0 * size + x0] * (1 - fx) + combinedSmall[y0 * size + x1] * fx;
      const bot =
        combinedSmall[y1 * size + x0] * (1 - fx) + combinedSmall[y1 * size + x1] * fx;
      upsampled[y * outSize + x] = top * (1 - fy) + bot * fy;
    }
  }

  const heatmapRgba = renderHeatmapRgba(upsampled, outSize, outSize, 220);
  return encodePngDataUrl(outSize, outSize, heatmapRgba);
}

export async function analyzeImageBuffer(
  buffer: Buffer,
  filename: string,
  requestedModelId?: ModelId
): Promise<ForensicAnalysisResult> {
  const benchmarks = loadBenchmarks();
  if (!benchmarks) {
    throw new Error('Models are still training or not yet initialized. Please try again shortly.');
  }

  const modelId: ModelId = requestedModelId ?? benchmarks.activeModelId;
  const blueprint = getModelBlueprint(modelId);
  const model = await getLoadedModel(modelId);

  const decoded = decodeImageBuffer(buffer);
  const sha256 = computeSha256(buffer);

  // Prepare display preview (max 384px) & model input (24x24)
  const previewSize = 384;
  const previewRgba = resizeRgbaBilinear(
    decoded.rgba,
    decoded.width,
    decoded.height,
    previewSize,
    previewSize
  );
  const originalDataUrl = encodePngDataUrl(previewSize, previewSize, previewRgba);

  const modelInputRgba = resizeRgbaBilinear(
    decoded.rgba,
    decoded.width,
    decoded.height,
    TARGET_IMAGE_SIZE,
    TARGET_IMAGE_SIZE
  );
  const normRgb = rgbaToNormalizedRgb(modelInputRgba, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);
  const freqVec = extractFrequencyFeatureVector(
    decoded.rgba,
    decoded.width,
    decoded.height
  );

  // Real model inference
  const inferStart = performance.now();
  const spatialT = tf.tensor4d(normRgb, [1, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3]);
  const freqT = blueprint.usesFrequencyBranch ? tf.tensor2d(freqVec, [1, 16]) : null;
  const predTensor = freqT
    ? (model.predict([spatialT, freqT]) as tf.Tensor)
    : (model.predict(spatialT) as tf.Tensor);
  const probData = await predTensor.data();
  const inferenceTimeMs = Number((performance.now() - inferStart).toFixed(2));
  predTensor.dispose();
  spatialT.dispose();
  if (freqT) freqT.dispose();

  const fakeProbability = Number(Math.max(0.0001, Math.min(0.9999, probData[0])).toFixed(4));
  const realProbability = Number((1 - fakeProbability).toFixed(4));
  const prediction: 'REAL' | 'FAKE' = fakeProbability >= 0.5 ? 'FAKE' : 'REAL';
  const confidence = Number(
    (Math.max(fakeProbability, realProbability) * 100).toFixed(2)
  );

  // Supporting visual & forensic analyses
  const overlaySize = 192;
  const gradCamHeatmapDataUrl = await computeModelExplainabilityHeatmap(
    model,
    modelId,
    normRgb,
    freqVec,
    fakeProbability,
    overlaySize
  );

  const ela = computeELA(decoded.rgba, decoded.width, decoded.height, overlaySize);
  const elaHeatmapDataUrl = encodePngDataUrl(overlaySize, overlaySize, ela.heatmapRgba);

  const texture = computeTextureNoiseAnalysis(
    decoded.rgba,
    decoded.width,
    decoded.height,
    overlaySize
  );
  const textureNoiseDataUrl = encodePngDataUrl(
    overlaySize,
    overlaySize,
    texture.noiseMapRgba
  );

  const fft = compute2DFFTSpectrum(decoded.rgba, decoded.width, decoded.height, 64);
  const fftSpectrumDataUrl = encodePngDataUrl(
    fft.spectrumSize,
    fft.spectrumSize,
    fft.spectrumRgba
  );

  const indicators: ForensicIndicator[] = [
    {
      id: 'fft_grid',
      name: 'Periodic Upsampling Grid Index',
      category: 'frequency',
      value: Number((fft.periodicGridScore * 100).toFixed(1)),
      unit: '%',
      baselineRange: '10% – 38%',
      assessment:
        fft.periodicGridScore > 0.55
          ? 'anomalous'
          : fft.periodicGridScore > 0.4
            ? 'elevated'
            : 'nominal',
      summary:
        'Measures off-diagonal vs cardinal-axis high-frequency energy peaks associated with deconvolution or upsampling kernels.',
    },
    {
      id: 'fft_hf',
      name: 'High-Frequency Spectral Ratio',
      category: 'frequency',
      value: Number((fft.highFreqRatio * 100).toFixed(1)),
      unit: '%',
      baselineRange: '18% – 42%',
      assessment:
        fft.highFreqRatio < 0.14 || fft.highFreqRatio > 0.52
          ? 'elevated'
          : 'nominal',
      summary:
        'Proportion of 2D-DFT spectral magnitude in outer Nyquist frequency bands; excessively low values can indicate synthetic smoothing.',
    },
    {
      id: 'ela_residual',
      name: 'Error Level Analysis (ELA) Variance',
      category: 'compression',
      value: ela.regionalVariance,
      unit: 'MSE',
      baselineRange: '0.2 – 3.5 MSE',
      assessment:
        ela.regionalVariance > 6.5
          ? 'anomalous'
          : ela.regionalVariance > 3.5
            ? 'elevated'
            : 'nominal',
      summary:
        'Block-wise variance of JPEG Q=90 recompression residuals, highlighting regions compressed at mismatched quality levels.',
    },
    {
      id: 'noise_inconsistency',
      name: 'Local Noise Residual Inconsistency',
      category: 'texture',
      value: Number((texture.localNoiseInconsistency * 100).toFixed(1)),
      unit: '% CV',
      baselineRange: '15% – 55%',
      assessment:
        texture.localNoiseInconsistency > 0.72
          ? 'anomalous'
          : texture.localNoiseInconsistency > 0.55
            ? 'elevated'
            : 'nominal',
      summary:
        'Coefficient of variation across 8x8 Laplacian high-pass noise blocks; spliced or selectively airbrushed regions alter local noise uniformity.',
    },
    {
      id: 'center_edge_sharpness',
      name: 'Facial-Center vs Periphery Sharpness',
      category: 'structural',
      value: texture.centerVsEdgeSharpnessRatio,
      unit: 'ratio',
      baselineRange: '0.75 – 1.85',
      assessment:
        texture.centerVsEdgeSharpnessRatio < 0.55 ||
        texture.centerVsEdgeSharpnessRatio > 2.4
          ? 'elevated'
          : 'nominal',
      summary:
        'Compares Laplacian detail density in the central subject zone against peripheral background regions.',
    },
    {
      id: 'chroma_corr',
      name: 'R-G Chrominance Cross-Correlation',
      category: 'texture',
      value: texture.chromaCrossCorrelation,
      unit: 'r',
      baselineRange: '0.65 – 0.96',
      assessment:
        texture.chromaCrossCorrelation > 0.985 || texture.chromaCrossCorrelation < 0.45
          ? 'elevated'
          : 'nominal',
      summary:
        'Pearson correlation across red and green color channels; unnatural skin smoothing or color transfer can shift channel coupling.',
    },
  ];

  const result: ForensicAnalysisResult = {
    id: `det_${Date.now()}_${sha256.slice(0, 8)}`,
    filename,
    imageWidth: decoded.width,
    imageHeight: decoded.height,
    sha256,
    analyzedAt: new Date().toISOString(),
    modelId,
    modelDisplayName: blueprint.displayName,
    prediction,
    confidence,
    fakeProbability,
    realProbability,
    inferenceTimeMs,
    originalDataUrl,
    gradCamHeatmapDataUrl,
    elaHeatmapDataUrl,
    textureNoiseDataUrl,
    fftSpectrumDataUrl,
    radialSpectrum: fft.radialBins,
    indicators,
    disclaimer:
      'Frequency-domain spectra, ELA residuals, and texture maps are supporting forensic indicators only. They highlight statistical anomalies for human review and do not independently prove authenticity or manipulation.',
  };

  // Persist compact thumbnail + full report to history
  const thumbRgba = resizeRgbaBilinear(decoded.rgba, decoded.width, decoded.height, 96, 96);
  const thumbnailDataUrl = encodePngDataUrl(96, 96, thumbRgba);

  appendHistoryEntry({
    id: result.id,
    filename: result.filename,
    analyzedAt: result.analyzedAt,
    modelId: result.modelId,
    modelDisplayName: result.modelDisplayName,
    prediction: result.prediction,
    confidence: result.confidence,
    fakeProbability: result.fakeProbability,
    inferenceTimeMs: result.inferenceTimeMs,
    thumbnailDataUrl,
    sha256: result.sha256,
    report: result,
  });

  return result;
}

export function loadHistory(): DetectionHistoryEntry[] {
  if (!fs.existsSync(HISTORY_PATH)) return [];
  try {
    return JSON.parse(fs.readFileSync(HISTORY_PATH, 'utf-8')) as DetectionHistoryEntry[];
  } catch {
    return [];
  }
}

function appendHistoryEntry(entry: DetectionHistoryEntry): void {
  fs.mkdirSync(DATA_ROOT, { recursive: true });
  const list = loadHistory();
  list.unshift(entry);
  // Keep latest 50 detections on disk
  const trimmed = list.slice(0, 50);
  fs.writeFileSync(HISTORY_PATH, JSON.stringify(trimmed, null, 2));
}

export function deleteHistoryEntry(id: string): DetectionHistoryEntry[] {
  const list = loadHistory().filter((item) => item.id !== id);
  fs.writeFileSync(HISTORY_PATH, JSON.stringify(list, null, 2));
  return list;
}
