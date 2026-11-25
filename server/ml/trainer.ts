import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import {
  prepareAndAuditDataset,
  seedStarterDatasetIfNeeded,
  TARGET_IMAGE_SIZE,
} from './dataset.ts';
import {
  decodeImageBuffer,
  extractFrequencyFeatureVector,
  resizeRgbaBilinear,
  rgbaToNormalizedRgb,
} from './imageUtils.ts';
import { evaluateBinaryPredictions } from './metrics.ts';
import {
  buildModel,
  getModelBlueprint,
  MODEL_BLUEPRINTS,
  MODELS_DIR,
  saveModelToDisk,
} from './models.ts';
import type {
  BenchmarkSuiteResult,
  DatasetSample,
  EpochMetric,
  ModelEvaluation,
  ModelId,
  TrainingProgressState,
} from './types.ts';

export const BENCHMARKS_PATH = path.join(MODELS_DIR, 'benchmarks.json');

interface SplitTensors {
  spatial: Float32Array;
  freq: Float32Array;
  labels: Float32Array;
  count: number;
}

let progressState: TrainingProgressState = {
  isTraining: false,
  currentModelId: null,
  currentModelName: null,
  completedModels: 0,
  totalModels: MODEL_BLUEPRINTS.length,
  currentEpoch: 0,
  totalEpochs: 0,
  stage: 'Idle',
  lastError: null,
  startedAt: null,
  finishedAt: null,
};

let progressReporter: ((state: TrainingProgressState) => void) | null = null;
let activeTrainingChild: ChildProcess | null = null;

export function setProgressReporter(fn: (state: TrainingProgressState) => void): void {
  progressReporter = fn;
}

function emitProgress(): void {
  if (progressReporter) {
    progressReporter({ ...progressState });
  }
}

export function getTrainingProgress(): TrainingProgressState {
  return { ...progressState };
}

export function startBackgroundTrainingJob(selectedModelIds?: ModelId[]): TrainingProgressState {
  if (progressState.isTraining || activeTrainingChild) {
    return getTrainingProgress();
  }

  const totalModels =
    selectedModelIds && selectedModelIds.length > 0
      ? selectedModelIds.length
      : MODEL_BLUEPRINTS.length;

  progressState = {
    isTraining: true,
    currentModelId: null,
    currentModelName: null,
    completedModels: 0,
    totalModels,
    currentEpoch: 0,
    totalEpochs: 0,
    stage: 'Launching isolated training worker',
    lastError: null,
    startedAt: new Date().toISOString(),
    finishedAt: null,
  };

  const jobScript = path.resolve(process.cwd(), 'server', 'ml', 'runTrainingJob.ts');
  const args = [...process.execArgv, jobScript];
  if (selectedModelIds && selectedModelIds.length > 0) {
    args.push(JSON.stringify(selectedModelIds));
  }

  const child = spawn(process.execPath, args, {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  activeTrainingChild = child;

  let stdoutBuf = '';
  child.stdout?.on('data', (chunk: Buffer) => {
    stdoutBuf += chunk.toString('utf-8');
    const lines = stdoutBuf.split('\n');
    stdoutBuf = lines.pop() ?? '';
    for (const line of lines) {
      if (line.startsWith('PROGRESS:')) {
        try {
          const parsed = JSON.parse(line.slice('PROGRESS:'.length)) as TrainingProgressState;
          progressState = parsed;
        } catch {
          // ignore malformed partial line
        }
      }
    }
  });

  let stderrBuf = '';
  child.stderr?.on('data', (chunk: Buffer) => {
    stderrBuf += chunk.toString('utf-8');
  });

  child.on('exit', (code) => {
    activeTrainingChild = null;
    if (code !== 0) {
      progressState.isTraining = false;
      progressState.stage = 'Failed';
      progressState.lastError = stderrBuf.trim() || `Training exited with code ${code}`;
      progressState.finishedAt = new Date().toISOString();
    } else {
      progressState.isTraining = false;
      progressState.stage = 'Completed';
      progressState.finishedAt = new Date().toISOString();
    }
  });

  return getTrainingProgress();
}

function loadSamplesIntoFlatArrays(samples: DatasetSample[], augmentTrain: boolean): SplitTensors {
  const size = TARGET_IMAGE_SIZE;
  const imgStride = size * size * 3;
  const freqStride = 16;

  const multiplier = augmentTrain ? 2 : 1;
  const totalCount = samples.length * multiplier;

  const spatial = new Float32Array(totalCount * imgStride);
  const freq = new Float32Array(totalCount * freqStride);
  const labels = new Float32Array(totalCount);

  let writeIdx = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const buf = fs.readFileSync(s.filePath);
    const decoded = decodeImageBuffer(buf);
    const resized = resizeRgbaBilinear(decoded.rgba, decoded.width, decoded.height, size, size);
    const normRgb = rgbaToNormalizedRgb(resized, size, size);
    const freqVec = extractFrequencyFeatureVector(decoded.rgba, decoded.width, decoded.height);
    const labelVal = s.label === 'fake' ? 1 : 0;

    spatial.set(normRgb, writeIdx * imgStride);
    freq.set(freqVec, writeIdx * freqStride);
    labels[writeIdx] = labelVal;
    writeIdx++;

    if (augmentTrain) {
      const flipped = new Float32Array(imgStride);
      const jitter = i % 2 === 0 ? 1.04 : 0.96;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const srcOff = (y * size + (size - 1 - x)) * 3;
          const dstOff = (y * size + x) * 3;
          flipped[dstOff] = Math.max(-1, Math.min(1, normRgb[srcOff] * jitter));
          flipped[dstOff + 1] = Math.max(-1, Math.min(1, normRgb[srcOff + 1] * jitter));
          flipped[dstOff + 2] = Math.max(-1, Math.min(1, normRgb[srcOff + 2] * jitter));
        }
      }
      spatial.set(flipped, writeIdx * imgStride);
      freq.set(freqVec, writeIdx * freqStride);
      labels[writeIdx] = labelVal;
      writeIdx++;
    }
  }

  return { spatial, freq, labels, count: totalCount };
}

function buildInputsForModel(
  modelId: ModelId,
  data: SplitTensors
): tf.Tensor | tf.Tensor[] {
  const size = TARGET_IMAGE_SIZE;
  const spatialTensor = tf.tensor4d(data.spatial, [data.count, size, size, 3]);
  const blueprint = getModelBlueprint(modelId);
  if (blueprint.usesFrequencyBranch) {
    const freqTensor = tf.tensor2d(data.freq, [data.count, 16]);
    return [spatialTensor, freqTensor];
  }
  return spatialTensor;
}

function disposeInputs(inputs: tf.Tensor | tf.Tensor[]): void {
  if (Array.isArray(inputs)) {
    for (const t of inputs) t.dispose();
  } else {
    inputs.dispose();
  }
}

async function trainSingleModel(
  modelId: ModelId,
  trainData: SplitTensors,
  valData: SplitTensors,
  testData: SplitTensors,
  classWeightMap: { 0: number; 1: number },
  modelIndex: number
): Promise<ModelEvaluation> {
  const blueprint = getModelBlueprint(modelId);
  progressState.currentModelId = modelId;
  progressState.currentModelName = blueprint.displayName;
  progressState.totalEpochs = blueprint.maxEpochs;
  progressState.currentEpoch = 0;
  progressState.stage = `Training ${blueprint.displayName}`;
  emitProgress();

  const model = buildModel(modelId);
  let currentLr = blueprint.initialLearningRate;
  const optimizer = tf.train.adam(currentLr);

  model.compile({
    optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  const trainInputs = buildInputsForModel(modelId, trainData);
  const trainY = tf.tensor2d(trainData.labels, [trainData.count, 1]);

  const valInputs = buildInputsForModel(modelId, valData);
  const valY = tf.tensor2d(valData.labels, [valData.count, 1]);

  const history: EpochMetric[] = [];
  let bestValLoss = Infinity;
  let bestEpoch = 1;
  let bestWeights: tf.Tensor[] = [];
  let patienceCounter = 0;
  let lrStallCounter = 0;
  const earlyStopPatience = 4;
  let earlyStopped = false;

  const trainStart = performance.now();

  for (let epoch = 1; epoch <= blueprint.maxEpochs; epoch++) {
    progressState.currentEpoch = epoch;
    emitProgress();

    const res = await model.fit(trainInputs, trainY, {
      epochs: 1,
      batchSize: 26,
      shuffle: true,
      validationData: [valInputs, valY],
      classWeight: classWeightMap,
      verbose: 0,
    });

    const rawTrainLoss = res.history.loss?.[0];
    const rawTrainAcc = res.history.acc?.[0] ?? res.history.accuracy?.[0];
    const rawValLoss = res.history.val_loss?.[0];
    const rawValAcc = res.history.val_acc?.[0] ?? res.history.val_accuracy?.[0];

    const trainLoss = Number((typeof rawTrainLoss === 'number' ? rawTrainLoss : 0).toFixed(4));
    const trainAcc = Number((typeof rawTrainAcc === 'number' ? rawTrainAcc : 0).toFixed(4));
    const valLoss = Number((typeof rawValLoss === 'number' ? rawValLoss : 0).toFixed(4));
    const valAcc = Number((typeof rawValAcc === 'number' ? rawValAcc : 0).toFixed(4));

    history.push({
      epoch,
      trainLoss,
      trainAccuracy: trainAcc,
      valLoss,
      valAccuracy: valAcc,
      learningRate: Number(currentLr.toFixed(6)),
    });

    if (valLoss < bestValLoss - 1e-4) {
      bestValLoss = valLoss;
      bestEpoch = epoch;
      patienceCounter = 0;
      lrStallCounter = 0;
      for (const w of bestWeights) w.dispose();
      bestWeights = model.getWeights().map((w) => w.clone());
    } else {
      patienceCounter++;
      lrStallCounter++;
      if (lrStallCounter >= 2 && currentLr > 1e-4) {
        currentLr *= 0.6;
        (optimizer as unknown as { learningRate: number }).learningRate = currentLr;
        lrStallCounter = 0;
      }
      if (patienceCounter >= earlyStopPatience && epoch >= 5) {
        earlyStopped = true;
        break;
      }
    }
  }

  const trainingTimeMs = Math.round(performance.now() - trainStart);

  if (bestWeights.length > 0) {
    model.setWeights(bestWeights);
    for (const w of bestWeights) w.dispose();
  }

  const testInputs = buildInputsForModel(modelId, testData);
  const inferStart = performance.now();
  const predsTensor = model.predict(testInputs) as tf.Tensor;
  const probsTyped = await predsTensor.data();
  const totalInferMs = performance.now() - inferStart;
  const inferenceTimeMs = Number((totalInferMs / Math.max(1, testData.count)).toFixed(2));
  predsTensor.dispose();

  const yTrue = Array.from(testData.labels);
  const yProb = Array.from(probsTyped);
  const metrics = evaluateBinaryPredictions(yTrue, yProb, 0.5);

  const lastHist = history[bestEpoch - 1] ?? history[history.length - 1];
  const genGap = Math.abs((lastHist?.trainAccuracy ?? 0) - (lastHist?.valAccuracy ?? 0));
  const speedBonus = Math.max(0, 1 - inferenceTimeMs / 25);
  const selectionScore = Number(
    (
      0.45 * metrics.rocAuc +
      0.35 * metrics.f1Score +
      0.15 * metrics.accuracy -
      0.1 * genGap +
      0.05 * speedBonus
    ).toFixed(4)
  );

  const parameterCount = model.countParams();
  await saveModelToDisk(modelId, model);

  disposeInputs(trainInputs);
  trainY.dispose();
  disposeInputs(valInputs);
  valY.dispose();
  disposeInputs(testInputs);
  model.dispose();
  optimizer.dispose();

  progressState.completedModels = modelIndex + 1;
  emitProgress();

  return {
    modelId,
    displayName: blueprint.displayName,
    family: blueprint.family,
    description: blueprint.description,
    usesFrequencyBranch: blueprint.usesFrequencyBranch,
    parameterCount,
    epochsTrained: history.length,
    bestEpoch,
    earlyStopped,
    trainingTimeMs,
    inferenceTimeMs,
    accuracy: metrics.accuracy,
    precision: metrics.precision,
    recall: metrics.recall,
    f1Score: metrics.f1Score,
    specificity: metrics.specificity,
    rocAuc: metrics.rocAuc,
    selectionScore,
    confusionMatrix: metrics.confusionMatrix,
    rocCurve: metrics.rocCurve,
    classificationReport: metrics.classificationReport,
    history,
    trainedAt: new Date().toISOString(),
  };
}

export async function runFullBenchmarkTraining(
  selectedModelIds?: ModelId[]
): Promise<BenchmarkSuiteResult> {
  const modelsToTrain =
    selectedModelIds && selectedModelIds.length > 0
      ? selectedModelIds
      : MODEL_BLUEPRINTS.map((m) => m.modelId);

  progressState = {
    isTraining: true,
    currentModelId: null,
    currentModelName: null,
    completedModels: 0,
    totalModels: modelsToTrain.length,
    currentEpoch: 0,
    totalEpochs: 0,
    stage: 'Preparing & Auditing Dataset',
    lastError: null,
    startedAt: new Date().toISOString(),
    finishedAt: null,
  };
  emitProgress();

  seedStarterDatasetIfNeeded();
  const { samples, stats } = prepareAndAuditDataset();

  const trainSamples = samples.filter((s) => s.split === 'train');
  const valSamples = samples.filter((s) => s.split === 'val');
  const testSamples = samples.filter((s) => s.split === 'test');

  if (trainSamples.length < 4 || valSamples.length < 2 || testSamples.length < 2) {
    throw new Error('Dataset splits have insufficient samples for training and evaluation.');
  }

  const trainData = loadSamplesIntoFlatArrays(trainSamples, true);
  const valData = loadSamplesIntoFlatArrays(valSamples, false);
  const testData = loadSamplesIntoFlatArrays(testSamples, false);

  const classWeightMap = {
    0: stats.classWeights.real,
    1: stats.classWeights.fake,
  };

  const evaluations: ModelEvaluation[] = [];
  for (let i = 0; i < modelsToTrain.length; i++) {
    const evalResult = await trainSingleModel(
      modelsToTrain[i],
      trainData,
      valData,
      testData,
      classWeightMap,
      i
    );
    evaluations.push(evalResult);
  }

  let mergedModels = evaluations;
  if (fs.existsSync(BENCHMARKS_PATH) && modelsToTrain.length < MODEL_BLUEPRINTS.length) {
    try {
      const prev = JSON.parse(fs.readFileSync(BENCHMARKS_PATH, 'utf-8')) as BenchmarkSuiteResult;
      const map = new Map<ModelId, ModelEvaluation>();
      for (const m of prev.models) map.set(m.modelId, m);
      for (const m of evaluations) map.set(m.modelId, m);
      mergedModels = MODEL_BLUEPRINTS.map((b) => map.get(b.modelId)).filter(
        (m): m is ModelEvaluation => Boolean(m)
      );
    } catch {
      // ignore
    }
  }

  const sorted = [...mergedModels].sort((a, b) => b.selectionScore - a.selectionScore);
  const best = sorted[0];
  const runnerUp = sorted[1];
  const selectionRationale = runnerUp
    ? `${best.displayName} was automatically selected with a composite generalization score of ${best.selectionScore.toFixed(3)} (Test ROC-AUC ${(best.rocAuc * 100).toFixed(1)}%, F1 ${(best.f1Score * 100).toFixed(1)}%, ${best.inferenceTimeMs.toFixed(2)} ms/img), outperforming ${runnerUp.displayName} (${runnerUp.selectionScore.toFixed(3)}).`
    : `${best.displayName} achieved Test ROC-AUC ${(best.rocAuc * 100).toFixed(1)}% and F1 ${(best.f1Score * 100).toFixed(1)}%.`;

  const result: BenchmarkSuiteResult = {
    bestModelId: best.modelId,
    activeModelId: best.modelId,
    selectionRationale,
    models: mergedModels,
    datasetStats: stats,
    updatedAt: new Date().toISOString(),
  };

  fs.mkdirSync(MODELS_DIR, { recursive: true });
  fs.writeFileSync(BENCHMARKS_PATH, JSON.stringify(result, null, 2));

  progressState.isTraining = false;
  progressState.stage = 'Completed';
  progressState.finishedAt = new Date().toISOString();
  emitProgress();
  return result;
}

export function loadBenchmarks(): BenchmarkSuiteResult | null {
  if (!fs.existsSync(BENCHMARKS_PATH)) return null;
  try {
    return JSON.parse(fs.readFileSync(BENCHMARKS_PATH, 'utf-8')) as BenchmarkSuiteResult;
  } catch {
    return null;
  }
}

export function setActiveModelId(modelId: ModelId): BenchmarkSuiteResult {
  const existing = loadBenchmarks();
  if (!existing) {
    throw new Error('No trained models found. Please run model training first.');
  }
  const exists = existing.models.some((m) => m.modelId === modelId);
  if (!exists) {
    throw new Error(`Model ${modelId} has not been trained yet.`);
  }
  existing.activeModelId = modelId;
  fs.writeFileSync(BENCHMARKS_PATH, JSON.stringify(existing, null, 2));
  return existing;
}
