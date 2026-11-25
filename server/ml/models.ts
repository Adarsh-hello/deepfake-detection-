import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { DATA_ROOT, TARGET_IMAGE_SIZE } from './dataset.ts';
import type { ModelId } from './types.ts';

export const MODELS_DIR = path.join(DATA_ROOT, 'models');

export interface ModelBlueprint {
  modelId: ModelId;
  displayName: string;
  family: string;
  description: string;
  usesFrequencyBranch: boolean;
  initialLearningRate: number;
  maxEpochs: number;
}

export const MODEL_BLUEPRINTS: ModelBlueprint[] = [
  {
    modelId: 'cnn_baseline',
    displayName: 'Spatial CNN Baseline',
    family: 'Convolutional Neural Network',
    description:
      'Three-stage spatial Conv2D stack with BatchNorm, MaxPooling, and Global Average Pooling for baseline local edge and texture discrimination.',
    usesFrequencyBranch: false,
    initialLearningRate: 0.004,
    maxEpochs: 8,
  },
  {
    modelId: 'resnet_lite',
    displayName: 'ResNet-Lite Residual Net',
    family: 'Residual Deep Architecture',
    description:
      'Skip-connected residual Conv2D blocks with L2 weight decay enabling deeper hierarchical artifact propagation without gradient degradation.',
    usesFrequencyBranch: false,
    initialLearningRate: 0.0035,
    maxEpochs: 8,
  },
  {
    modelId: 'xception_mobilenet',
    displayName: 'Xception / MobileNet-DS',
    family: 'Depthwise-Separable ConvNet',
    description:
      'Factorized depthwise-separable convolutions decoupling spatial correlation from cross-channel mixing for high efficiency and generalization.',
    usesFrequencyBranch: false,
    initialLearningRate: 0.004,
    maxEpochs: 8,
  },
  {
    modelId: 'hybrid_sfnet',
    displayName: 'Hybrid Spatial-Frequency Net',
    family: 'Dual-Branch Spatial + Spectral DL',
    description:
      'Multi-input architecture fusing spatial Conv2D features with a 16-dimensional 2D-FFT, Laplacian noise, and ELA residual descriptor branch.',
    usesFrequencyBranch: true,
    initialLearningRate: 0.0035,
    maxEpochs: 8,
  },
];

export function getModelBlueprint(modelId: ModelId): ModelBlueprint {
  const found = MODEL_BLUEPRINTS.find((m) => m.modelId === modelId);
  if (!found) {
    throw new Error(`Unknown modelId: ${modelId}`);
  }
  return found;
}

export function buildModel(modelId: ModelId): tf.LayersModel {
  const size = TARGET_IMAGE_SIZE;

  if (modelId === 'cnn_baseline') {
    const input = tf.input({ shape: [size, size, 3], name: 'spatial_input' });
    let x = tf.layers
      .conv2d({ filters: 6, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' })
      .apply(input) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;

    x = tf.layers
      .conv2d({ filters: 12, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' })
      .apply(x) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;

    x = tf.layers.globalAveragePooling2d({}).apply(x) as tf.SymbolicTensor;
    x = tf.layers.dropout({ rate: 0.2 }).apply(x) as tf.SymbolicTensor;
    const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x) as tf.SymbolicTensor;
    return tf.model({ inputs: input, outputs: out, name: 'cnn_baseline' });
  }

  if (modelId === 'resnet_lite') {
    const input = tf.input({ shape: [size, size, 3], name: 'spatial_input' });
    const reg = tf.regularizers.l2({ l2: 1e-4 });

    let x = tf.layers
      .conv2d({
        filters: 8,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        kernelRegularizer: reg,
      })
      .apply(input) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
    x = tf.layers.reLU().apply(x) as tf.SymbolicTensor;

    // Residual Block (8 channels)
    let r1 = tf.layers
      .conv2d({ filters: 8, kernelSize: 3, padding: 'same', kernelRegularizer: reg })
      .apply(x) as tf.SymbolicTensor;
    r1 = tf.layers.batchNormalization().apply(r1) as tf.SymbolicTensor;
    r1 = tf.layers.reLU().apply(r1) as tf.SymbolicTensor;
    r1 = tf.layers
      .conv2d({ filters: 8, kernelSize: 3, padding: 'same', kernelRegularizer: reg })
      .apply(r1) as tf.SymbolicTensor;
    r1 = tf.layers.batchNormalization().apply(r1) as tf.SymbolicTensor;
    x = tf.layers.add().apply([x, r1]) as tf.SymbolicTensor;
    x = tf.layers.reLU().apply(x) as tf.SymbolicTensor;

    x = tf.layers.globalAveragePooling2d({}).apply(x) as tf.SymbolicTensor;
    x = tf.layers.dropout({ rate: 0.2 }).apply(x) as tf.SymbolicTensor;
    const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x) as tf.SymbolicTensor;
    return tf.model({ inputs: input, outputs: out, name: 'resnet_lite' });
  }

  if (modelId === 'xception_mobilenet') {
    const input = tf.input({ shape: [size, size, 3], name: 'spatial_input' });
    let x = tf.layers
      .separableConv2d({ filters: 8, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' })
      .apply(input) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;

    x = tf.layers
      .separableConv2d({ filters: 16, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' })
      .apply(x) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;

    x = tf.layers.globalAveragePooling2d({}).apply(x) as tf.SymbolicTensor;
    x = tf.layers.dropout({ rate: 0.2 }).apply(x) as tf.SymbolicTensor;
    const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x) as tf.SymbolicTensor;
    return tf.model({ inputs: input, outputs: out, name: 'xception_mobilenet' });
  }

  // Hybrid Spatial-Frequency Dual-Branch Network
  const spatialInput = tf.input({ shape: [size, size, 3], name: 'spatial_input' });
  const freqInput = tf.input({ shape: [16], name: 'frequency_input' });

  let sx = tf.layers
    .separableConv2d({ filters: 10, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' })
    .apply(spatialInput) as tf.SymbolicTensor;
  sx = tf.layers.batchNormalization().apply(sx) as tf.SymbolicTensor;
  sx = tf.layers
    .separableConv2d({ filters: 18, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' })
    .apply(sx) as tf.SymbolicTensor;
  sx = tf.layers.globalAveragePooling2d({}).apply(sx) as tf.SymbolicTensor;

  let fx = tf.layers
    .dense({ units: 24, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }) })
    .apply(freqInput) as tf.SymbolicTensor;
  fx = tf.layers.batchNormalization().apply(fx) as tf.SymbolicTensor;
  fx = tf.layers.dense({ units: 16, activation: 'relu' }).apply(fx) as tf.SymbolicTensor;

  let fused = tf.layers.concatenate().apply([sx, fx]) as tf.SymbolicTensor;
  fused = tf.layers.dropout({ rate: 0.2 }).apply(fused) as tf.SymbolicTensor;
  fused = tf.layers.dense({ units: 16, activation: 'relu' }).apply(fused) as tf.SymbolicTensor;
  const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(fused) as tf.SymbolicTensor;

  return tf.model({
    inputs: [spatialInput, freqInput],
    outputs: out,
    name: 'hybrid_sfnet',
  });
}

function createDiskSaveHandler(dirPath: string): tf.io.IOHandler {
  return {
    save: async (modelArtifacts: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> => {
      fs.mkdirSync(dirPath, { recursive: true });
      const weightsBuf = modelArtifacts.weightData
        ? Buffer.from(modelArtifacts.weightData as ArrayBuffer)
        : Buffer.alloc(0);
      fs.writeFileSync(path.join(dirPath, 'weights.bin'), weightsBuf);

      const jsonPayload = {
        modelTopology: modelArtifacts.modelTopology,
        format: modelArtifacts.format,
        generatedBy: modelArtifacts.generatedBy,
        convertedBy: modelArtifacts.convertedBy,
        weightsManifest: [
          {
            paths: ['weights.bin'],
            weights: modelArtifacts.weightSpecs ?? [],
          },
        ],
      };
      fs.writeFileSync(path.join(dirPath, 'model.json'), JSON.stringify(jsonPayload, null, 2));
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
          weightDataBytes: weightsBuf.byteLength,
        },
      };
    },
  };
}

function createDiskLoadHandler(dirPath: string): tf.io.IOHandler {
  return {
    load: async (): Promise<tf.io.ModelArtifacts> => {
      const jsonText = fs.readFileSync(path.join(dirPath, 'model.json'), 'utf-8');
      const parsed = JSON.parse(jsonText);
      const weightsBuf = fs.readFileSync(path.join(dirPath, 'weights.bin'));
      const weightData = weightsBuf.buffer.slice(
        weightsBuf.byteOffset,
        weightsBuf.byteOffset + weightsBuf.byteLength
      );
      return {
        modelTopology: parsed.modelTopology,
        format: parsed.format,
        generatedBy: parsed.generatedBy,
        convertedBy: parsed.convertedBy,
        weightSpecs: parsed.weightsManifest?.[0]?.weights ?? [],
        weightData,
      };
    },
  };
}

export async function saveModelToDisk(modelId: ModelId, model: tf.LayersModel): Promise<void> {
  const dir = path.join(MODELS_DIR, modelId);
  await model.save(createDiskSaveHandler(dir));
}

export async function loadModelFromDisk(modelId: ModelId): Promise<tf.LayersModel | null> {
  const modelJson = path.join(MODELS_DIR, modelId, 'model.json');
  const weightsBin = path.join(MODELS_DIR, modelId, 'weights.bin');
  if (!fs.existsSync(modelJson) || !fs.existsSync(weightsBin)) {
    return null;
  }
  return tf.loadLayersModel(createDiskLoadHandler(path.join(MODELS_DIR, modelId)));
}
