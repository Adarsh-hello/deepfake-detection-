export type ModelId =
  | 'cnn_baseline'
  | 'resnet_lite'
  | 'xception_mobilenet'
  | 'hybrid_sfnet';

export type SplitName = 'train' | 'val' | 'test';
export type LabelName = 'real' | 'fake';

export interface DatasetSample {
  id: string;
  filename: string;
  label: LabelName;
  split: SplitName;
  sourceGroupId: string;
  sha256: string;
  dHash: string;
  width: number;
  height: number;
  filePath: string;
  isUserUploaded: boolean;
}

export interface DatasetStats {
  totalScanned: number;
  validCount: number;
  corruptedRemoved: number;
  duplicatesRemoved: number;
  realCount: number;
  fakeCount: number;
  imbalanceRatio: number;
  classWeights: {
    real: number;
    fake: number;
  };
  splits: {
    train: { real: number; fake: number; total: number };
    val: { real: number; fake: number; total: number };
    test: { real: number; fake: number; total: number };
  };
  leakageVerified: boolean;
  uniqueSourceGroups: number;
  targetSize: number;
  augmentationMethods: string[];
  lastPreparedAt: string;
}

export interface EpochMetric {
  epoch: number;
  trainLoss: number;
  trainAccuracy: number;
  valLoss: number;
  valAccuracy: number;
  learningRate: number;
}

export interface ConfusionMatrixData {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

export interface ClassReportRow {
  label: string;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ModelEvaluation {
  modelId: ModelId;
  displayName: string;
  family: string;
  description: string;
  usesFrequencyBranch: boolean;
  parameterCount: number;
  epochsTrained: number;
  bestEpoch: number;
  earlyStopped: boolean;
  trainingTimeMs: number;
  inferenceTimeMs: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  specificity: number;
  rocAuc: number;
  selectionScore: number;
  confusionMatrix: ConfusionMatrixData;
  rocCurve: RocPoint[];
  classificationReport: ClassReportRow[];
  history: EpochMetric[];
  trainedAt: string;
}

export interface BenchmarkSuiteResult {
  bestModelId: ModelId;
  activeModelId: ModelId;
  selectionRationale: string;
  models: ModelEvaluation[];
  datasetStats: DatasetStats;
  updatedAt: string;
}

export interface TrainingProgressState {
  isTraining: boolean;
  currentModelId: ModelId | null;
  currentModelName: string | null;
  completedModels: number;
  totalModels: number;
  currentEpoch: number;
  totalEpochs: number;
  stage: string;
  lastError: string | null;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface ForensicIndicator {
  id: string;
  name: string;
  category: 'frequency' | 'texture' | 'compression' | 'structural';
  value: number;
  unit: string;
  baselineRange: string;
  assessment: 'nominal' | 'elevated' | 'anomalous';
  summary: string;
}

export interface ForensicAnalysisResult {
  id: string;
  filename: string;
  imageWidth: number;
  imageHeight: number;
  sha256: string;
  analyzedAt: string;
  modelId: ModelId;
  modelDisplayName: string;
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  fakeProbability: number;
  realProbability: number;
  inferenceTimeMs: number;
  originalDataUrl: string;
  gradCamHeatmapDataUrl: string;
  elaHeatmapDataUrl: string;
  textureNoiseDataUrl: string;
  fftSpectrumDataUrl: string;
  radialSpectrum: Array<{ band: string; power: number }>;
  indicators: ForensicIndicator[];
  disclaimer: string;
}

export interface DetectionHistoryEntry {
  id: string;
  filename: string;
  analyzedAt: string;
  modelId: ModelId;
  modelDisplayName: string;
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  fakeProbability: number;
  inferenceTimeMs: number;
  thumbnailDataUrl: string;
  sha256: string;
  report: ForensicAnalysisResult;
}
