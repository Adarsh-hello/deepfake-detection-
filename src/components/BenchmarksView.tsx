import { useRef, useState } from 'react';
import {
  CheckCircle2,
  Database,
  Loader2,
  Play,
  RefreshCw,
  UploadCloud,
} from 'lucide-react';
import type {
  BenchmarkSuiteResult,
  DatasetStats,
  ModelId,
  TrainingProgressState,
} from '../types.ts';
import {
  ConfusionMatrixChart,
  RocCurveChart,
  TrainingCurvesChart,
} from './Charts.tsx';

interface BenchmarksViewProps {
  benchmarks: BenchmarkSuiteResult | null;
  datasetStats: DatasetStats | null;
  training: TrainingProgressState;
  onStartTraining: () => void;
  onSelectActiveModel: (id: ModelId) => void;
  onUploadDatasetSamples: (
    files: FileList,
    label: 'real' | 'fake'
  ) => Promise<string>;
  onAuditDataset: () => void;
}

export function BenchmarksView({
  benchmarks,
  datasetStats,
  training,
  onStartTraining,
  onSelectActiveModel,
  onUploadDatasetSamples,
  onAuditDataset,
}: BenchmarksViewProps) {
  const [inspectedModelId, setInspectedModelId] = useState<ModelId | null>(null);
  const [uploadLabel, setUploadLabel] = useState<'real' | 'fake'>('real');
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const dsInputRef = useRef<HTMLInputElement>(null);

  const models = benchmarks?.models ?? [];
  const sortedModels = [...models].sort(
    (a, b) => b.selectionScore - a.selectionScore
  );

  const inspectedModel =
    sortedModels.find((m) => m.modelId === inspectedModelId) ??
    sortedModels.find((m) => m.modelId === benchmarks?.bestModelId) ??
    sortedModels[0] ??
    null;

  const stats = datasetStats ?? benchmarks?.datasetStats ?? null;

  const handleDatasetFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setUploadStatus(null);
    try {
      const msg = await onUploadDatasetSamples(files, uploadLabel);
      setUploadStatus(msg);
    } catch (err) {
      setUploadStatus(
        err instanceof Error ? err.message : 'Dataset upload failed.'
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Training Header & Live Progress */}
      <div className="glass-panel rounded-2xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="font-display text-2xl font-bold text-white">
              Intelligent Model Selection &amp; Benchmarks
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              All models are trained and evaluated on identical source-grouped splits with early stopping and learning-rate scheduling.
            </p>
          </div>
          <button
            type="button"
            disabled={training.isTraining}
            onClick={onStartTraining}
            className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-4 py-2.5 text-xs font-semibold text-white transition-opacity hover:opacity-95 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            {training.isTraining ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Training in Progress...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Retrain &amp; Benchmark All Models
              </>
            )}
          </button>
        </div>

        {training.isTraining && (
          <div className="mt-5 rounded-xl border border-blue-500/35 bg-blue-950/25 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-medium text-blue-200">
                {training.stage}
                {training.currentModelName ? ` — ${training.currentModelName}` : ''}
              </span>
              <span className="font-mono tabular-nums text-blue-300">
                Model {Math.min(training.completedModels + 1, training.totalModels)}/{training.totalModels} · Epoch {training.currentEpoch}/{training.totalEpochs || '—'}
              </span>
            </div>
            <div className="mt-2.5 h-2 w-full overflow-hidden rounded-full bg-slate-900">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-300"
                style={{
                  width: `${Math.min(
                    100,
                    Math.max(
                      6,
                      ((training.completedModels +
                        (training.totalEpochs > 0
                          ? training.currentEpoch / training.totalEpochs
                          : 0)) /
                        Math.max(1, training.totalModels)) *
                        100
                    )
                  )}%`,
                }}
              />
            </div>
          </div>
        )}

        {benchmarks && !training.isTraining && (
          <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-950/20 px-4 py-3 text-xs text-emerald-200">
            <span className="font-semibold">Automatic Selection Rationale:</span>{' '}
            {benchmarks.selectionRationale}
          </div>
        )}
      </div>

      {/* Model Comparison Table */}
      {sortedModels.length > 0 && (
        <div className="glass-panel rounded-2xl p-6">
          <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
            <div>
              <h2 className="font-display text-lg font-semibold text-white">
                Model Performance Comparison Table
              </h2>
              <p className="text-xs text-slate-400">
                Click any row to inspect confusion matrix, training curves, and classification report
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="pb-3 pr-4 font-medium">Architecture</th>
                  <th className="pb-3 px-2 text-right font-medium">Accuracy</th>
                  <th className="pb-3 px-2 text-right font-medium">Precision</th>
                  <th className="pb-3 px-2 text-right font-medium">Recall</th>
                  <th className="pb-3 px-2 text-right font-medium">F1 Score</th>
                  <th className="pb-3 px-2 text-right font-medium">ROC-AUC</th>
                  <th className="pb-3 px-2 text-right font-medium">Train Time</th>
                  <th className="pb-3 px-2 text-right font-medium">Latency</th>
                  <th className="pb-3 px-2 text-right font-medium">Score</th>
                  <th className="pb-3 pl-3 text-right font-medium">Inference Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {sortedModels.map((m) => {
                  const isBest = m.modelId === benchmarks?.bestModelId;
                  const isActive = m.modelId === benchmarks?.activeModelId;
                  const isInspected = inspectedModel?.modelId === m.modelId;

                  return (
                    <tr
                      key={m.modelId}
                      onClick={() => setInspectedModelId(m.modelId)}
                      className={`cursor-pointer transition-colors ${
                        isInspected
                          ? 'bg-blue-950/35'
                          : 'hover:bg-slate-900/50'
                      }`}
                    >
                      <td className="py-3.5 pr-4">
                        <div className="font-medium text-white">
                          {m.displayName}
                        </div>
                        <div className="text-[11px] text-slate-400">
                          {m.family} · {m.parameterCount.toLocaleString()} params
                        </div>
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums text-slate-200">
                        {(m.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums text-slate-200">
                        {(m.precision * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums text-slate-200">
                        {(m.recall * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums font-semibold text-blue-300">
                        {(m.f1Score * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums font-semibold text-violet-300">
                        {(m.rocAuc * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums text-slate-400">
                        {(m.trainingTimeMs / 1000).toFixed(1)}s
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums text-slate-400">
                        {m.inferenceTimeMs.toFixed(2)} ms
                      </td>
                      <td className="px-2 py-3.5 text-right font-mono tabular-nums text-slate-100">
                        {m.selectionScore.toFixed(3)}
                      </td>
                      <td
                        className="py-3.5 pl-3 text-right"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {isActive ? (
                          <span className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-400 whitespace-nowrap">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            Active{isBest ? ' · Best' : ''}
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => onSelectActiveModel(m.modelId)}
                            className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-[11px] font-medium text-slate-200 transition-colors hover:border-blue-500 hover:text-white whitespace-nowrap"
                          >
                            Use for Inference
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      {inspectedModel && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ConfusionMatrixChart
            matrix={inspectedModel.confusionMatrix}
            modelName={inspectedModel.displayName}
          />
          <RocCurveChart models={sortedModels} />
          <TrainingCurvesChart
            history={inspectedModel.history}
            modelName={inspectedModel.displayName}
          />

          {/* Classification Report */}
          <div className="glass-panel rounded-xl p-5">
            <div className="mb-3">
              <h3 className="font-display text-base font-semibold text-white">
                Classification Report
              </h3>
              <p className="text-xs text-slate-400">
                {inspectedModel.displayName} · Best Epoch {inspectedModel.bestEpoch}/{inspectedModel.epochsTrained}
                {inspectedModel.earlyStopped ? ' (Early Stopped)' : ''}
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="pb-2 font-medium">Class / Average</th>
                    <th className="pb-2 text-right font-medium">Precision</th>
                    <th className="pb-2 text-right font-medium">Recall</th>
                    <th className="pb-2 text-right font-medium">F1-Score</th>
                    <th className="pb-2 text-right font-medium">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                  {inspectedModel.classificationReport.map((row) => (
                    <tr key={row.label}>
                      <td className="py-2.5 font-sans font-medium text-slate-200">
                        {row.label}
                      </td>
                      <td className="py-2.5 text-right text-slate-200">
                        {(row.precision * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-right text-slate-200">
                        {(row.recall * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-right text-blue-300">
                        {(row.f1Score * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-right text-slate-400">
                        {row.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Dataset Preparation & Custom Sample Upload */}
      {stats && (
        <div className="glass-panel rounded-2xl p-6">
          <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <Database className="h-5 w-5 text-blue-400" />
              <div>
                <h2 className="font-display text-lg font-semibold text-white">
                  Dataset Preparation &amp; Hygiene Audit
                </h2>
                <p className="text-xs text-slate-400">
                  SHA-256 &amp; perceptual dHash deduplication · Source-grouped splits preventing data leakage
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onAuditDataset}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-500 whitespace-nowrap shrink-0"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Re-Audit Dataset
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3.5">
              <div className="text-[11px] text-slate-400">Valid Samples</div>
              <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-white">
                {stats.validCount}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3.5">
              <div className="text-[11px] text-slate-400">Corrupted Removed</div>
              <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-slate-200">
                {stats.corruptedRemoved}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3.5">
              <div className="text-[11px] text-slate-400">Duplicates Purged</div>
              <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-slate-200">
                {stats.duplicatesRemoved}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3.5">
              <div className="text-[11px] text-slate-400">Train / Val / Test</div>
              <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-blue-300">
                {stats.splits.train.total}/{stats.splits.val.total}/{stats.splits.test.total}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3.5">
              <div className="text-[11px] text-slate-400">Class Weights (R/F)</div>
              <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-slate-200">
                {stats.classWeights.real.toFixed(2)} / {stats.classWeights.fake.toFixed(2)}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3.5">
              <div className="text-[11px] text-slate-400">Split Leakage Check</div>
              <div
                className={`mt-1 text-sm font-semibold ${
                  stats.leakageVerified ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {stats.leakageVerified ? '✓ Disjoint Groups' : '▲ Overlap Detected'}
              </div>
            </div>
          </div>

          {/* Expand Dataset Form */}
          <div className="mt-5 flex flex-wrap items-center justify-between gap-4 border-t border-slate-800/80 pt-5">
            <div className="text-xs text-slate-300">
              <span className="font-medium text-white">Expand Training Corpus:</span>{' '}
              Upload custom JPEG/PNG images labeled as Real or Fake, then click Retrain Models.
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-900/90 p-1">
                <button
                  type="button"
                  onClick={() => setUploadLabel('real')}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                    uploadLabel === 'real'
                      ? 'bg-emerald-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Label: REAL
                </button>
                <button
                  type="button"
                  onClick={() => setUploadLabel('fake')}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                    uploadLabel === 'fake'
                      ? 'bg-rose-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Label: FAKE
                </button>
              </div>

              <input
                ref={dsInputRef}
                type="file"
                multiple
                accept="image/jpeg,image/png"
                className="hidden"
                onChange={(e) => void handleDatasetFiles(e.target.files)}
              />
              <button
                type="button"
                disabled={isUploading}
                onClick={() => dsInputRef.current?.click()}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3.5 py-2 text-xs font-medium text-slate-200 hover:border-blue-500 disabled:opacity-50 whitespace-nowrap shrink-0"
              >
                <UploadCloud className="h-3.5 w-3.5 text-blue-400" />
                {isUploading ? 'Ingesting...' : 'Add Images to Dataset'}
              </button>
            </div>
          </div>

          {uploadStatus && (
            <div className="mt-3 text-xs text-blue-300">{uploadStatus}</div>
          )}
        </div>
      )}
    </div>
  );
}
