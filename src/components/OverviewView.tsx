import { motion } from 'motion/react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Cpu,
  Layers,
  ScanSearch,
  ShieldCheck,
  SlidersHorizontal,
} from 'lucide-react';
import type {
  BenchmarkSuiteResult,
  DetectionHistoryEntry,
  SampleGalleryItem,
  TrainingProgressState,
} from '../types.ts';

interface OverviewViewProps {
  benchmarks: BenchmarkSuiteResult | null;
  training: TrainingProgressState;
  samples: SampleGalleryItem[];
  history: DetectionHistoryEntry[];
  onGoAnalyze: () => void;
  onGoBenchmarks: () => void;
  onGoHistory: () => void;
  onAnalyzeSample: (filename: string) => void;
}

export function OverviewView({
  benchmarks,
  training,
  samples,
  history,
  onGoAnalyze,
  onGoBenchmarks,
  onGoHistory,
  onAnalyzeSample,
}: OverviewViewProps) {
  const bestModel =
    benchmarks?.models.find((m) => m.modelId === benchmarks.bestModelId) ?? null;
  const activeModel =
    benchmarks?.models.find((m) => m.modelId === benchmarks.activeModelId) ??
    bestModel;
  const cnnModel =
    benchmarks?.models.find((m) => m.modelId === 'cnn_baseline') ?? activeModel;

  return (
    <div className="space-y-10">
      {/* Hero Section with Animated Ambient Gradient */}
      <section className="relative overflow-hidden rounded-2xl">
        <div
          aria-hidden="true"
          className="hero-aurora pointer-events-none absolute -inset-24 bg-[radial-gradient(ellipse_at_top_left,rgba(59,130,246,0.22),transparent_55%),radial-gradient(ellipse_at_bottom_right,rgba(139,92,246,0.22),transparent_55%)]"
        />

        <div className="relative grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-stretch">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="glass-panel flex flex-col justify-between rounded-2xl p-8 lg:col-span-7"
          >
            <div className="space-y-4">
              <div className="text-xs font-semibold tracking-wider text-blue-400">
                CNN &amp; Multi-Architecture Forensics
              </div>
              <h1 className="font-display text-3xl font-extrabold tracking-tight text-white sm:text-5xl [text-wrap:balance]">
                DEEPFAKE DETECTOR
              </h1>
              <p className="text-lg font-medium text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-violet-300">
                AI-Powered Image Authenticity Verification.
              </p>
              <p className="max-w-2xl text-[15px] leading-relaxed text-slate-300">
                Upload any JPG, JPEG, or PNG portrait to classify authenticity using
                our trained Convolutional Neural Network and benchmarked deepfake
                architectures—paired with Grad-CAM attribution heatmaps, 2D-FFT
                spectral checks, and error-level compression residuals.
              </p>
            </div>

            <div className="mt-8 space-y-5">
              {benchmarks ? (
                <div className="rounded-xl border border-slate-700/40 bg-slate-900/55 p-4 text-xs leading-relaxed text-slate-300">
                  <span className="font-semibold text-blue-300">
                    Active Model Telemetry:
                  </span>{' '}
                  {benchmarks.selectionRationale}
                </div>
              ) : (
                <div className="rounded-xl border border-blue-500/30 bg-blue-950/30 p-4 text-xs text-blue-200">
                  {training.isTraining
                    ? `Training neural models: ${training.stage} (${training.completedModels}/${training.totalModels}, epoch ${training.currentEpoch}/${training.totalEpochs})...`
                    : 'Initializing trained CNN & benchmark suite...'}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={onGoAnalyze}
                  className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-2.5 text-sm font-semibold text-white transition-all hover:opacity-95 hover:scale-[1.01] whitespace-nowrap shrink-0"
                >
                  <ScanSearch className="h-4 w-4" />
                  Upload &amp; Verify Image
                  <ArrowRight className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  onClick={onGoBenchmarks}
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-700/70 bg-slate-900/60 px-5 py-2.5 text-sm font-medium text-slate-200 transition-colors hover:border-slate-500 hover:text-white whitespace-nowrap shrink-0"
                >
                  <Activity className="h-4 w-4 text-violet-400" />
                  Model Performance &amp; Training
                </button>
              </div>
            </div>
          </motion.div>

          {/* Right Hero Column: Instant Test Portraits */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.06, ease: [0.16, 1, 0.3, 1] }}
            className="glass-panel flex flex-col justify-between rounded-2xl p-6 lg:col-span-5"
          >
            <div className="mb-4">
              <h2 className="font-display text-base font-semibold text-white">
                Instant Verification Samples
              </h2>
              <p className="text-xs text-slate-400">
                Click any portrait below to run live CNN inference and inspect its heatmap
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {samples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => onAnalyzeSample(s.filename)}
                  disabled={!benchmarks}
                  className="group relative overflow-hidden rounded-xl border border-slate-700/50 bg-slate-900/80 text-left transition-all hover:border-blue-400/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:opacity-50"
                >
                  <div className="aspect-square w-full overflow-hidden bg-slate-900">
                    <img
                      src={s.url}
                      alt={s.title}
                      referrerPolicy="no-referrer"
                      className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                  </div>
                  <div className="bg-gradient-to-t from-[#050814]/95 via-[#050814]/75 to-transparent p-2">
                    <div className="truncate text-[11px] font-medium text-slate-100">
                      {s.title}
                    </div>
                    <div className="truncate text-[10px] text-slate-400">
                      {s.category}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Full-Width 4-Column Model Performance Summary Strip */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs font-medium text-slate-400">
            Selected Best Model
          </div>
          <div className="mt-2 font-display text-lg font-semibold text-white truncate">
            {activeModel ? activeModel.displayName : 'Training...'}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            {activeModel ? activeModel.family : 'Initializing weights'}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs font-medium text-slate-400">
            Test Accuracy &amp; F1 Score
          </div>
          <div className="mt-2 font-mono text-2xl font-bold tabular-nums text-blue-300">
            {activeModel ? `${(activeModel.accuracy * 100).toFixed(1)}%` : '—'}
          </div>
          <div className="mt-1 font-mono text-xs tabular-nums text-slate-400">
            {activeModel
              ? `F1 ${(activeModel.f1Score * 100).toFixed(1)}% · ROC-AUC ${(activeModel.rocAuc * 100).toFixed(1)}%`
              : 'Evaluating held-out test split'}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs font-medium text-slate-400">
            CNN Baseline Test AUC
          </div>
          <div className="mt-2 font-mono text-2xl font-bold tabular-nums text-violet-300">
            {cnnModel ? `${(cnnModel.rocAuc * 100).toFixed(1)}%` : '—'}
          </div>
          <div className="mt-1 font-mono text-xs tabular-nums text-slate-400">
            {cnnModel
              ? `Precision ${(cnnModel.precision * 100).toFixed(1)}% · ${cnnModel.inferenceTimeMs.toFixed(2)} ms/img`
              : 'Measuring CNN metrics'}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs font-medium text-slate-400">
            Verified Dataset Split
          </div>
          <div className="mt-2 font-mono text-2xl font-bold tabular-nums text-emerald-300">
            {benchmarks
              ? `${benchmarks.datasetStats.validCount} images`
              : '—'}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            {benchmarks
              ? `Train ${benchmarks.datasetStats.splits.train.total} · Val ${benchmarks.datasetStats.splits.val.total} · Test ${benchmarks.datasetStats.splits.test.total}`
              : 'Zero-leakage group split'}
          </div>
        </div>
      </section>

      {/* Architecture Comparison + Recent Detections Preview */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="glass-panel rounded-2xl p-6 lg:col-span-7">
          <div className="mb-4 flex items-center justify-between gap-2">
            <div>
              <h2 className="font-display text-lg font-semibold text-white">
                Evaluated Neural Architectures
              </h2>
              <p className="text-xs text-slate-400">
                Real test-split metrics across CNN, Residual, Depthwise-Separable, and Hybrid architectures
              </p>
            </div>
            <button
              type="button"
              onClick={onGoBenchmarks}
              className="text-xs font-medium text-blue-400 hover:text-blue-300 whitespace-nowrap"
            >
              Full Analytics →
            </button>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {(benchmarks?.models ?? []).map((m) => {
              const isBest = m.modelId === benchmarks?.bestModelId;
              return (
                <div
                  key={m.modelId}
                  className={`rounded-xl border p-4 transition-colors ${
                    isBest
                      ? 'border-blue-500/45 bg-blue-950/20'
                      : 'border-slate-800 bg-slate-900/50'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-medium text-blue-300">
                      {m.family}
                    </span>
                    {isBest && (
                      <span className="text-[11px] font-semibold text-emerald-400">
                        Best Selected
                      </span>
                    )}
                  </div>
                  <div className="mt-1 font-display text-sm font-semibold text-white">
                    {m.displayName}
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
                    <div>
                      <div className="text-slate-500">Acc</div>
                      <div className="text-slate-100">{(m.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-slate-500">F1</div>
                      <div className="text-slate-100">{(m.f1Score * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-slate-500">AUC</div>
                      <div className="text-violet-300">{(m.rocAuc * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent Detection History Preview */}
        <div className="glass-panel flex flex-col justify-between rounded-2xl p-6 lg:col-span-5">
          <div>
            <div className="mb-4 flex items-center justify-between gap-2">
              <div>
                <h2 className="font-display text-lg font-semibold text-white">
                  Recent Detections
                </h2>
                <p className="text-xs text-slate-400">
                  Latest verified images and confidence scores
                </p>
              </div>
              <button
                type="button"
                onClick={onGoHistory}
                className="text-xs font-medium text-blue-400 hover:text-blue-300 whitespace-nowrap"
              >
                View All →
              </button>
            </div>

            {history.length === 0 ? (
              <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 text-center text-xs text-slate-400">
                No detections logged yet. Click any sample portrait above or upload an image to run verification.
              </div>
            ) : (
              <div className="divide-y divide-slate-800/70">
                {history.slice(0, 4).map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-center justify-between gap-3 py-2.5"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <img
                        src={entry.thumbnailDataUrl}
                        alt={entry.filename}
                        className="h-9 w-9 rounded-lg border border-slate-700 object-cover shrink-0"
                      />
                      <div className="min-w-0">
                        <div className="truncate text-xs font-medium text-white">
                          {entry.filename}
                        </div>
                        <div className="text-[11px] text-slate-400">
                          {entry.modelDisplayName}
                        </div>
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div
                        className={`inline-flex items-center gap-1 text-xs font-semibold ${
                          entry.prediction === 'REAL'
                            ? 'text-emerald-400'
                            : 'text-rose-400'
                        }`}
                      >
                        {entry.prediction === 'REAL' ? (
                          <ShieldCheck className="h-3.5 w-3.5" />
                        ) : (
                          <AlertTriangle className="h-3.5 w-3.5" />
                        )}
                        {entry.prediction}
                      </div>
                      <div className="font-mono text-[11px] tabular-nums text-slate-400">
                        {entry.confidence.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Supporting Forensic Methodology */}
      <section className="glass-panel rounded-2xl p-6">
        <div className="mb-5">
          <h2 className="font-display text-lg font-semibold text-white">
            Supporting Forensic Signals &amp; Python Keras Suite
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Auxiliary computer-vision indicators presented strictly as supporting context alongside neural predictions.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div className="space-y-2">
            <Layers className="h-5 w-5 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">
              Grad-CAM Attribution
            </h3>
            <p className="text-xs leading-relaxed text-slate-400">
              Combines backpropagated input-gradient sensitivity with spatial occlusion deltas on the trained model.
            </p>
          </div>
          <div className="space-y-2">
            <Activity className="h-5 w-5 text-violet-400" />
            <h3 className="text-sm font-semibold text-white">
              2D-FFT Spectral Analysis
            </h3>
            <p className="text-xs leading-relaxed text-slate-400">
              Computes Hann-windowed 2D Fourier log-magnitude energy and radial roll-off for upsampling artifacts.
            </p>
          </div>
          <div className="space-y-2">
            <SlidersHorizontal className="h-5 w-5 text-sky-400" />
            <h3 className="text-sm font-semibold text-white">
              Error Level Analysis
            </h3>
            <p className="text-xs leading-relaxed text-slate-400">
              Measures block-wise JPEG recompression residual variance across potential splice zones.
            </p>
          </div>
          <div className="space-y-2">
            <Cpu className="h-5 w-5 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">
              Keras 128×128 Pipeline
            </h3>
            <p className="text-xs leading-relaxed text-slate-400">
              Complete TensorFlow/Keras CNN training, Pandas/Matplotlib evaluation, and FastAPI server in <code className="text-slate-300">python_pipeline/</code>.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
