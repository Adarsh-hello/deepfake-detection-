import { useRef, useState } from 'react';
import {
  AlertTriangle,
  Download,
  FileJson,
  Info,
  Loader2,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import type {
  BenchmarkSuiteResult,
  ForensicAnalysisResult,
  ModelId,
  SampleGalleryItem,
} from '../types.ts';
import { downloadHtmlReport, downloadJsonReport } from '../utils/reportExport.ts';
import { RadialSpectrumChart } from './Charts.tsx';

type OverlayMode = 'original' | 'gradcam' | 'ela' | 'noise' | 'fft';

interface AnalyzeViewProps {
  benchmarks: BenchmarkSuiteResult | null;
  samples: SampleGalleryItem[];
  result: ForensicAnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
  selectedModelId: ModelId | undefined;
  onSelectModelId: (id: ModelId) => void;
  onAnalyzeFile: (file: File, modelId?: ModelId) => void;
  onAnalyzeSample: (filename: string, modelId?: ModelId) => void;
}

export function AnalyzeView({
  benchmarks,
  samples,
  result,
  isAnalyzing,
  error,
  selectedModelId,
  onSelectModelId,
  onAnalyzeFile,
  onAnalyzeSample,
}: AnalyzeViewProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('gradcam');
  const [overlayOpacity, setOverlayOpacity] = useState(65);

  const activeModelId =
    selectedModelId ?? benchmarks?.activeModelId ?? 'hybrid_sfnet';

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    onAnalyzeFile(files[0], activeModelId);
  };

  const overlayUrl =
    !result || overlayMode === 'original'
      ? null
      : overlayMode === 'gradcam'
        ? result.gradCamHeatmapDataUrl
        : overlayMode === 'ela'
          ? result.elaHeatmapDataUrl
          : overlayMode === 'noise'
            ? result.textureNoiseDataUrl
            : result.fftSpectrumDataUrl;

  const isStandaloneLayer = overlayMode === 'fft';

  return (
    <div className="space-y-8">
      {/* Top Upload & Model Selection Bar */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`glass-panel flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-7 text-center transition-colors lg:col-span-7 ${
            isDragging
              ? 'border-blue-400 bg-blue-950/30'
              : 'border-slate-700/70 hover:border-slate-500'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-500/15 text-blue-300">
            {isAnalyzing ? (
              <Loader2 className="h-6 w-6 animate-spin" />
            ) : (
              <Upload className="h-6 w-6" />
            )}
          </div>
          <h2 className="mt-3 font-display text-base font-semibold text-white">
            {isAnalyzing
              ? 'Running Neural Inference & Forensic Extraction...'
              : 'Drop a JPEG or PNG Image for Forensic Inspection'}
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Decoded server-side, evaluated by your selected trained neural model, with Grad-CAM, ELA, and 2D-FFT overlays
          </p>
          <button
            type="button"
            disabled={isAnalyzing || !benchmarks}
            onClick={() => fileInputRef.current?.click()}
            className="mt-4 rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-4 py-2 text-xs font-semibold text-white transition-opacity hover:opacity-95 disabled:opacity-50"
          >
            Select Image File
          </button>
        </div>

        {/* Model Selector & Quick Samples */}
        <div className="glass-panel flex flex-col justify-between rounded-2xl p-6 lg:col-span-5">
          <div>
            <label
              htmlFor="inference-model-select"
              className="block text-xs font-medium text-slate-300"
            >
              Active Inference Architecture
            </label>
            <select
              id="inference-model-select"
              value={activeModelId}
              disabled={!benchmarks || isAnalyzing}
              onChange={(e) => onSelectModelId(e.target.value as ModelId)}
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-900/90 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
            >
              {(benchmarks?.models ?? []).map((m) => (
                <option key={m.modelId} value={m.modelId}>
                  {m.displayName} (AUC {(m.rocAuc * 100).toFixed(1)}%
                  {m.modelId === benchmarks?.bestModelId ? ' — Best' : ''})
                </option>
              ))}
            </select>
          </div>

          <div className="mt-5">
            <div className="mb-2 text-xs font-medium text-slate-400">
              Quick Sample Test Portraits
            </div>
            <div className="grid grid-cols-6 gap-2">
              {samples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  disabled={!benchmarks || isAnalyzing}
                  onClick={() => onAnalyzeSample(s.filename, activeModelId)}
                  title={s.title}
                  className="overflow-hidden rounded-lg border border-slate-700/70 bg-slate-900 transition-all hover:border-blue-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:opacity-50"
                >
                  <img
                    src={s.url}
                    alt={s.title}
                    referrerPolicy="no-referrer"
                    className="aspect-square w-full object-cover"
                  />
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/35 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      {/* Results Viewport */}
      {result ? (
        <div className="space-y-8">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Visual Inspection & Overlay Stage */}
            <div className="glass-panel rounded-2xl p-6 lg:col-span-7">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-slate-900/90 p-1">
                  {(
                    [
                      { id: 'original', label: 'Original' },
                      { id: 'gradcam', label: 'Grad-CAM' },
                      { id: 'ela', label: 'ELA Residual' },
                      { id: 'noise', label: 'Texture Noise' },
                      { id: 'fft', label: '2D-FFT' },
                    ] as Array<{ id: OverlayMode; label: string }>
                  ).map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setOverlayMode(tab.id)}
                      className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                        overlayMode === tab.id
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {overlayUrl && !isStandaloneLayer && (
                  <div className="flex items-center gap-2.5 text-xs text-slate-300">
                    <span>Overlay</span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                      className="w-24 accent-blue-500"
                      aria-label="Heatmap overlay opacity"
                    />
                    <span className="font-mono w-9 text-right tabular-nums">
                      {overlayOpacity}%
                    </span>
                  </div>
                )}
              </div>

              <div className="relative mx-auto aspect-square w-full max-w-[460px] overflow-hidden rounded-xl border border-slate-800 bg-[#040610]">
                {!isStandaloneLayer && (
                  <img
                    src={result.originalDataUrl}
                    alt={result.filename}
                    className="h-full w-full object-contain"
                  />
                )}
                {overlayUrl && (
                  <img
                    src={overlayUrl}
                    alt={`${overlayMode} visualization`}
                    className="absolute inset-0 h-full w-full object-contain transition-opacity duration-150"
                    style={{
                      opacity: isStandaloneLayer ? 1 : overlayOpacity / 100,
                    }}
                  />
                )}
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
                <span>
                  {overlayMode === 'original' && 'Unaltered decoded image preview'}
                  {overlayMode === 'gradcam' &&
                    'Model gradient + 8×8 spatial occlusion sensitivity attribution'}
                  {overlayMode === 'ela' &&
                    'JPEG Q=90 recompression residual difference heatmap'}
                  {overlayMode === 'noise' &&
                    'Laplacian high-pass texture and local noise residual density'}
                  {overlayMode === 'fft' &&
                    'Centered 2D Discrete Fourier Transform log-magnitude power spectrum'}
                </span>
                {overlayMode !== 'original' && (
                  <div className="flex items-center gap-2">
                    <span>Low</span>
                    <div className="h-2 w-24 rounded-full bg-gradient-to-r from-[#0a0c22] via-[#dd513a] to-[#fcffa4]" />
                    <span>High</span>
                  </div>
                )}
              </div>
            </div>

            {/* Prediction Verdict & Telemetry Panel */}
            <div className="glass-panel flex flex-col justify-between rounded-2xl p-6 lg:col-span-5">
              <div className="space-y-6">
                <div>
                  <div className="text-xs font-medium text-slate-400">
                    Classification Verdict
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-4">
                    <div
                      className={`inline-flex items-center gap-2.5 text-3xl font-bold font-display ${
                        result.prediction === 'REAL'
                          ? 'text-emerald-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {result.prediction === 'REAL' ? (
                        <ShieldCheck className="h-8 w-8" />
                      ) : (
                        <AlertTriangle className="h-8 w-8" />
                      )}
                      <span>{result.prediction}</span>
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-2xl font-bold tabular-nums text-white">
                        {result.confidence.toFixed(2)}%
                      </div>
                      <div className="text-xs text-slate-400">Confidence</div>
                    </div>
                  </div>
                </div>

                {/* Animated Confidence Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between font-mono text-xs tabular-nums">
                    <span className="text-slate-300">Model Confidence Score</span>
                    <span className="font-semibold text-white">
                      {result.confidence.toFixed(2)}%
                    </span>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-900 border border-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ease-out ${
                        result.prediction === 'REAL'
                          ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                          : 'bg-gradient-to-r from-rose-500 to-amber-400'
                      }`}
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Probability Distribution Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between font-mono text-xs tabular-nums">
                    <span className="text-emerald-300">
                      P(REAL) {(result.realProbability * 100).toFixed(2)}%
                    </span>
                    <span className="text-rose-300">
                      P(FAKE) {(result.fakeProbability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex h-2 w-full overflow-hidden rounded-full bg-slate-900">
                    <div
                      className="bg-emerald-500 transition-all duration-300"
                      style={{ width: `${result.realProbability * 100}%` }}
                    />
                    <div
                      className="bg-rose-500 transition-all duration-300"
                      style={{ width: `${result.fakeProbability * 100}%` }}
                    />
                  </div>
                </div>

                {/* Prediction Explanation */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/55 p-3.5 text-xs leading-relaxed text-slate-300">
                  <span className="font-semibold text-white">
                    Prediction Explanation:
                  </span>{' '}
                  {result.modelDisplayName} assigned a{' '}
                  <span className="font-mono text-slate-100">
                    {(result.fakeProbability * 100).toFixed(1)}%
                  </span>{' '}
                  synthetic probability ({result.prediction}).{' '}
                  {(() => {
                    const flagged = result.indicators.filter(
                      (i) => i.assessment !== 'nominal'
                    );
                    if (flagged.length === 0) {
                      return 'Supporting frequency, ELA, and texture noise metrics all remain within nominal baseline ranges.';
                    }
                    return `Supporting signals flagged ${flagged
                      .map((f) => `${f.name} (${f.value} ${f.unit})`)
                      .join(', ')} as elevated/anomalous.`;
                  })()}
                </div>

                {/* Metadata Grid */}
                <div className="grid grid-cols-2 gap-3 border-t border-slate-800/80 pt-5 text-xs">
                  <div>
                    <div className="text-slate-500">Model Used</div>
                    <div className="mt-0.5 font-medium text-slate-100">
                      {result.modelDisplayName}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500">Inference Latency</div>
                    <div className="mt-0.5 font-mono tabular-nums text-slate-100">
                      {result.inferenceTimeMs.toFixed(2)} ms
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500">Input Dimensions</div>
                    <div className="mt-0.5 font-mono tabular-nums text-slate-100">
                      {result.imageWidth} × {result.imageHeight} px
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500">SHA-256 Digest</div>
                    <div className="mt-0.5 truncate font-mono text-slate-300">
                      {result.sha256.slice(0, 16)}…
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 space-y-3 border-t border-slate-800/80 pt-5">
                <div className="text-xs font-medium text-slate-400">
                  Export Forensic Report
                </div>
                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => downloadHtmlReport(result)}
                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-blue-600/90 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-500 whitespace-nowrap shrink-0"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download HTML Report
                  </button>
                  <button
                    type="button"
                    onClick={() => downloadJsonReport(result)}
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-4 py-2 text-xs font-medium text-slate-200 transition-colors hover:border-slate-500 whitespace-nowrap shrink-0"
                  >
                    <FileJson className="h-3.5 w-3.5" />
                    Export JSON
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Supporting Indicators + Radial Power Spectrum */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="glass-panel rounded-2xl p-6 lg:col-span-7">
              <div className="mb-4">
                <h3 className="font-display text-base font-semibold text-white">
                  Supporting Forensic Indicators
                </h3>
                <p className="text-xs text-slate-400">
                  Auxiliary statistical signals extracted directly from pixel, spectral, and compression residuals
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="pb-2.5 font-medium">Indicator</th>
                      <th className="pb-2.5 text-right font-medium">Measured</th>
                      <th className="pb-2.5 text-right font-medium">Reference</th>
                      <th className="pb-2.5 text-right font-medium">Signal</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {result.indicators.map((ind) => (
                      <tr key={ind.id} className="hover:bg-slate-900/40">
                        <td className="py-3 pr-3">
                          <div className="font-medium text-slate-100">
                            {ind.name}
                          </div>
                          <div className="mt-0.5 text-[11px] text-slate-400">
                            {ind.summary}
                          </div>
                        </td>
                        <td className="py-3 text-right font-mono tabular-nums text-blue-300 whitespace-nowrap">
                          {ind.value} {ind.unit}
                        </td>
                        <td className="py-3 text-right font-mono tabular-nums text-slate-400 whitespace-nowrap">
                          {ind.baselineRange}
                        </td>
                        <td className="py-3 text-right font-medium whitespace-nowrap">
                          <span
                            className={
                              ind.assessment === 'anomalous'
                                ? 'text-rose-400'
                                : ind.assessment === 'elevated'
                                  ? 'text-amber-300'
                                  : 'text-emerald-400'
                            }
                          >
                            {ind.assessment === 'anomalous'
                              ? '▲ Anomalous'
                              : ind.assessment === 'elevated'
                                ? '● Elevated'
                                : '✓ Nominal'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-slate-800 bg-slate-900/60 p-3.5 text-xs text-slate-300">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-400" />
                <span>{result.disclaimer}</span>
              </div>
            </div>

            <div className="lg:col-span-5">
              <RadialSpectrumChart bins={result.radialSpectrum} />
            </div>
          </div>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl p-10 text-center">
          <p className="text-sm text-slate-300">
            Upload a JPEG/PNG image above or click any quick test portrait to inspect real model inference, Grad-CAM attribution, and frequency-domain residuals.
          </p>
        </div>
      )}
    </div>
  );
}
