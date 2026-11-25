import { useState } from 'react';
import type {
  ConfusionMatrixData,
  EpochMetric,
  ModelEvaluation,
} from '../types.ts';

const MODEL_COLORS: Record<string, string> = {
  cnn_baseline: '#60a5fa',
  resnet_lite: '#a78bfa',
  xception_mobilenet: '#38bdf8',
  hybrid_sfnet: '#c084fc',
};

export function ConfusionMatrixChart({
  matrix,
  modelName,
}: {
  matrix: ConfusionMatrixData;
  modelName: string;
}) {
  const total = Math.max(1, matrix.tn + matrix.fp + matrix.fn + matrix.tp);
  const cells = [
    {
      label: 'True Negative (TN)',
      sub: 'Actual REAL · Pred REAL',
      count: matrix.tn,
      pct: (matrix.tn / total) * 100,
      tone: 'bg-emerald-500/15 border-emerald-500/35 text-emerald-300',
    },
    {
      label: 'False Positive (FP)',
      sub: 'Actual REAL · Pred FAKE',
      count: matrix.fp,
      pct: (matrix.fp / total) * 100,
      tone: 'bg-rose-500/12 border-rose-500/30 text-rose-300',
    },
    {
      label: 'False Negative (FN)',
      sub: 'Actual FAKE · Pred REAL',
      count: matrix.fn,
      pct: (matrix.fn / total) * 100,
      tone: 'bg-amber-500/12 border-amber-500/30 text-amber-300',
    },
    {
      label: 'True Positive (TP)',
      sub: 'Actual FAKE · Pred FAKE',
      count: matrix.tp,
      pct: (matrix.tp / total) * 100,
      tone: 'bg-blue-500/15 border-blue-500/35 text-blue-300',
    },
  ];

  return (
    <div className="glass-panel rounded-xl p-5">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <div>
          <h3 className="font-display text-base font-semibold text-white">
            Confusion Matrix
          </h3>
          <p className="text-xs text-slate-400">
            Held-out test split · {modelName} · n={total}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {cells.map((cell) => (
          <div
            key={cell.label}
            className={`rounded-lg border p-4 transition-colors ${cell.tone}`}
          >
            <div className="text-xs font-medium text-slate-300">{cell.label}</div>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="font-mono text-2xl font-bold tabular-nums text-white">
                {cell.count}
              </span>
              <span className="font-mono text-xs tabular-nums">
                {cell.pct.toFixed(1)}%
              </span>
            </div>
            <div className="mt-1.5 text-[11px] text-slate-400">{cell.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function RocCurveChart({ models }: { models: ModelEvaluation[] }) {
  const w = 440;
  const h = 280;
  const pad = { top: 18, right: 18, bottom: 38, left: 44 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  return (
    <div className="glass-panel rounded-xl p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="font-display text-base font-semibold text-white">
            Multi-Model ROC Curve
          </h3>
          <p className="text-xs text-slate-400">
            True Positive Rate vs False Positive Rate on held-out test split
          </p>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full h-auto"
        role="img"
        aria-label="ROC curve comparing trained models"
      >
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const x = pad.left + t * plotW;
          const y = pad.top + (1 - t) * plotH;
          return (
            <g key={t}>
              <line
                x1={pad.left}
                y1={y}
                x2={w - pad.right}
                y2={y}
                stroke="rgba(148,163,184,0.12)"
                strokeWidth="1"
              />
              <line
                x1={x}
                y1={pad.top}
                x2={x}
                y2={h - pad.bottom}
                stroke="rgba(148,163,184,0.12)"
                strokeWidth="1"
              />
              <text
                x={pad.left - 8}
                y={y + 3}
                textAnchor="end"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {t.toFixed(2)}
              </text>
              <text
                x={x}
                y={h - pad.bottom + 14}
                textAnchor="middle"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {t.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Random guess diagonal */}
        <line
          x1={pad.left}
          y1={h - pad.bottom}
          x2={w - pad.right}
          y2={pad.top}
          stroke="rgba(148,163,184,0.35)"
          strokeDasharray="4 4"
          strokeWidth="1"
        />

        {/* Model ROC polylines */}
        {models.map((m) => {
          const pts = m.rocCurve
            .map((p) => {
              const x = pad.left + p.fpr * plotW;
              const y = pad.top + (1 - p.tpr) * plotH;
              return `${x.toFixed(1)},${y.toFixed(1)}`;
            })
            .join(' ');
          const color = MODEL_COLORS[m.modelId] ?? '#60a5fa';
          return (
            <polyline
              key={m.modelId}
              fill="none"
              stroke={color}
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={pts}
            />
          );
        })}

        <text
          x={pad.left + plotW / 2}
          y={h - 6}
          textAnchor="middle"
          className="fill-slate-400 text-[11px]"
        >
          False Positive Rate
        </text>
        <text
          x={12}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${pad.top + plotH / 2})`}
          className="fill-slate-400 text-[11px]"
        >
          True Positive Rate
        </text>
      </svg>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-xs">
        {models.map((m) => (
          <div key={m.modelId} className="flex items-center gap-2 text-slate-300">
            <span
              className="h-2.5 w-2.5 rounded-xs"
              style={{ backgroundColor: MODEL_COLORS[m.modelId] ?? '#60a5fa' }}
            />
            <span>{m.displayName}</span>
            <span className="font-mono text-slate-400 tabular-nums">
              AUC {m.rocAuc.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TrainingCurvesChart({
  history,
  modelName,
}: {
  history: EpochMetric[];
  modelName: string;
}) {
  const [metricMode, setMetricMode] = useState<'accuracy' | 'loss'>('accuracy');
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const w = 480;
  const h = 250;
  const pad = { top: 18, right: 18, bottom: 34, left: 44 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const trainVals = history.map((e) =>
    metricMode === 'accuracy' ? e.trainAccuracy : e.trainLoss
  );
  const valVals = history.map((e) =>
    metricMode === 'accuracy' ? e.valAccuracy : e.valLoss
  );
  const maxVal =
    metricMode === 'accuracy'
      ? 1.0
      : Math.max(0.5, ...trainVals, ...valVals) * 1.1;
  const minVal = 0;

  const toCoords = (vals: number[]) =>
    vals
      .map((v, i) => {
        const x =
          pad.left +
          (vals.length <= 1 ? plotW / 2 : (i / (vals.length - 1)) * plotW);
        const y =
          pad.top + (1 - (v - minVal) / Math.max(1e-5, maxVal - minVal)) * plotH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  const hovered = hoverIdx !== null ? history[hoverIdx] : null;

  return (
    <div className="glass-panel rounded-xl p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-display text-base font-semibold text-white">
            Training &amp; Validation Curves
          </h3>
          <p className="text-xs text-slate-400">
            {modelName} · {history.length} epochs recorded
          </p>
        </div>

        <div className="flex items-center gap-1 rounded-lg bg-slate-900/90 p-1 border border-slate-800">
          <button
            type="button"
            onClick={() => setMetricMode('accuracy')}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              metricMode === 'accuracy'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Accuracy
          </button>
          <button
            type="button"
            onClick={() => setMetricMode('loss')}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              metricMode === 'loss'
                ? 'bg-violet-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Loss
          </button>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const val = minVal + frac * (maxVal - minVal);
          const y = pad.top + (1 - frac) * plotH;
          return (
            <g key={frac}>
              <line
                x1={pad.left}
                y1={y}
                x2={w - pad.right}
                y2={y}
                stroke="rgba(148,163,184,0.12)"
              />
              <text
                x={pad.left - 8}
                y={y + 3}
                textAnchor="end"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {metricMode === 'accuracy'
                  ? `${Math.round(val * 100)}%`
                  : val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {history.length > 0 && (
          <>
            <polyline
              fill="none"
              stroke="#38bdf8"
              strokeWidth="2.2"
              points={toCoords(trainVals)}
            />
            <polyline
              fill="none"
              stroke="#a855f7"
              strokeWidth="2.2"
              strokeDasharray="5 3"
              points={toCoords(valVals)}
            />
            {history.map((ep, i) => {
              const x =
                pad.left +
                (history.length <= 1
                  ? plotW / 2
                  : (i / (history.length - 1)) * plotW);
              return (
                <g key={ep.epoch}>
                  <rect
                    x={x - 10}
                    y={pad.top}
                    width={20}
                    height={plotH}
                    fill="transparent"
                    onMouseEnter={() => setHoverIdx(i)}
                    onMouseLeave={() => setHoverIdx(null)}
                  />
                  {i % Math.max(1, Math.floor(history.length / 6)) === 0 && (
                    <text
                      x={x}
                      y={h - pad.bottom + 15}
                      textAnchor="middle"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      E{ep.epoch}
                    </text>
                  )}
                </g>
              );
            })}
          </>
        )}
      </svg>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-4 text-slate-300">
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 bg-sky-400 inline-block" />
            Train {metricMode === 'accuracy' ? 'Accuracy' : 'Loss'}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 border-t-2 border-dashed border-purple-400 inline-block" />
            Val {metricMode === 'accuracy' ? 'Accuracy' : 'Loss'}
          </span>
        </div>
        {hovered ? (
          <div className="font-mono text-slate-300 tabular-nums">
            Epoch {hovered.epoch} · Train:{' '}
            {metricMode === 'accuracy'
              ? `${(hovered.trainAccuracy * 100).toFixed(1)}%`
              : hovered.trainLoss.toFixed(3)}{' '}
            · Val:{' '}
            {metricMode === 'accuracy'
              ? `${(hovered.valAccuracy * 100).toFixed(1)}%`
              : hovered.valLoss.toFixed(3)}
          </div>
        ) : (
          <div className="text-slate-500">Hover chart for epoch telemetry</div>
        )}
      </div>
    </div>
  );
}

export function RadialSpectrumChart({
  bins,
}: {
  bins: Array<{ band: string; power: number }>;
}) {
  const maxPower = Math.max(1, ...bins.map((b) => b.power));

  return (
    <div className="glass-panel rounded-xl p-5">
      <div className="mb-3">
        <h3 className="font-display text-base font-semibold text-white">
          Azimuthal 2D-FFT Radial Power Profile
        </h3>
        <p className="text-xs text-slate-400">
          Log-magnitude spectral energy from DC center (F1) to Nyquist edge (F12)
        </p>
      </div>

      <div className="flex h-36 items-end gap-2 pt-3">
        {bins.map((b) => {
          const pct = Math.max(4, (b.power / maxPower) * 100);
          return (
            <div key={b.band} className="flex flex-1 flex-col items-center gap-1.5 h-full justify-end">
              <div className="w-full flex-1 flex items-end">
                <div
                  className="w-full rounded-t-sm bg-gradient-to-t from-blue-600/80 to-violet-400/90 transition-all"
                  style={{ height: `${pct}%` }}
                  title={`${b.band}: ${b.power.toFixed(3)}`}
                />
              </div>
              <span className="font-mono text-[10px] text-slate-400 tabular-nums">
                {b.band}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
