import { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Download,
  Eye,
  Search,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import type { DetectionHistoryEntry, ForensicAnalysisResult } from '../types.ts';
import { downloadHtmlReport } from '../utils/reportExport.ts';

interface HistoryViewProps {
  history: DetectionHistoryEntry[];
  onInspectReport: (report: ForensicAnalysisResult) => void;
  onDeleteEntry: (id: string) => void;
  onGoAnalyze: () => void;
}

export function HistoryView({
  history,
  onInspectReport,
  onDeleteEntry,
  onGoAnalyze,
}: HistoryViewProps) {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<'ALL' | 'REAL' | 'FAKE'>('ALL');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return history.filter((entry) => {
      if (filter !== 'ALL' && entry.prediction !== filter) return false;
      if (!q) return true;
      return (
        entry.filename.toLowerCase().includes(q) ||
        entry.modelDisplayName.toLowerCase().includes(q) ||
        entry.sha256.toLowerCase().includes(q)
      );
    });
  }, [history, query, filter]);

  return (
    <div className="space-y-6">
      <div className="glass-panel rounded-2xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="font-display text-2xl font-bold text-white">
              Detection History
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Audit log of evaluated images with confidence scores, active models, and downloadable forensic reports
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search filename or SHA-256..."
                className="w-60 rounded-lg border border-slate-700 bg-slate-900/90 py-1.5 pl-8 pr-3 text-xs text-white placeholder:text-slate-500 focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-900/90 p-1">
              {(['ALL', 'REAL', 'FAKE'] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setFilter(tab)}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    filter === tab
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {tab === 'ALL' ? 'All' : tab}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="glass-panel rounded-2xl p-12 text-center">
          <h2 className="font-display text-base font-semibold text-white">
            {history.length === 0
              ? 'No images analyzed yet'
              : 'No detections match your filter'}
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            {history.length === 0
              ? 'Upload an image or run a quick test portrait to record your first forensic detection.'
              : 'Clear your search query or switch the verdict filter.'}
          </p>
          {history.length === 0 && (
            <button
              type="button"
              onClick={onGoAnalyze}
              className="mt-4 rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-4 py-2 text-xs font-semibold text-white"
            >
              Analyze First Image
            </button>
          )}
        </div>
      ) : (
        <div className="glass-panel overflow-hidden rounded-2xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/40 text-slate-400">
                  <th className="px-5 py-3.5 font-medium">Sample</th>
                  <th className="px-3 py-3.5 font-medium">Model</th>
                  <th className="px-3 py-3.5 font-medium">Verdict</th>
                  <th className="px-3 py-3.5 text-right font-medium">Confidence</th>
                  <th className="px-3 py-3.5 text-right font-medium">Latency</th>
                  <th className="px-3 py-3.5 font-medium">Analyzed</th>
                  <th className="px-5 py-3.5 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {filtered.map((entry) => (
                  <tr
                    key={entry.id}
                    className="transition-colors hover:bg-slate-900/45"
                  >
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-3">
                        <img
                          src={entry.thumbnailDataUrl}
                          alt={entry.filename}
                          className="h-10 w-10 rounded-lg border border-slate-700 object-cover shrink-0"
                        />
                        <div className="min-w-0">
                          <div className="max-w-[220px] truncate font-medium text-white">
                            {entry.filename}
                          </div>
                          <div className="font-mono text-[11px] text-slate-500">
                            {entry.sha256.slice(0, 12)}…
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-3 text-slate-300">
                      {entry.modelDisplayName}
                    </td>
                    <td className="px-3 py-3">
                      <span
                        className={`inline-flex items-center gap-1.5 font-semibold ${
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
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right font-mono tabular-nums text-white">
                      {entry.confidence.toFixed(2)}%
                    </td>
                    <td className="px-3 py-3 text-right font-mono tabular-nums text-slate-400">
                      {entry.inferenceTimeMs.toFixed(2)} ms
                    </td>
                    <td className="px-3 py-3 font-mono text-[11px] tabular-nums text-slate-400 whitespace-nowrap">
                      {new Date(entry.analyzedAt).toLocaleString()}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <div className="inline-flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => onInspectReport(entry.report)}
                          title="Inspect forensic report"
                          className="rounded-md border border-slate-700 bg-slate-900 p-1.5 text-slate-300 hover:border-blue-500 hover:text-white"
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => downloadHtmlReport(entry.report)}
                          title="Download HTML report"
                          className="rounded-md border border-slate-700 bg-slate-900 p-1.5 text-slate-300 hover:border-blue-500 hover:text-white"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onDeleteEntry(entry.id)}
                          title="Delete entry"
                          className="rounded-md border border-slate-700 bg-slate-900 p-1.5 text-slate-400 hover:border-rose-500 hover:text-rose-300"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
