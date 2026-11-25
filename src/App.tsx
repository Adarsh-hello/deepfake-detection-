import { useCallback, useEffect, useState } from 'react';
import { AnalyzeView } from './components/AnalyzeView.tsx';
import { BenchmarksView } from './components/BenchmarksView.tsx';
import { HistoryView } from './components/HistoryView.tsx';
import { Navbar, type NavTab } from './components/Navbar.tsx';
import { OverviewView } from './components/OverviewView.tsx';
import type {
  BenchmarkSuiteResult,
  DatasetStats,
  DetectionHistoryEntry,
  ForensicAnalysisResult,
  ModelId,
  SampleGalleryItem,
  TrainingProgressState,
} from './types.ts';

const INITIAL_TRAINING_STATE: TrainingProgressState = {
  isTraining: false,
  currentModelId: null,
  currentModelName: null,
  completedModels: 0,
  totalModels: 4,
  currentEpoch: 0,
  totalEpochs: 0,
  stage: 'Idle',
  lastError: null,
  startedAt: null,
  finishedAt: null,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [benchmarks, setBenchmarks] = useState<BenchmarkSuiteResult | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [training, setTraining] = useState<TrainingProgressState>(INITIAL_TRAINING_STATE);
  const [samples, setSamples] = useState<SampleGalleryItem[]>([]);
  const [history, setHistory] = useState<DetectionHistoryEntry[]>([]);

  const [selectedModelId, setSelectedModelId] = useState<ModelId | undefined>(undefined);
  const [analysisResult, setAnalysisResult] = useState<ForensicAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const fetchAllState = useCallback(async () => {
    try {
      const [benchRes, statsRes, samplesRes, histRes] = await Promise.all([
        fetch('/api/models/benchmarks'),
        fetch('/api/dataset/stats'),
        fetch('/api/samples'),
        fetch('/api/history'),
      ]);
      if (benchRes.ok) {
        const data = await benchRes.json();
        setBenchmarks(data.benchmarks ?? null);
        if (data.training) setTraining(data.training);
      }
      if (statsRes.ok) {
        const data = await statsRes.json();
        setDatasetStats(data.stats ?? null);
      }
      if (samplesRes.ok) {
        const data = await samplesRes.json();
        setSamples(data.samples ?? []);
      }
      if (histRes.ok) {
        const data = await histRes.json();
        setHistory(data.history ?? []);
      }
    } catch {
      // Transient startup errors handled on next poll
    }
  }, []);

  useEffect(() => {
    void fetchAllState();
  }, [fetchAllState]);

  // Poll training status when benchmarks aren't ready yet or training is running
  useEffect(() => {
    if (benchmarks && !training.isTraining) return;
    const timer = window.setInterval(async () => {
      try {
        const res = await fetch('/api/training/status');
        if (!res.ok) return;
        const data = await res.json();
        if (data.training) setTraining(data.training);
        if (data.benchmarks) {
          setBenchmarks(data.benchmarks);
          setDatasetStats(data.benchmarks.datasetStats);
        }
      } catch {
        // ignore transient poll error
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [benchmarks, training.isTraining]);

  const handleAnalyzeFile = async (file: File, modelId?: ModelId) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setActiveTab('analyze');
    try {
      const form = new FormData();
      form.append('image', file);
      if (modelId) form.append('modelId', modelId);

      const res = await fetch('/api/detect', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error ?? 'Failed to analyze uploaded image.');
      }
      setAnalysisResult(data.result);
      const histRes = await fetch('/api/history');
      if (histRes.ok) {
        const hData = await histRes.json();
        setHistory(hData.history ?? []);
      }
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : 'Analysis failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeSample = async (filename: string, modelId?: ModelId) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setActiveTab('analyze');
    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sampleFilename: filename, modelId }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error ?? 'Failed to analyze sample image.');
      }
      setAnalysisResult(data.result);
      const histRes = await fetch('/api/history');
      if (histRes.ok) {
        const hData = await histRes.json();
        setHistory(hData.history ?? []);
      }
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : 'Analysis failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleStartTraining = async () => {
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.training) setTraining(data.training);
    } catch {
      // ignore
    }
  };

  const handleSelectActiveModel = async (modelId: ModelId) => {
    const res = await fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId }),
    });
    if (res.ok) {
      const data = await res.json();
      setBenchmarks(data.benchmarks);
      setSelectedModelId(modelId);
    }
  };

  const handleUploadDatasetSamples = async (
    files: FileList,
    label: 'real' | 'fake'
  ): Promise<string> => {
    const form = new FormData();
    form.append('label', label);
    for (let i = 0; i < files.length; i++) {
      form.append('images', files[i]);
    }
    const res = await fetch('/api/dataset/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error ?? 'Failed to upload dataset images.');
    }
    if (data.stats) setDatasetStats(data.stats);
    const addedCount = (data.results ?? []).filter((r: { added: boolean }) => r.added).length;
    const rejectedCount = (data.results ?? []).length - addedCount;
    return `Ingested ${addedCount} valid ${label.toUpperCase()} image(s)${
      rejectedCount > 0 ? ` (${rejectedCount} duplicate/corrupted skipped)` : ''
    }. Click "Retrain & Benchmark All Models" to update model weights.`;
  };

  const handleAuditDataset = async () => {
    const res = await fetch('/api/dataset/prepare', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      setDatasetStats(data.stats);
    }
  };

  const handleDeleteHistory = async (id: string) => {
    const res = await fetch(`/api/history/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (res.ok) {
      const data = await res.json();
      setHistory(data.history ?? []);
    }
  };

  return (
    <div className="min-h-screen bg-[#050814] ambient-backdrop text-slate-100 flex flex-col">
      <Navbar
        activeTab={activeTab}
        onNavigate={setActiveTab}
        onPrimaryAction={() => setActiveTab('analyze')}
      />

      <main className="mx-auto w-full max-w-[1380px] flex-1 px-6 py-8">
        {activeTab === 'overview' && (
          <OverviewView
            benchmarks={benchmarks}
            training={training}
            samples={samples}
            history={history}
            onGoAnalyze={() => setActiveTab('analyze')}
            onGoBenchmarks={() => setActiveTab('benchmarks')}
            onGoHistory={() => setActiveTab('history')}
            onAnalyzeSample={(fname) => void handleAnalyzeSample(fname, selectedModelId)}
          />
        )}

        {activeTab === 'analyze' && (
          <AnalyzeView
            benchmarks={benchmarks}
            samples={samples}
            result={analysisResult}
            isAnalyzing={isAnalyzing}
            error={analysisError}
            selectedModelId={selectedModelId}
            onSelectModelId={(id) => {
              setSelectedModelId(id);
              void handleSelectActiveModel(id);
            }}
            onAnalyzeFile={(f, mId) => void handleAnalyzeFile(f, mId)}
            onAnalyzeSample={(fname, mId) => void handleAnalyzeSample(fname, mId)}
          />
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarksView
            benchmarks={benchmarks}
            datasetStats={datasetStats}
            training={training}
            onStartTraining={() => void handleStartTraining()}
            onSelectActiveModel={(id) => void handleSelectActiveModel(id)}
            onUploadDatasetSamples={handleUploadDatasetSamples}
            onAuditDataset={() => void handleAuditDataset()}
          />
        )}

        {activeTab === 'history' && (
          <HistoryView
            history={history}
            onInspectReport={(report) => {
              setAnalysisResult(report);
              setActiveTab('analyze');
            }}
            onDeleteEntry={(id) => void handleDeleteHistory(id)}
            onGoAnalyze={() => setActiveTab('analyze')}
          />
        )}
      </main>

      <footer className="border-t border-slate-800/60 py-5 text-center text-xs text-slate-500">
        SpectraTrace Deepfake Forensics Platform · Multi-Architecture ML &amp; Supporting Spectral Diagnostics
      </footer>
    </div>
  );
}
