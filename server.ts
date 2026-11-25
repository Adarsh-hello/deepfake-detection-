import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import multer from 'multer';
import { createServer as createViteServer } from 'vite';
import { z } from 'zod';
import {
  addUserDatasetSample,
  getDatasetStats,
  prepareAndAuditDataset,
  seedStarterDatasetIfNeeded,
} from './server/ml/dataset.ts';
import {
  analyzeImageBuffer,
  clearModelCache,
  deleteHistoryEntry,
  loadHistory,
} from './server/ml/forensics.ts';
import { MODEL_BLUEPRINTS } from './server/ml/models.ts';
import {
  getTrainingProgress,
  loadBenchmarks,
  setActiveModelId,
  startBackgroundTrainingJob,
} from './server/ml/trainer.ts';
import type { ModelId } from './server/ml/types.ts';

const PORT = 3000;
const isProd = process.env.NODE_ENV === 'production';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
});

const SAMPLE_GALLERY = [
  {
    id: 'nat_a',
    title: 'Outdoor Natural Portrait',
    filename: 'sample_portrait_natural_a_1795625579546.jpg',
    url: '/src/assets/images/sample_portrait_natural_a_1795625579546.jpg',
    category: 'Unretouched Camera',
    notes: 'Overcast daylight, organic skin pores, 35mm optical grain',
  },
  {
    id: 'nat_b',
    title: 'Window Light Editorial',
    filename: 'sample_portrait_natural_b_1795625580963.jpg',
    url: '/src/assets/images/sample_portrait_natural_b_1795625580963.jpg',
    category: 'Unretouched Camera',
    notes: 'DSLR 85mm prime, natural hair flyaways, fine texture',
  },
  {
    id: 'nat_c',
    title: 'Library Archival Portrait',
    filename: 'sample_portrait_natural_c_1795625580852.jpg',
    url: '/src/assets/images/sample_portrait_natural_c_1795625580852.jpg',
    category: 'Unretouched Camera',
    notes: 'Held-out test identity, natural specular highlights & creases',
  },
  {
    id: 'syn_a',
    title: 'Airbrushed Studio Synthesis',
    filename: 'sample_portrait_synthetic_a_1795625580022.jpg',
    url: '/src/assets/images/sample_portrait_synthetic_a_1795625580022.jpg',
    category: 'Synthetic / Retouched',
    notes: 'Uniform skin sheen, symmetric facial alignment, studio rim glow',
  },
  {
    id: 'syn_b',
    title: 'Cyberpunk Neural Render',
    filename: 'sample_portrait_synthetic_b_1795625580752.jpg',
    url: '/src/assets/images/sample_portrait_synthetic_b_1795625580752.jpg',
    category: 'Synthetic / Retouched',
    notes: 'Poreless render texture, chromatic neon highlights, bokeh anomaly',
  },
  {
    id: 'syn_c',
    title: 'Mismatched Lighting Composite',
    filename: 'sample_portrait_synthetic_c_1795625581640.jpg',
    url: '/src/assets/images/sample_portrait_synthetic_c_1795625581640.jpg',
    category: 'Synthetic / Retouched',
    notes: 'Held-out test identity, asymmetric illumination & hair boundary smoothing',
  },
];

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  // Seed starter dataset on disk immediately
  seedStarterDatasetIfNeeded();

  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      benchmarksReady: Boolean(loadBenchmarks()),
      training: getTrainingProgress(),
    });
  });

  app.get('/api/samples', (_req, res) => {
    res.json({ samples: SAMPLE_GALLERY });
  });

  app.get('/api/dataset/stats', (_req, res) => {
    try {
      res.json({ stats: getDatasetStats() });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Failed to load stats' });
    }
  });

  app.post('/api/dataset/prepare', (_req, res) => {
    try {
      const { stats } = prepareAndAuditDataset();
      res.json({ stats });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Audit failed' });
    }
  });

  app.post('/api/dataset/upload', upload.array('images', 20), (req, res) => {
    try {
      const labelParsed = z.enum(['real', 'fake']).safeParse(req.body?.label);
      if (!labelParsed.success) {
        return res.status(400).json({ error: 'Valid label ("real" or "fake") is required.' });
      }
      const splitParsed = z.enum(['train', 'val', 'test']).optional().safeParse(req.body?.split);
      const files = (req.files as Express.Multer.File[]) ?? [];
      if (files.length === 0) {
        return res.status(400).json({ error: 'No image files provided.' });
      }

      const results = files.map((file) => ({
        filename: file.originalname,
        ...addUserDatasetSample(
          file.buffer,
          file.originalname,
          labelParsed.data,
          splitParsed.success ? splitParsed.data : undefined
        ),
      }));

      const stats = getDatasetStats();
      res.json({ results, stats });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Dataset upload failed' });
    }
  });

  app.get('/api/models/benchmarks', (_req, res) => {
    const benchmarks = loadBenchmarks();
    res.json({
      benchmarks,
      blueprints: MODEL_BLUEPRINTS,
      training: getTrainingProgress(),
    });
  });

  app.post('/api/models/select', (req, res) => {
    const schema = z.object({
      modelId: z.enum(['cnn_baseline', 'resnet_lite', 'xception_mobilenet', 'hybrid_sfnet']),
    });
    const parsed = schema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid modelId.' });
    }
    try {
      const updated = setActiveModelId(parsed.data.modelId);
      res.json({ benchmarks: updated });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Model selection failed' });
    }
  });

  app.get('/api/training/status', (_req, res) => {
    res.json({
      training: getTrainingProgress(),
      benchmarks: loadBenchmarks(),
    });
  });

  app.post('/api/training/start', (req, res) => {
    const progress = getTrainingProgress();
    if (progress.isTraining) {
      return res.status(409).json({ error: 'Training is already in progress.', training: progress });
    }
    const schema = z.object({
      modelIds: z
        .array(z.enum(['cnn_baseline', 'resnet_lite', 'xception_mobilenet', 'hybrid_sfnet']))
        .optional(),
    });
    const parsed = schema.safeParse(req.body ?? {});
    const modelIds = parsed.success ? (parsed.data.modelIds as ModelId[] | undefined) : undefined;

    clearModelCache();
    const training = startBackgroundTrainingJob(modelIds);
    res.json({ started: true, training });
  });

  app.post('/api/detect', upload.single('image'), async (req, res) => {
    try {
      const requestedModelId = z
        .enum(['cnn_baseline', 'resnet_lite', 'xception_mobilenet', 'hybrid_sfnet'])
        .optional()
        .safeParse(req.body?.modelId);

      let buffer: Buffer | null = null;
      let filename = 'uploaded_image.jpg';

      if (req.file) {
        buffer = req.file.buffer;
        filename = req.file.originalname;
      } else if (typeof req.body?.sampleFilename === 'string') {
        const safeName = path.basename(req.body.sampleFilename);
        const samplePath = path.resolve(process.cwd(), 'src', 'assets', 'images', safeName);
        if (!fs.existsSync(samplePath)) {
          return res.status(404).json({ error: 'Sample image not found.' });
        }
        buffer = fs.readFileSync(samplePath);
        filename = safeName;
      }

      if (!buffer) {
        return res.status(400).json({ error: 'Please provide an image file or sampleFilename.' });
      }

      const result = await analyzeImageBuffer(
        buffer,
        filename,
        requestedModelId.success ? requestedModelId.data : undefined
      );
      res.json({ result });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Failed to analyze image.',
      });
    }
  });

  app.get('/api/history', (_req, res) => {
    res.json({ history: loadHistory() });
  });

  app.delete('/api/history/:id', (req, res) => {
    const history = deleteHistoryEntry(req.params.id);
    res.json({ history });
  });

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[SpectraTrace] Server running at http://0.0.0.0:${PORT}`);
    if (!loadBenchmarks()) {
      startBackgroundTrainingJob();
    }
  });
}

startServer().catch((err) => {
  console.error('[SpectraTrace] Fatal server error:', err);
  process.exit(1);
});
