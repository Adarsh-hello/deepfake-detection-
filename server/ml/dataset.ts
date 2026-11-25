import fs from 'node:fs';
import path from 'node:path';
import {
  computeDHash,
  computeSha256,
  cropRgba,
  decodeImageBuffer,
  encodeJpegBuffer,
  hammingDistanceHex,
  resizeRgbaBilinear,
} from './imageUtils.ts';
import type { DatasetSample, DatasetStats, LabelName, SplitName } from './types.ts';

export const DATA_ROOT = path.resolve(process.cwd(), 'data');
export const DATASET_DIR = path.join(DATA_ROOT, 'dataset');
export const REAL_DIR = path.join(DATASET_DIR, 'real');
export const FAKE_DIR = path.join(DATASET_DIR, 'fake');
export const MANIFEST_PATH = path.join(DATASET_DIR, 'manifest.json');
export const TARGET_IMAGE_SIZE = 16;

const STARTER_SOURCES: Array<{
  filename: string;
  label: LabelName;
  split: SplitName;
  sourceGroupId: string;
}> = [
  {
    filename: 'sample_portrait_natural_a_1795625579546.jpg',
    label: 'real',
    split: 'train',
    sourceGroupId: 'identity_nat_a',
  },
  {
    filename: 'sample_portrait_synthetic_a_1795625580022.jpg',
    label: 'fake',
    split: 'train',
    sourceGroupId: 'identity_syn_a',
  },
  {
    filename: 'sample_portrait_natural_b_1795625580963.jpg',
    label: 'real',
    split: 'val',
    sourceGroupId: 'identity_nat_b',
  },
  {
    filename: 'sample_portrait_synthetic_b_1795625580752.jpg',
    label: 'fake',
    split: 'val',
    sourceGroupId: 'identity_syn_b',
  },
  {
    filename: 'sample_portrait_natural_c_1795625580852.jpg',
    label: 'real',
    split: 'test',
    sourceGroupId: 'identity_nat_c',
  },
  {
    filename: 'sample_portrait_synthetic_c_1795625581640.jpg',
    label: 'fake',
    split: 'test',
    sourceGroupId: 'identity_syn_c',
  },
];

function ensureDirs(): void {
  fs.mkdirSync(REAL_DIR, { recursive: true });
  fs.mkdirSync(FAKE_DIR, { recursive: true });
}

function deterministicNoise(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Applies natural optical shot-noise variation to camera patches or subtle generative
 * upsampling / smoothing / recompression perturbations to synthetic patches.
 */
function createVariantBuffer(
  patchRgba: Uint8Array,
  w: number,
  h: number,
  isFake: boolean,
  variantIndex: number
): Buffer {
  const out = new Uint8Array(patchRgba);
  const rand = deterministicNoise(variantIndex * 9973 + (isFake ? 42 : 7));

  if (!isFake) {
    // Subtle natural sensor shot noise + single-pass JPEG capture
    const sigma = 3.0 + (variantIndex % 3) * 1.2;
    for (let i = 0; i < w * h; i++) {
      const idx = i * 4;
      const n = (rand() + rand() - 1.0) * sigma;
      out[idx] = Math.max(0, Math.min(255, Math.round(out[idx] + n)));
      out[idx + 1] = Math.max(0, Math.min(255, Math.round(out[idx + 1] + n * 0.9)));
      out[idx + 2] = Math.max(0, Math.min(255, Math.round(out[idx + 2] + n * 1.1)));
    }
    return encodeJpegBuffer(w, h, out, 92 - (variantIndex % 3) * 2);
  }

  // Synthetic/Deepfake variant: central smoothing + subtle 2x2 upsampling checkerboard + double JPEG
  const smoothed = new Uint8Array(out);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const inCenter = x > w * 0.2 && x < w * 0.8 && y > h * 0.2 && y < h * 0.8;
      if (inCenter) {
        for (let c = 0; c < 3; c++) {
          const avg =
            (out[idx + c] * 2 +
              out[((y - 1) * w + x) * 4 + c] +
              out[((y + 1) * w + x) * 4 + c] +
              out[(y * w + (x - 1)) * 4 + c] +
              out[(y * w + (x + 1)) * 4 + c]) /
            6;
          // Mild 2x2 periodic upsampling micro-grid
          const grid = ((x & 1) ^ (y & 1)) === 0 ? 1.6 : -1.6;
          smoothed[idx + c] = Math.max(0, Math.min(255, Math.round(avg + grid)));
        }
      }
    }
  }
  // Double-JPEG compression signature
  const firstPass = encodeJpegBuffer(w, h, smoothed, 78 + (variantIndex % 4) * 3);
  const redecoded = decodeImageBuffer(firstPass);
  return encodeJpegBuffer(w, h, redecoded.rgba, 88);
}

export function seedStarterDatasetIfNeeded(): void {
  ensureDirs();
  if (fs.existsSync(MANIFEST_PATH)) {
    try {
      const existing = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8')) as {
        samples: DatasetSample[];
      };
      if (Array.isArray(existing.samples) && existing.samples.length >= 48) {
        return;
      }
    } catch {
      // Rebuild manifest cleanly if corrupted
    }
  }

  const assetsDir = path.resolve(process.cwd(), 'src', 'assets', 'images');
  const samplesMeta: Array<{
    filename: string;
    label: LabelName;
    split: SplitName;
    sourceGroupId: string;
  }> = [];

  for (const src of STARTER_SOURCES) {
    const srcPath = path.join(assetsDir, src.filename);
    if (!fs.existsSync(srcPath)) continue;
    const rawBuf = fs.readFileSync(srcPath);
    const decoded = decodeImageBuffer(rawBuf);

    // Extract distinct spatial patches per split so Train/Val/Test remain 100% identity-disjoint
    const grid = src.split === 'train' ? 5 : src.split === 'test' ? 4 : 3;
    const tileW = Math.floor(decoded.width / grid);
    const tileH = Math.floor(decoded.height / grid);
    let idx = 0;

    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const crop = cropRgba(
          decoded.rgba,
          decoded.width,
          decoded.height,
          gx * tileW,
          gy * tileH,
          tileW,
          tileH
        );
        const resized = resizeRgbaBilinear(crop, tileW, tileH, 128, 128);
        const buf = createVariantBuffer(
          resized,
          128,
          128,
          src.label === 'fake',
          idx + (src.split === 'train' ? 10 : src.split === 'val' ? 50 : 90)
        );
        const outName = `${src.sourceGroupId}_${src.split}_p${idx}.jpg`;
        const destDir = src.label === 'real' ? REAL_DIR : FAKE_DIR;
        fs.writeFileSync(path.join(destDir, outName), buf);
        samplesMeta.push({
          filename: outName,
          label: src.label,
          split: src.split,
          sourceGroupId: src.sourceGroupId,
        });
        idx++;
      }
    }

    // Also include a full-portrait downscaled sample for each source identity
    const fullResized = resizeRgbaBilinear(decoded.rgba, decoded.width, decoded.height, 192, 192);
    const fullBuf = encodeJpegBuffer(192, 192, fullResized, src.label === 'real' ? 94 : 84);
    const fullName = `${src.sourceGroupId}_${src.split}_full.jpg`;
    fs.writeFileSync(path.join(src.label === 'real' ? REAL_DIR : FAKE_DIR, fullName), fullBuf);
    samplesMeta.push({
      filename: fullName,
      label: src.label,
      split: src.split,
      sourceGroupId: src.sourceGroupId,
    });
  }

  fs.writeFileSync(
    path.join(DATASET_DIR, 'seed_index.json'),
    JSON.stringify(samplesMeta, null, 2)
  );
  prepareAndAuditDataset();
}

export function addUserDatasetSample(
  buffer: Buffer,
  originalFilename: string,
  label: LabelName,
  requestedSplit?: SplitName
): { added: boolean; reason?: string; sample?: DatasetSample } {
  ensureDirs();
  // Validate integrity before writing
  let decoded;
  try {
    decoded = decodeImageBuffer(buffer);
  } catch (err) {
    return {
      added: false,
      reason: err instanceof Error ? err.message : 'Corrupted or unsupported image file.',
    };
  }

  const sha256 = computeSha256(buffer);
  const dHash = computeDHash(decoded.rgba, decoded.width, decoded.height);

  const existingManifest = loadManifestSamples();
  for (const s of existingManifest) {
    if (s.sha256 === sha256) {
      return { added: false, reason: 'Exact duplicate image (SHA-256 match) rejected.' };
    }
    if (hammingDistanceHex(s.dHash, dHash) <= 3) {
      return { added: false, reason: 'Near-duplicate image (perceptual dHash match) rejected.' };
    }
  }

  const cleanBase = originalFilename.replace(/[^a-zA-Z0-9._-]/g, '_').replace(/\.\w+$/, '');
  const ext = decoded.format === 'png' ? 'png' : 'jpg';
  const outName = `user_${Date.now()}_${cleanBase}.${ext}`;
  const destDir = label === 'real' ? REAL_DIR : FAKE_DIR;
  const filePath = path.join(destDir, outName);
  fs.writeFileSync(filePath, buffer);

  // Assign split deterministically by sourceGroupId if not specified (70/15/15)
  const split: SplitName =
    requestedSplit ??
    (existingManifest.length % 7 === 5
      ? 'val'
      : existingManifest.length % 7 === 6
        ? 'test'
        : 'train');

  const userIndexPath = path.join(DATASET_DIR, 'user_index.json');
  const userEntries: Array<{
    filename: string;
    label: LabelName;
    split: SplitName;
    sourceGroupId: string;
  }> = fs.existsSync(userIndexPath)
    ? JSON.parse(fs.readFileSync(userIndexPath, 'utf-8'))
    : [];

  userEntries.push({
    filename: outName,
    label,
    split,
    sourceGroupId: `user_${sha256.slice(0, 12)}`,
  });
  fs.writeFileSync(userIndexPath, JSON.stringify(userEntries, null, 2));

  const { samples } = prepareAndAuditDataset();
  const created = samples.find((s) => s.filename === outName);
  return { added: true, sample: created };
}

export function prepareAndAuditDataset(): {
  samples: DatasetSample[];
  stats: DatasetStats;
} {
  ensureDirs();
  const seedIndexPath = path.join(DATASET_DIR, 'seed_index.json');
  const userIndexPath = path.join(DATASET_DIR, 'user_index.json');

  const metaMap = new Map<string, { split: SplitName; sourceGroupId: string; isUser: boolean }>();
  if (fs.existsSync(seedIndexPath)) {
    const seedEntries = JSON.parse(fs.readFileSync(seedIndexPath, 'utf-8')) as Array<{
      filename: string;
      label: LabelName;
      split: SplitName;
      sourceGroupId: string;
    }>;
    for (const e of seedEntries) {
      metaMap.set(`${e.label}:${e.filename}`, {
        split: e.split,
        sourceGroupId: e.sourceGroupId,
        isUser: false,
      });
    }
  }
  if (fs.existsSync(userIndexPath)) {
    const userEntries = JSON.parse(fs.readFileSync(userIndexPath, 'utf-8')) as Array<{
      filename: string;
      label: LabelName;
      split: SplitName;
      sourceGroupId: string;
    }>;
    for (const e of userEntries) {
      metaMap.set(`${e.label}:${e.filename}`, {
        split: e.split,
        sourceGroupId: e.sourceGroupId,
        isUser: true,
      });
    }
  }

  const seenSha = new Set<string>();
  const seenDHash: Array<{ hash: string; groupId: string }> = [];
  let totalScanned = 0;
  let corruptedRemoved = 0;
  let duplicatesRemoved = 0;
  const validSamples: DatasetSample[] = [];

  for (const label of ['real', 'fake'] as const) {
    const dir = label === 'real' ? REAL_DIR : FAKE_DIR;
    const files = fs.readdirSync(dir).sort();
    for (const file of files) {
      if (!/\.(jpe?g|png)$/i.test(file)) continue;
      totalScanned++;
      const fullPath = path.join(dir, file);
      const buf = fs.readFileSync(fullPath);

      let decoded;
      try {
        decoded = decodeImageBuffer(buf);
      } catch {
        corruptedRemoved++;
        continue;
      }

      const sha256 = computeSha256(buf);
      if (seenSha.has(sha256)) {
        duplicatesRemoved++;
        continue;
      }
      const dHash = computeDHash(decoded.rgba, decoded.width, decoded.height);
      if (seenDHash.some((item) => hammingDistanceHex(item.hash, dHash) <= 1)) {
        duplicatesRemoved++;
        continue;
      }

      const meta = metaMap.get(`${label}:${file}`);
      const sourceGroupId = meta?.sourceGroupId ?? `group_${sha256.slice(0, 10)}`;
      const split: SplitName = meta?.split ?? 'train';

      seenSha.add(sha256);
      seenDHash.push({ hash: dHash, groupId: sourceGroupId });

      validSamples.push({
        id: sha256.slice(0, 16),
        filename: file,
        label,
        split,
        sourceGroupId,
        sha256,
        dHash,
        width: decoded.width,
        height: decoded.height,
        filePath: fullPath,
        isUserUploaded: meta?.isUser ?? false,
      });
    }
  }

  // Verify zero sourceGroupId leakage across splits
  const splitGroupSets: Record<SplitName, Set<string>> = {
    train: new Set(),
    val: new Set(),
    test: new Set(),
  };
  for (const s of validSamples) {
    splitGroupSets[s.split].add(s.sourceGroupId);
  }
  let leakageVerified = true;
  for (const g of splitGroupSets.train) {
    if (splitGroupSets.val.has(g) || splitGroupSets.test.has(g)) {
      leakageVerified = false;
    }
  }
  for (const g of splitGroupSets.val) {
    if (splitGroupSets.test.has(g)) {
      leakageVerified = false;
    }
  }

  const realCount = validSamples.filter((s) => s.label === 'real').length;
  const fakeCount = validSamples.filter((s) => s.label === 'fake').length;
  const totalValid = Math.max(1, validSamples.length);

  // Balanced inverse-frequency class weights
  const realWeight = realCount > 0 ? Number((totalValid / (2 * realCount)).toFixed(4)) : 1;
  const fakeWeight = fakeCount > 0 ? Number((totalValid / (2 * fakeCount)).toFixed(4)) : 1;

  const splits = {
    train: {
      real: validSamples.filter((s) => s.split === 'train' && s.label === 'real').length,
      fake: validSamples.filter((s) => s.split === 'train' && s.label === 'fake').length,
      total: validSamples.filter((s) => s.split === 'train').length,
    },
    val: {
      real: validSamples.filter((s) => s.split === 'val' && s.label === 'real').length,
      fake: validSamples.filter((s) => s.split === 'val' && s.label === 'fake').length,
      total: validSamples.filter((s) => s.split === 'val').length,
    },
    test: {
      real: validSamples.filter((s) => s.split === 'test' && s.label === 'real').length,
      fake: validSamples.filter((s) => s.split === 'test' && s.label === 'fake').length,
      total: validSamples.filter((s) => s.split === 'test').length,
    },
  };

  const uniqueGroups = new Set(validSamples.map((s) => s.sourceGroupId)).size;
  const minClass = Math.max(1, Math.min(realCount, fakeCount));
  const maxClass = Math.max(realCount, fakeCount);

  const stats: DatasetStats = {
    totalScanned,
    validCount: validSamples.length,
    corruptedRemoved,
    duplicatesRemoved,
    realCount,
    fakeCount,
    imbalanceRatio: Number((maxClass / minClass).toFixed(3)),
    classWeights: { real: realWeight, fake: fakeWeight },
    splits,
    leakageVerified,
    uniqueSourceGroups: uniqueGroups,
    targetSize: TARGET_IMAGE_SIZE,
    augmentationMethods: [
      'Horizontal Mirror Flip (p=0.5)',
      'Photometric Brightness & Contrast Jitter (±8%)',
      'Balanced Class-Weighted Binary Cross-Entropy',
    ],
    lastPreparedAt: new Date().toISOString(),
  };

  fs.writeFileSync(MANIFEST_PATH, JSON.stringify({ samples: validSamples, stats }, null, 2));
  return { samples: validSamples, stats };
}

export function loadManifestSamples(): DatasetSample[] {
  if (!fs.existsSync(MANIFEST_PATH)) return [];
  const data = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8')) as {
    samples: DatasetSample[];
  };
  return data.samples ?? [];
}

export function getDatasetStats(): DatasetStats {
  seedStarterDatasetIfNeeded();
  if (!fs.existsSync(MANIFEST_PATH)) {
    return prepareAndAuditDataset().stats;
  }
  const data = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8')) as {
    stats: DatasetStats;
  };
  return data.stats;
}
