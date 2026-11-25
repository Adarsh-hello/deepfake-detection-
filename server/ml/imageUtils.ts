import crypto from 'node:crypto';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';

export interface DecodedImage {
  width: number;
  height: number;
  rgba: Uint8Array;
  format: 'jpeg' | 'png';
}

export function decodeImageBuffer(buffer: Buffer): DecodedImage {
  if (buffer.length < 12) {
    throw new Error('Buffer too small to be a valid image.');
  }

  // Check PNG signature
  if (
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47
  ) {
    const parsed = PNG.sync.read(buffer);
    if (parsed.width <= 0 || parsed.height <= 0) {
      throw new Error('Invalid PNG dimensions.');
    }
    return {
      width: parsed.width,
      height: parsed.height,
      rgba: new Uint8Array(parsed.data),
      format: 'png',
    };
  }

  // Check JPEG signature (FF D8 FF)
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    const decoded = jpeg.decode(buffer, { useTArray: true, formatAsRGBA: true });
    if (decoded.width <= 0 || decoded.height <= 0) {
      throw new Error('Invalid JPEG dimensions.');
    }
    return {
      width: decoded.width,
      height: decoded.height,
      rgba: new Uint8Array(decoded.data),
      format: 'jpeg',
    };
  }

  throw new Error('Unsupported image format. Please provide a valid JPEG or PNG image.');
}

export function encodePngDataUrl(width: number, height: number, rgba: Uint8Array): string {
  const png = new PNG({ width, height });
  png.data = Buffer.from(rgba);
  const buf = PNG.sync.write(png);
  return `data:image/png;base64,${buf.toString('base64')}`;
}

export function encodePngBuffer(width: number, height: number, rgba: Uint8Array): Buffer {
  const png = new PNG({ width, height });
  png.data = Buffer.from(rgba);
  return PNG.sync.write(png);
}

export function encodeJpegBuffer(
  width: number,
  height: number,
  rgba: Uint8Array,
  quality = 90
): Buffer {
  const rawImageData = {
    data: Buffer.from(rgba),
    width,
    height,
  };
  return jpeg.encode(rawImageData, quality).data;
}

export function computeSha256(buffer: Buffer): string {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

export function resizeRgbaBilinear(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  const out = new Uint8Array(dstW * dstH * 4);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.max(0, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(srcH - 1, y0 + 1);
    const fy = sy - y0;

    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.max(0, (x + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(srcW - 1, x0 + 1);
      const fx = sx - x0;

      const idx00 = (y0 * srcW + x0) * 4;
      const idx10 = (y0 * srcW + x1) * 4;
      const idx01 = (y1 * srcW + x0) * 4;
      const idx11 = (y1 * srcW + x1) * 4;
      const dstIdx = (y * dstW + x) * 4;

      for (let c = 0; c < 3; c++) {
        const top = src[idx00 + c] * (1 - fx) + src[idx10 + c] * fx;
        const bot = src[idx01 + c] * (1 - fx) + src[idx11 + c] * fx;
        out[dstIdx + c] = Math.round(top * (1 - fy) + bot * fy);
      }
      out[dstIdx + 3] = 255;
    }
  }
  return out;
}

export function cropRgba(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  x0: number,
  y0: number,
  cropW: number,
  cropH: number
): Uint8Array {
  const out = new Uint8Array(cropW * cropH * 4);
  for (let y = 0; y < cropH; y++) {
    const sy = Math.min(srcH - 1, Math.max(0, y0 + y));
    for (let x = 0; x < cropW; x++) {
      const sx = Math.min(srcW - 1, Math.max(0, x0 + x));
      const si = (sy * srcW + sx) * 4;
      const di = (y * cropW + x) * 4;
      out[di] = src[si];
      out[di + 1] = src[si + 1];
      out[di + 2] = src[si + 2];
      out[di + 3] = 255;
    }
  }
  return out;
}

/**
 * 64-bit Difference Hash (dHash) on a 9x8 grayscale downsample for fast near-duplicate detection.
 */
export function computeDHash(rgba: Uint8Array, width: number, height: number): string {
  const small = resizeRgbaBilinear(rgba, width, height, 9, 8);
  const gray = new Float32Array(9 * 8);
  for (let i = 0; i < 9 * 8; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * small[idx] + 0.587 * small[idx + 1] + 0.114 * small[idx + 2];
  }
  let bits = '';
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const left = gray[y * 9 + x];
      const right = gray[y * 9 + x + 1];
      bits += left > right ? '1' : '0';
    }
  }
  let hex = '';
  for (let i = 0; i < 64; i += 4) {
    hex += parseInt(bits.slice(i, i + 4), 2).toString(16);
  }
  return hex;
}

export function hammingDistanceHex(a: string, b: string): number {
  if (a.length !== b.length) return 64;
  let dist = 0;
  for (let i = 0; i < a.length; i++) {
    const xor = parseInt(a[i], 16) ^ parseInt(b[i], 16);
    dist += ((xor >> 3) & 1) + ((xor >> 2) & 1) + ((xor >> 1) & 1) + (xor & 1);
  }
  return dist;
}

export function rgbaToLuminance(rgba: Uint8Array, width: number, height: number): Float32Array {
  const lum = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    lum[i] = (0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2]) / 255.0;
  }
  return lum;
}

export function rgbaToNormalizedRgb(rgba: Uint8Array, width: number, height: number): Float32Array {
  const out = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const si = i * 4;
    const di = i * 3;
    out[di] = rgba[si] / 127.5 - 1.0;
    out[di + 1] = rgba[si + 1] / 127.5 - 1.0;
    out[di + 2] = rgba[si + 2] / 127.5 - 1.0;
  }
  return out;
}

/**
 * Inferno-inspired scientific colormap (0..1 -> RGB).
 */
export function infernoColor(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const stops: Array<[number, number, number]> = [
    [10, 12, 34],
    [66, 20, 110],
    [147, 38, 103],
    [221, 81, 58],
    [252, 165, 10],
    [252, 255, 164],
  ];
  const scaled = v * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const c0 = stops[idx];
  const c1 = stops[idx + 1];
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * frac),
    Math.round(c0[1] + (c1[1] - c0[1]) * frac),
    Math.round(c0[2] + (c1[2] - c0[2]) * frac),
  ];
}

export function renderHeatmapRgba(
  values: Float32Array,
  w: number,
  h: number,
  alpha = 210
): Uint8Array {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  const span = Math.max(1e-6, max - min);
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const norm = (values[i] - min) / span;
    const [r, g, b] = infernoColor(norm);
    const di = i * 4;
    out[di] = r;
    out[di + 1] = g;
    out[di + 2] = b;
    out[di + 3] = Math.round(alpha * (0.25 + 0.75 * norm));
  }
  return out;
}

/**
 * Separable 2D FFT (Cooley-Tukey radix-2) for 64x64 frequency-domain analysis.
 */
function fft1d(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(angle);
    const wlenIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * wRe - im[i + k + len / 2] * wIm;
        const vIm = re[i + k + len / 2] * wIm + im[i + k + len / 2] * wRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
}

export interface FrequencyAnalysis {
  spectrumRgba: Uint8Array;
  spectrumSize: number;
  radialBins: Array<{ band: string; power: number }>;
  highFreqRatio: number;
  periodicGridScore: number;
  spectralSlope: number;
}

export function compute2DFFTSpectrum(
  rgba: Uint8Array,
  width: number,
  height: number,
  size = 64
): FrequencyAnalysis {
  const resized = width === size && height === size ? rgba : resizeRgbaBilinear(rgba, width, height, size, size);
  const lum = rgbaToLuminance(resized, size, size);

  // Subtract DC mean and apply Hann window to reduce boundary leakage
  let mean = 0;
  for (let i = 0; i < lum.length; i++) mean += lum[i];
  mean /= lum.length;

  const re = new Float32Array(size * size);
  const im = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const wy = 0.5 * (1 - Math.cos((2 * Math.PI * y) / (size - 1)));
    for (let x = 0; x < size; x++) {
      const wx = 0.5 * (1 - Math.cos((2 * Math.PI * x) / (size - 1)));
      re[y * size + x] = (lum[y * size + x] - mean) * wx * wy;
    }
  }

  // Row-wise 1D FFT
  const rowRe = new Float32Array(size);
  const rowIm = new Float32Array(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      rowRe[x] = re[y * size + x];
      rowIm[x] = im[y * size + x];
    }
    fft1d(rowRe, rowIm);
    for (let x = 0; x < size; x++) {
      re[y * size + x] = rowRe[x];
      im[y * size + x] = rowIm[x];
    }
  }

  // Column-wise 1D FFT
  const colRe = new Float32Array(size);
  const colIm = new Float32Array(size);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      colRe[y] = re[y * size + x];
      colIm[y] = im[y * size + x];
    }
    fft1d(colRe, colIm);
    for (let y = 0; y < size; y++) {
      re[y * size + x] = colRe[y];
      im[y * size + x] = colIm[y];
    }
  }

  // FFT shift + log magnitude
  const mag = new Float32Array(size * size);
  const half = size / 2;
  let maxLog = 0;
  let totalEnergy = 0;
  let highFreqEnergy = 0;
  let axisPeakEnergy = 0;
  let offAxisHighEnergy = 0;

  const numBins = 12;
  const binSums = new Float64Array(numBins);
  const binCounts = new Int32Array(numBins);
  const maxRadius = Math.sqrt(2) * half;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = (x + half) % size;
      const sy = (y + half) % size;
      const idx = sy * size + sx;
      const power = Math.sqrt(re[idx] * re[idx] + im[idx] * im[idx]);
      const logVal = Math.log1p(power);
      mag[y * size + x] = logVal;
      if (logVal > maxLog) maxLog = logVal;

      const dx = x - half;
      const dy = y - half;
      const r = Math.sqrt(dx * dx + dy * dy);
      const normR = r / half;

      totalEnergy += power;
      if (normR > 0.45) {
        highFreqEnergy += power;
        // Check periodic upsampling peaks near cardinal axes or half-Nyquist grid
        if (Math.abs(dx) <= 1 || Math.abs(dy) <= 1 || Math.abs(Math.abs(dx) - half / 2) <= 1) {
          axisPeakEnergy += power;
        } else {
          offAxisHighEnergy += power;
        }
      }

      const b = Math.min(numBins - 1, Math.floor((r / maxRadius) * numBins));
      binSums[b] += logVal;
      binCounts[b] += 1;
    }
  }

  const spectrumRgba = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const norm = maxLog > 0 ? mag[i] / maxLog : 0;
    const [r, g, b] = infernoColor(Math.pow(norm, 0.8));
    const di = i * 4;
    spectrumRgba[di] = r;
    spectrumRgba[di + 1] = g;
    spectrumRgba[di + 2] = b;
    spectrumRgba[di + 3] = 255;
  }

  const radialBins = Array.from({ length: numBins }, (_, i) => ({
    band: `F${i + 1}`,
    power: Number((binCounts[i] > 0 ? binSums[i] / binCounts[i] : 0).toFixed(4)),
  }));

  const highFreqRatio = totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0;
  const periodicGridScore =
    offAxisHighEnergy > 0 ? Math.min(1, (axisPeakEnergy / (offAxisHighEnergy + 1e-6)) * 2.5) : 0;

  const lowMean = radialBins[1]?.power ?? 1;
  const highMean = radialBins[numBins - 2]?.power ?? 0;
  const spectralSlope = Number((lowMean - highMean).toFixed(4));

  return {
    spectrumRgba,
    spectrumSize: size,
    radialBins,
    highFreqRatio: Number(highFreqRatio.toFixed(4)),
    periodicGridScore: Number(periodicGridScore.toFixed(4)),
    spectralSlope,
  };
}

export interface ElaAnalysis {
  heatmapRgba: Uint8Array;
  meanResidual: number;
  maxResidual: number;
  regionalVariance: number;
}

export function computeELA(
  rgba: Uint8Array,
  width: number,
  height: number,
  targetSize = 192
): ElaAnalysis {
  const resized = resizeRgbaBilinear(rgba, width, height, targetSize, targetSize);
  const recompressedBuf = encodeJpegBuffer(targetSize, targetSize, resized, 90);
  const decoded = decodeImageBuffer(recompressedBuf);

  const diffs = new Float32Array(targetSize * targetSize);
  let sum = 0;
  let maxDiff = 0;
  for (let i = 0; i < targetSize * targetSize; i++) {
    const idx = i * 4;
    const dr = Math.abs(resized[idx] - decoded.rgba[idx]);
    const dg = Math.abs(resized[idx + 1] - decoded.rgba[idx + 1]);
    const db = Math.abs(resized[idx + 2] - decoded.rgba[idx + 2]);
    const val = (dr + dg + db) / 3.0;
    diffs[i] = val;
    sum += val;
    if (val > maxDiff) maxDiff = val;
  }

  const meanResidual = sum / diffs.length;
  // Block variance across 8x8 blocks
  const blockSize = Math.floor(targetSize / 8);
  const blockMeans: number[] = [];
  for (let by = 0; by < 8; by++) {
    for (let bx = 0; bx < 8; bx++) {
      let bSum = 0;
      let count = 0;
      for (let y = by * blockSize; y < (by + 1) * blockSize; y++) {
        for (let x = bx * blockSize; x < (bx + 1) * blockSize; x++) {
          bSum += diffs[y * targetSize + x];
          count++;
        }
      }
      blockMeans.push(bSum / Math.max(1, count));
    }
  }
  const bmAvg = blockMeans.reduce((a, b) => a + b, 0) / blockMeans.length;
  const regionalVariance =
    blockMeans.reduce((acc, v) => acc + (v - bmAvg) ** 2, 0) / blockMeans.length;

  const heatmapRgba = renderHeatmapRgba(diffs, targetSize, targetSize, 230);
  return {
    heatmapRgba,
    meanResidual: Number(meanResidual.toFixed(3)),
    maxResidual: Number(maxDiff.toFixed(2)),
    regionalVariance: Number(regionalVariance.toFixed(3)),
  };
}

export interface TextureNoiseAnalysis {
  noiseMapRgba: Uint8Array;
  laplacianVariance: number;
  centerVsEdgeSharpnessRatio: number;
  localNoiseInconsistency: number;
  chromaCrossCorrelation: number;
}

export function computeTextureNoiseAnalysis(
  rgba: Uint8Array,
  width: number,
  height: number,
  targetSize = 192
): TextureNoiseAnalysis {
  const resized = resizeRgbaBilinear(rgba, width, height, targetSize, targetSize);
  const lum = rgbaToLuminance(resized, targetSize, targetSize);
  const lap = new Float32Array(targetSize * targetSize);

  let sumLap = 0;
  let sumLapSq = 0;
  let centerEnergy = 0;
  let centerCount = 0;
  let edgeEnergy = 0;
  let edgeCount = 0;

  for (let y = 1; y < targetSize - 1; y++) {
    for (let x = 1; x < targetSize - 1; x++) {
      const i = y * targetSize + x;
      const val =
        -4 * lum[i] +
        lum[i - 1] +
        lum[i + 1] +
        lum[i - targetSize] +
        lum[i + targetSize];
      const absVal = Math.abs(val);
      lap[i] = absVal;
      sumLap += absVal;
      sumLapSq += absVal * absVal;

      const inCenter =
        x > targetSize * 0.28 &&
        x < targetSize * 0.72 &&
        y > targetSize * 0.2 &&
        y < targetSize * 0.78;
      if (inCenter) {
        centerEnergy += absVal;
        centerCount++;
      } else {
        edgeEnergy += absVal;
        edgeCount++;
      }
    }
  }

  const n = (targetSize - 2) * (targetSize - 2);
  const meanLap = sumLap / n;
  const laplacianVariance = sumLapSq / n - meanLap * meanLap;

  const blockSize = Math.floor(targetSize / 8);
  const blockStds: number[] = [];
  for (let by = 0; by < 8; by++) {
    for (let bx = 0; bx < 8; bx++) {
      let bSum = 0;
      let bSq = 0;
      let count = 0;
      for (let y = by * blockSize; y < (by + 1) * blockSize; y++) {
        for (let x = bx * blockSize; x < (bx + 1) * blockSize; x++) {
          const v = lap[y * targetSize + x];
          bSum += v;
          bSq += v * v;
          count++;
        }
      }
      const m = bSum / Math.max(1, count);
      blockStds.push(Math.sqrt(Math.max(0, bSq / Math.max(1, count) - m * m)));
    }
  }
  const stdMean = blockStds.reduce((a, b) => a + b, 0) / blockStds.length;
  const localNoiseInconsistency =
    stdMean > 1e-6
      ? Math.sqrt(
          blockStds.reduce((acc, s) => acc + (s - stdMean) ** 2, 0) / blockStds.length
        ) / stdMean
      : 0;

  // Chroma R-G channel correlation
  let sumR = 0;
  let sumG = 0;
  let sumRR = 0;
  let sumGG = 0;
  let sumRG = 0;
  const totalPx = targetSize * targetSize;
  for (let i = 0; i < totalPx; i++) {
    const r = resized[i * 4] / 255;
    const g = resized[i * 4 + 1] / 255;
    sumR += r;
    sumG += g;
    sumRR += r * r;
    sumGG += g * g;
    sumRG += r * g;
  }
  const cov = sumRG / totalPx - (sumR / totalPx) * (sumG / totalPx);
  const stdR = Math.sqrt(Math.max(1e-8, sumRR / totalPx - (sumR / totalPx) ** 2));
  const stdG = Math.sqrt(Math.max(1e-8, sumGG / totalPx - (sumG / totalPx) ** 2));
  const chromaCrossCorrelation = cov / (stdR * stdG);

  const cMean = centerCount > 0 ? centerEnergy / centerCount : 0;
  const eMean = edgeCount > 0 ? edgeEnergy / edgeCount : 1e-5;
  const centerVsEdgeSharpnessRatio = eMean > 1e-6 ? cMean / eMean : 1;

  const noiseMapRgba = renderHeatmapRgba(lap, targetSize, targetSize, 225);

  return {
    noiseMapRgba,
    laplacianVariance: Number((laplacianVariance * 1000).toFixed(4)),
    centerVsEdgeSharpnessRatio: Number(centerVsEdgeSharpnessRatio.toFixed(3)),
    localNoiseInconsistency: Number(localNoiseInconsistency.toFixed(3)),
    chromaCrossCorrelation: Number(chromaCrossCorrelation.toFixed(4)),
  };
}

/**
 * Extracts a 16-dimensional forensic feature vector used by the Hybrid Spatial-Frequency Network.
 */
export function extractFrequencyFeatureVector(
  rgba: Uint8Array,
  width: number,
  height: number
): Float32Array {
  const fft = compute2DFFTSpectrum(rgba, width, height, 32);
  const tex = computeTextureNoiseAnalysis(rgba, width, height, 48);
  const ela = computeELA(rgba, width, height, 48);

  const vec = new Float32Array(16);
  for (let i = 0; i < 10; i++) {
    vec[i] = (fft.radialBins[i]?.power ?? 0) / 4.0;
  }
  vec[10] = fft.highFreqRatio * 3.0;
  vec[11] = fft.periodicGridScore;
  vec[12] = Math.min(2.0, tex.laplacianVariance / 5.0);
  vec[13] = Math.min(2.0, tex.localNoiseInconsistency);
  vec[14] = Math.min(2.0, ela.meanResidual / 10.0);
  vec[15] = tex.chromaCrossCorrelation;
  return vec;
}
