import type { ClassReportRow, ConfusionMatrixData, RocPoint } from './types.ts';

export interface ComputedMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  specificity: number;
  rocAuc: number;
  confusionMatrix: ConfusionMatrixData;
  rocCurve: RocPoint[];
  classificationReport: ClassReportRow[];
}

export function evaluateBinaryPredictions(
  yTrue: number[],
  yProb: number[],
  threshold = 0.5
): ComputedMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const actual = yTrue[i] >= 0.5 ? 1 : 0;
    const pred = yProb[i] >= threshold ? 1 : 0;
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 0 && pred === 0) tn++;
    else fn++;
  }

  const total = Math.max(1, yTrue.length);
  const accuracy = (tp + tn) / total;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;

  // Real class precision/recall/F1
  const realPrec = tn + fn > 0 ? tn / (tn + fn) : 0;
  const realRec = tn + fp > 0 ? tn / (tn + fp) : 0;
  const realF1 = realPrec + realRec > 0 ? (2 * realPrec * realRec) / (realPrec + realRec) : 0;

  const realSupport = tn + fp;
  const fakeSupport = tp + fn;

  const classificationReport: ClassReportRow[] = [
    {
      label: 'REAL (Authentic)',
      precision: Number(realPrec.toFixed(4)),
      recall: Number(realRec.toFixed(4)),
      f1Score: Number(realF1.toFixed(4)),
      support: realSupport,
    },
    {
      label: 'FAKE (Manipulated)',
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1Score: Number(f1Score.toFixed(4)),
      support: fakeSupport,
    },
    {
      label: 'Macro Avg',
      precision: Number(((realPrec + precision) / 2).toFixed(4)),
      recall: Number(((realRec + recall) / 2).toFixed(4)),
      f1Score: Number(((realF1 + f1Score) / 2).toFixed(4)),
      support: total,
    },
    {
      label: 'Weighted Avg',
      precision: Number(((realPrec * realSupport + precision * fakeSupport) / total).toFixed(4)),
      recall: Number(((realRec * realSupport + recall * fakeSupport) / total).toFixed(4)),
      f1Score: Number(((realF1 * realSupport + f1Score * fakeSupport) / total).toFixed(4)),
      support: total,
    },
  ];

  // Exact ROC Curve & AUC via sorted threshold sweep
  const totalPos = Math.max(1, fakeSupport);
  const totalNeg = Math.max(1, realSupport);
  const uniqueThresholds = Array.from(new Set([1.01, ...yProb, -0.01])).sort((a, b) => b - a);
  const rocCurve: RocPoint[] = [];

  for (const t of uniqueThresholds) {
    let curTp = 0;
    let curFp = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yProb[i] >= t) {
        if (yTrue[i] >= 0.5) curTp++;
        else curFp++;
      }
    }
    rocCurve.push({
      fpr: Number((curFp / totalNeg).toFixed(4)),
      tpr: Number((curTp / totalPos).toFixed(4)),
      threshold: Number(t.toFixed(4)),
    });
  }

  // Trapezoidal integration
  let rocAuc = 0;
  for (let i = 1; i < rocCurve.length; i++) {
    const dx = rocCurve[i].fpr - rocCurve[i - 1].fpr;
    const avgY = (rocCurve[i].tpr + rocCurve[i - 1].tpr) / 2;
    rocAuc += dx * avgY;
  }

  return {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1Score: Number(f1Score.toFixed(4)),
    specificity: Number(specificity.toFixed(4)),
    rocAuc: Number(Math.max(0, Math.min(1, rocAuc)).toFixed(4)),
    confusionMatrix: { tp, fp, tn, fn },
    rocCurve,
    classificationReport,
  };
}
