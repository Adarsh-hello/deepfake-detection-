import type { ForensicAnalysisResult } from '../types.ts';

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadJsonReport(result: ForensicAnalysisResult): void {
  const cleanName = result.filename.replace(/[^a-zA-Z0-9._-]/g, '_');
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: 'application/json',
  });
  triggerDownload(blob, `spectratrace_report_${cleanName}.json`);
}

export function downloadHtmlReport(result: ForensicAnalysisResult): void {
  const cleanName = result.filename.replace(/[^a-zA-Z0-9._-]/g, '_');
  const verdictColor = result.prediction === 'REAL' ? '#10b981' : '#f43f5e';
  const rows = result.indicators
    .map(
      (ind) => `
      <tr>
        <td style="padding:10px 12px;border-bottom:1px solid #1e293b;font-weight:600;color:#e2e8f0;">${ind.name}</td>
        <td style="padding:10px 12px;border-bottom:1px solid #1e293b;font-family:monospace;color:#93c5fd;">${ind.value} ${ind.unit}</td>
        <td style="padding:10px 12px;border-bottom:1px solid #1e293b;color:#94a3b8;">${ind.baselineRange}</td>
        <td style="padding:10px 12px;border-bottom:1px solid #1e293b;text-transform:uppercase;font-size:12px;color:${
          ind.assessment === 'anomalous'
            ? '#fb7185'
            : ind.assessment === 'elevated'
              ? '#fbbf24'
              : '#34d399'
        };">${ind.assessment}</td>
      </tr>`
    )
    .join('');

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SpectraTrace Forensic Report — ${result.filename}</title>
</head>
<body style="margin:0;padding:32px;background:#050814;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;line-height:1.5;">
  <div style="max-width:960px;margin:0 auto;background:#0b1228;border:1px solid #1e293b;border-radius:12px;padding:32px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #1e293b;padding-bottom:20px;margin-bottom:24px;">
      <div>
        <div style="font-size:12px;color:#60a5fa;letter-spacing:0.08em;text-transform:uppercase;">SpectraTrace Forensic Report</div>
        <h1 style="margin:6px 0 4px;font-size:24px;color:#f8fafc;">${result.filename}</h1>
        <div style="font-size:12px;color:#94a3b8;font-family:monospace;">SHA-256: ${result.sha256}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:28px;font-weight:700;color:${verdictColor};">${result.prediction}</div>
        <div style="font-size:14px;color:#cbd5e1;">Confidence: <strong>${result.confidence.toFixed(2)}%</strong></div>
        <div style="font-size:12px;color:#94a3b8;">Model: ${result.modelDisplayName}</div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;">
        <div style="font-size:11px;color:#94a3b8;">P(FAKE)</div>
        <div style="font-size:18px;font-weight:600;font-family:monospace;">${(result.fakeProbability * 100).toFixed(2)}%</div>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;">
        <div style="font-size:11px;color:#94a3b8;">P(REAL)</div>
        <div style="font-size:18px;font-weight:600;font-family:monospace;">${(result.realProbability * 100).toFixed(2)}%</div>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;">
        <div style="font-size:11px;color:#94a3b8;">Resolution</div>
        <div style="font-size:18px;font-weight:600;font-family:monospace;">${result.imageWidth}×${result.imageHeight}</div>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px;">
        <div style="font-size:11px;color:#94a3b8;">Inference Time</div>
        <div style="font-size:18px;font-weight:600;font-family:monospace;">${result.inferenceTimeMs.toFixed(2)} ms</div>
      </div>
    </div>

    <h2 style="font-size:16px;margin:0 0 12px;color:#f8fafc;">Visual Evidence &amp; Attribution Maps</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px;">
      <div>
        <img src="${result.originalDataUrl}" style="width:100%;border-radius:8px;border:1px solid #1e293b;" />
        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">Original Input</div>
      </div>
      <div>
        <img src="${result.gradCamHeatmapDataUrl}" style="width:100%;border-radius:8px;border:1px solid #1e293b;background:#050814;" />
        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">Model Saliency Heatmap</div>
      </div>
      <div>
        <img src="${result.elaHeatmapDataUrl}" style="width:100%;border-radius:8px;border:1px solid #1e293b;background:#050814;" />
        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">ELA Residual Map</div>
      </div>
      <div>
        <img src="${result.fftSpectrumDataUrl}" style="width:100%;border-radius:8px;border:1px solid #1e293b;background:#050814;" />
        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">2D-FFT Spectrum</div>
      </div>
    </div>

    <h2 style="font-size:16px;margin:0 0 12px;color:#f8fafc;">Supporting Forensic Indicators</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;">
      <thead>
        <tr style="text-align:left;color:#94a3b8;border-bottom:1px solid #334155;">
          <th style="padding:8px 12px;">Indicator</th>
          <th style="padding:8px 12px;">Measured</th>
          <th style="padding:8px 12px;">Reference Range</th>
          <th style="padding:8px 12px;">Assessment</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>

    <div style="padding:14px 16px;background:#0f172a;border:1px solid #1e293b;border-radius:8px;font-size:12px;color:#94a3b8;">
      <strong>Methodological Note:</strong> ${result.disclaimer}
      <div style="margin-top:4px;font-family:monospace;">Generated at ${result.analyzedAt}</div>
    </div>
  </div>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  triggerDownload(blob, `spectratrace_report_${cleanName}.html`);
}
