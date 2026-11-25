# DEEPFAKE DETECTOR — AI-Powered Image Authenticity Verification

A production-quality AI Deepfake Detection platform featuring a custom Convolutional Neural Network (CNN) pipeline in **TensorFlow / Keras** (`deepfake_model.keras`) and a live interactive **dark-theme glassmorphism web dashboard** powered by **TensorFlow.js**, **Express**, and **React + Tailwind CSS**.

---

## Project Folder Structure

```
├── server.ts                         # Express + Vite server & REST inference/training API
├── server/ml/
│   ├── types.ts                      # Shared ML & forensic TypeScript contracts
│   ├── imageUtils.ts                 # Image decoding, SHA-256/dHash, 2D-FFT, ELA, texture maps
│   ├── dataset.ts                    # Leak-free dataset audit, deduplication & augmentation
│   ├── models.ts                     # Custom CNN, ResNet-Lite, Xception/MobileNet-DS, Hybrid
│   ├── metrics.ts                    # Confusion Matrix, Precision/Recall/F1, ROC-AUC
│   ├── trainer.ts                    # Adam training, Early Stopping, LR scheduling, checkpoints
│   ├── runTrainingJob.ts             # Isolated background training worker
│   └── forensics.ts                  # Real inference, Grad-CAM attribution, detection history
├── python_pipeline/                  # Standalone TensorFlow/Keras + FastAPI ML Pipeline
│   ├── dataset_prep.py               # 128×128 preprocessing, 70/15/15 split & augmentation
│   ├── models.py                     # Custom CNN (Conv2D, BatchNorm, MaxPool, Dropout, Adam)
│   ├── train.py                      # EarlyStopping + ModelCheckpoint -> deepfake_model.keras
│   ├── evaluate.py                   # Scikit-learn metrics + Matplotlib Accuracy/Loss/ROC/CM plots
│   ├── forensics.py                  # Supporting 2D-FFT, ELA, and texture diagnostics
│   └── api.py                        # FastAPI upload, validation & prediction backend
├── src/
│   ├── App.tsx                       # Main application state & API orchestration
│   ├── index.css                     # Dark navy/purple/electric-blue glassmorphism styles
│   ├── types.ts                      # Frontend data contracts
│   ├── utils/reportExport.ts         # Downloadable HTML & JSON forensic report generator
│   └── components/
│       ├── Navbar.tsx                # Single-row navigation bar
│       ├── OverviewView.tsx          # Hero ("DEEPFAKE DETECTOR"), KPIs, live sample launcher
│       ├── AnalyzeView.tsx           # Drag-and-drop upload, preview, REAL/FAKE bar & Grad-CAM
│       ├── BenchmarksView.tsx        # Confusion matrix, ROC-AUC, accuracy/loss charts & dataset lab
│       ├── HistoryView.tsx           # Searchable detection history table
│       └── Charts.tsx                # Custom SVG evaluation visualizations
├── requirements.txt                  # Python dependencies (TensorFlow, Keras, FastAPI, Scikit-learn)
├── package.json                      # Node.js / Vite / Express dependencies
└── README.md                         # Project documentation & setup instructions
```

---

## Live Web Application (Port 3000)

```bash
# 1. Install dependencies
npm install

# 2. Start the live full-stack server (Express + TensorFlow.js + React)
npm run dev

# 3. Build & run in production mode
npm run build
npm start
```

Open `http://localhost:3000`. On startup, the server prepares the verified dataset and trains the neural models so you can immediately upload `JPG`, `JPEG`, or `PNG` images (or click any sample portrait) for real-time prediction, confidence scoring, Grad-CAM heatmaps, model benchmarks, and detection history.

---

## Python TensorFlow / Keras & FastAPI Pipeline

```bash
# 1. Create and activate a Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python ML & FastAPI dependencies
pip install -r requirements.txt

# 3. Prepare 128×128 dataset with strict 70% train / 15% val / 15% test split
python python_pipeline/dataset_prep.py \
  --real-dir data/dataset/real \
  --fake-dir data/dataset/fake \
  --output-csv python_pipeline/artifacts/splits.csv

# 4. Train custom CNN and save best model as deepfake_model.keras
python python_pipeline/train.py \
  --splits-csv python_pipeline/artifacts/splits.csv \
  --artifacts-dir python_pipeline/artifacts \
  --epochs 30 \
  --batch-size 32

# 5. Evaluate on the held-out 15% test split & export Matplotlib charts
python python_pipeline/evaluate.py \
  --splits-csv python_pipeline/artifacts/splits.csv \
  --artifacts-dir python_pipeline/artifacts

# 6. Run the FastAPI inference server
uvicorn python_pipeline.api:app --host 0.0.0.0 --port 8000
```

### API Endpoints (FastAPI `python_pipeline/api.py`)
- `GET /health` — Server and `deepfake_model.keras` readiness check.
- `POST /predict` — Accepts `JPG`, `JPEG`, or `PNG` multipart file upload, resizes to `128×128` RGB normalized to `[0, 1]`, and returns `REAL` / `FAKE` classification with confidence percentage.
- `GET /metrics` — Returns held-out test Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.
- `GET /history` — Returns logged predictions with filenames and timestamps.
