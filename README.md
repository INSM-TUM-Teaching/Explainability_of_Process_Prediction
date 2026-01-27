# Explainability_of_Process_Prediction


---

##  Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/INSM-TUM-Teaching/Explainability_of_Process_Prediction.git
cd Explainability_of_Process_Prediction

# 2. Create and activate a virtual environment
python -m venv venv_bpm
source venv_bpm/bin/activate    # On Windows: venv_bpm\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt



Directory Structure

Explainability_of_Process_Prediction/
BPM_RESEARCH_APP/
│
├── BPI_Models/
│   ├── BPI_logs_csv/                          # Raw BPI event logs
│   ├── BPI_logs_preprocessed_csv/             # Preprocessed CSV files (input datasets)
│   └── BPI_logs_xes/                          # XES format logs
│
├── conv_and_viz/
│   ├── preprocessor_csv.py                    # CSV preprocessing utilities
│   ├── visualize.py                           # Data visualization tools
│   └── xes_to_csv.py                          # XES to CSV conversion
│   └── process_model.py                       # Process model utilities
│
├── gnns/                                       # GNN Package
│   ├── __init__.py                            # Package initialization
│   ├── model.py                               # HeteroGNN architecture
│   ├── dataset_builder.py                     # Graph dataset builder
│   ├── prefix_generation.py                   # Prefix generation (standalone)
│   └── prediction/
│       ├── __init__.py
│       └── gnn_predictor.py                   # Unified GNN predictor - For all 3 tasks
│
├── transformers/                               # Transformer Package
│   ├── __init__.py                            # Package initialization
│   ├── model.py                               # Transformer architecture
│   └── prediction/
│       ├── __init__.py
│       ├── next_activity.py                   # Next activity predictor
│       ├── event_time.py                      # Event time predictor
│       └── remaining_time.py                  # Remaining time predictor
│
├── explainability/                            # Explainability Package
│   ├── __init__.py                            
│   ├── gnns/
│       └── gnn_explainer.py                   # Unified GNN predictor
│       └── __init__.py                                                   
│   └── transformers/
│       └── transformer_explainer.py            # Unified Transformer Explainer
│       └── __init__.py
│
├── results/                                    # Output directory (auto-created)
├── comprehensive_test_results/                 # Output directory for test results (auto-created)
│
├── main.py                                     # Main entry point to the pipeline
├── automated_test.py                          # Automated testing Script to the main pipeline.(INDEPENDENT Sript)
├── requirements.txt                           # Transformer dependencies
├── README.md                                  # Main documentation



Usage Steps (After executing main.py)
1. Select Model Type
   Option 1: Transformer (choose task: Next Activity / Event Time / Remaining Time)
   Option 2: GNN (predicts all 3 tasks simultaneously)

2. Select Dataset
   Choose from available CSV files in BPI_Models/BPI_logs_preprocessed_csv/

3. Configure Data Split
   Choose preset (70-15-15, 80-10-10, 60-20-20) or custom split

4. Configure Model
   Use default hyperparameters or customize

5. Training Starts Automatically
   Progress displayed in real-time

6. Results saved to results/


Frontend
Prerequisites

Node.js 18+ (recommended) and npm

Setup & Run (Development)
# From repo root:
cd frontend

# Install dependencies
npm install

# Create local env file (frontend uses Vite env vars)
# macOS/Linux:
cp .env.example .env.local 2>/dev/null || true
# Windows (PowerShell):
# Copy-Item .env.example .env.local -ErrorAction SilentlyContinue

# Put this line into frontend/.env.local
# (Adjust if your backend runs on a different host/port)
VITE_API_BASE_URL=http://localhost:8000/api

# Start the frontend dev server
npm run dev

---

## Deployment (Firebase Hosting + Cloud Run)

This repo is set up so the frontend calls the backend via same-origin `/api/*`:

- Firebase Hosting serves `frontend/dist`
- Firebase Hosting rewrites `/api/**` to Cloud Run service `ppm-backend` in `europe-west1`
- Backend endpoints are under `/api/...`

### One-time setup (Console)

1. Firebase Console: create/select project `explainability-bedf8`
2. Enable Billing (Blaze)
3. Google Cloud APIs: enable Cloud Run Admin API, Cloud Build API, Artifact Registry API
4. (Optional) Firebase Auth can be added later if needed.

### Deploy backend (Cloud Run)

```bash
gcloud auth login
gcloud config set project explainability-bedf8
gcloud run deploy ppm-backend --source . --region europe-west1 --allow-unauthenticated
```

### Deploy frontend (Firebase Hosting)

```bash
firebase login
firebase use explainability-bedf8

npm --prefix frontend ci
npm --prefix frontend run build

firebase deploy --only hosting
```

### API authentication

Auth is currently not enforced. If you want to add authentication later, we can integrate Firebase Auth or another provider.


Frontend will be available at:

http://localhost:5173

Backend (FastAPI)
Setup & Run
# From repo root:
cd backend

# Activate venv (example)
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install backend dependencies (if not installed yet)
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload --port 8000


Backend will be available at:

API: http://localhost:8000

Swagger: http://localhost:8000/docs

Run Both (Two Terminals)

Terminal 1 (Backend):

cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000


Terminal 2 (Frontend):

cd frontend
npm install
npm run dev
