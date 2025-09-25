# Image Defect Detection System using Deep Learning

This repo contains a **full-stack implementation**:
- **backend/** — FastAPI + PyTorch (Mask R-CNN via `torchvision`) for inference
- **frontend/** — React (Vite) UI for uploading images and overlaying detections

## Quickstart

### 1) Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Frontend
```bash
cd ../frontend
npm install
# optional: set API base for local if backend on another host/port
# Windows PowerShell:
#   $env:VITE_API_BASE="http://127.0.0.1:8000"; npm run dev
# macOS/Linux (bash/zsh):
#   VITE_API_BASE=http://127.0.0.1:8000 npm run dev
npm run dev
# Open the printed URL (e.g., http://localhost:5173)
```

### Notes
- First run of the backend will **download pretrained weights** automatically (through torchvision). Ensure internet access on that first run.
- For **custom defects**, fine-tune Mask R-CNN on your dataset and replace the weights loading logic in `backend/model.py`.

## Folder Structure
```
image-defect-detection/
├─ backend/
│  ├─ main.py
│  ├─ model.py
│  ├─ utils.py
│  ├─ requirements.txt
│  └─ README.md
└─ frontend/
   ├─ index.html
   ├─ vite.config.js
   ├─ package.json
   ├─ src/
   │  ├─ main.jsx
   │  └─ App.jsx
   └─ README.md
```

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose New → Blueprint and select this repo.
3. The blueprint `render.yaml` defines:
   - Backend: Docker web service built from `backend/Dockerfile`, port 8000.
   - Frontend: Static site built from `frontend`, publish path `frontend/dist`.
4. After the first deploy, copy the backend public URL and set it as `VITE_API_BASE` env var on the frontend service. Redeploy the frontend.
5. Verify health at `<backend-url>/health` and then test the UI.

## License
MIT (adjust as needed)
