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

## Deploy for Free

### 🚀 **FASTEST: Replit (Both services in 2 minutes)**

1. Go to [Replit.com](https://replit.com) → Sign up
2. Click **"Create Repl"** → **"Import from GitHub"**
3. Paste: `https://github.com/hardik1511/image-defect-detection`
4. Click **"Import"** and wait for setup
5. **Backend runs automatically** on port 5000
6. **Frontend**: Open another tab → `https://replit.com/new` → Import same repo → Set root to `frontend`
7. **Done!** Both services running

### 🔥 **EASIEST: Heroku (No Docker needed)**

**Backend on Heroku:**
1. Go to [Heroku.com](https://heroku.com) → Sign up
2. Click **"New"** → **"Create new app"**
3. Connect GitHub → Select your repo
4. **Enable auto-deploy** from main branch
5. **Deploy!** (Uses `Procfile` automatically)

**Frontend on Netlify:** (Same as below)

### Option 1: Railway (Backend) + Netlify (Frontend)

**Backend on Railway:**
1. Go to [Railway.app](https://railway.app) and sign up with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `image-defect-detection` repository
4. Railway will auto-detect the `railway.json` config and deploy the backend
5. Copy the backend URL (e.g., `https://your-app.railway.app`)

**Frontend on Netlify:**
1. Go to [Netlify.com](https://netlify.com) and sign up
2. Click "New site from Git" → Connect GitHub
3. Select your repository
4. Set build settings:
   - **Base directory**: `frontend`
   - **Build command**: `npm ci && npm run build`
   - **Publish directory**: `frontend/dist`
5. Add environment variable: `VITE_API_BASE` = `your-railway-backend-url`
6. Deploy!

### Option 2: Railway (Backend) + Vercel (Frontend)

**Backend on Railway:** (Same as above)

**Frontend on Vercel:**
1. Go to [Vercel.com](https://vercel.com) and sign up with GitHub
2. Click "New Project" → Import your repository
3. Set **Root Directory** to `frontend`
4. Add environment variable: `VITE_API_BASE` = `your-railway-backend-url`
5. Deploy!

### Option 3: All on Railway (Both services)

1. Deploy backend as above
2. For frontend, create a new Railway service:
   - **Source**: Same GitHub repo
   - **Root Directory**: `frontend`
   - **Build Command**: `npm ci && npm run build`
   - **Start Command**: `npx serve -s dist -l 3000`
   - **Environment**: `VITE_API_BASE` = `your-backend-url`

### Verify Deployment
- Backend health: `https://your-backend-url/health`
- Frontend: Visit your frontend URL and test image upload

## License
MIT (adjust as needed)
