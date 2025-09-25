# Frontend (React + Vite)

## Setup & Run
```bash
npm install
npm run dev
# Opens at http://localhost:5173
```

- Set backend URL via `.env` variable `VITE_API_BASE` (default: `http://localhost:8000`).
- Upload an image and click **Run Detection**.

## Configure API base URL

Use `VITE_API_BASE` to point the UI to your backend (no trailing slash).

- Local: `VITE_API_BASE=http://127.0.0.1:8000`
- Production (e.g., Render): `VITE_API_BASE=https://<your-backend>.onrender.com`

You can provide it inline: `VITE_API_BASE=... npm run dev`, or via a `.env` file (not committed).
