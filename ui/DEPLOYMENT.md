# Lucent Frontend - Static Deployment Guide

## Overview

The Lucent frontend is a client-only React (Vite) application.  
It can be deployed to any static hosting service.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | Yes | Backend API URL (e.g., `https://api.lucent.example.com`) |

---

## Build

```bash
cd ui

# Install dependencies
npm install

# Set API URL and build
VITE_API_URL=https://your-backend-url.com npm run build
```

Output: `dist/` folder containing static files.

---

## Vercel Deployment

### Option 1: CLI

```bash
npm install -g vercel
cd ui
vercel
```

### Option 2: GitHub Integration

1. Connect repo to Vercel
2. Set root directory: `ui`
3. Framework preset: Vite
4. Environment variable: `VITE_API_URL=https://your-backend-url.com`
5. Deploy

### vercel.json (Optional)

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

---

## Netlify Deployment

### Option 1: CLI

```bash
npm install -g netlify-cli
cd ui
netlify deploy --prod --dir=dist
```

### Option 2: Dashboard

1. Connect repo to Netlify
2. Base directory: `ui`
3. Build command: `npm run build`
4. Publish directory: `ui/dist`
5. Environment variable: `VITE_API_URL=https://your-backend-url.com`

### netlify.toml (Optional)

```toml
[build]
  base = "ui"
  command = "npm run build"
  publish = "dist"

[build.environment]
  VITE_API_URL = "https://your-backend-url.com"
```

---

## CORS Configuration

Ensure your backend allows requests from your frontend domain:

```python
# server.py - Already configured
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Verification Checklist

- [ ] Frontend loads without console errors
- [ ] "Lucent" branding appears correctly
- [ ] Chat tab: Ask a question → Response received
- [ ] Upload tab: PDF upload → "Indexed" status
- [ ] Refusal messages display correctly (no raw JSON)
- [ ] Confidence badges show (✓ Confident / ⚠ Limited Confidence)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CORS error | Check backend `allow_origins` includes frontend URL |
| Network error | Verify `VITE_API_URL` is correct |
| 404 on refresh | Add SPA redirect rules (Vercel/Netlify handle this automatically) |

---

## File Structure

```
ui/
├── dist/           # Built static files (deploy this)
├── src/
│   ├── lib/api.ts  # API configuration (uses VITE_API_URL)
│   ├── components/
│   └── App.tsx
├── package.json
└── vite.config.ts
```
