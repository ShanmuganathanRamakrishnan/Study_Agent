# Lucent

**Answers grounded in your study material.**

Lucent is a source-grounded question-answering system for academic study. It retrieves and synthesizes answers strictly from your uploaded materials. When information is insufficient, it refuses rather than fabricates.

---

## What Lucent Does

- ✅ Answers questions using **only** your uploaded study materials
- ✅ Provides supporting quotes for every answer
- ✅ Classifies confidence (HIGH / LOW) for transparency
- ✅ Refuses to answer when evidence is insufficient
- ✅ Supports PDF and TXT uploads
- ✅ Provides structured explanations (definition, formula, example)

## What Lucent Does NOT Do

- ❌ Access the internet or external knowledge bases
- ❌ Remember conversations across sessions
- ❌ Process images, diagrams, or scanned PDFs
- ❌ Solve complex mathematical derivations
- ❌ Provide answers without source material support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React/Vite)                │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────┐  │
│  │  Chat   │  │  Study  │  │         Upload              │  │
│  │   Tab   │  │   Tab   │  │          Tab                │  │
│  └────┬────┘  └────┬────┘  └──────────────┬──────────────┘  │
└───────┼────────────┼───────────────────────┼────────────────┘
        │            │                       │
        ▼            ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Intent       │  │ RAG System   │  │ PDF Extractor    │   │
│  │ Router       │  │ (Embedding)  │  │ (pdfplumber)     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘   │
│         │                 │                                  │
│         ▼                 ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LLM Generation (Qwen 3B, 4-bit)         │   │
│  │              GPU Required / CUDA 12.1                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 12.1 (8GB+ VRAM)

### Backend

```bash
# Clone repository
git clone https://github.com/your-username/lucent.git
cd lucent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

### Frontend

```bash
cd ui

# Install dependencies
npm install

# Development
npm run dev

# Production build
npm run build
```

---

## Usage

### Ask Questions

1. Upload a PDF or TXT file with study materials
2. Ask questions in the Chat tab
3. Receive grounded answers with supporting quotes

### Confidence Behavior

| Confidence | Meaning | Behavior |
|------------|---------|----------|
| ✓ Confident | Strong match in materials | Full structured answer |
| ⚠ Limited | Weak or ambiguous match | Brief answer, stricter mode |
| Refused | No relevant content | Polite refusal message |

### Upload Materials

- Supported: `.pdf`, `.txt`
- Maximum: 100 pages (PDF), 10 MB
- Minimum: 100 words

---

## Known Limitations

| Limitation | Description |
|------------|-------------|
| Model Size | 3B parameters - complex reasoning may be limited |
| No Internet | Cannot access external knowledge |
| GPU Required | CPU inference not supported |
| Text-Only PDFs | Scanned/image PDFs not supported |
| 100-Page Max | Larger PDFs are truncated |
| Single Session | Conversation resets on page refresh |

---

## Security & Safety

- **Hallucination Prevention:** All answers require retrieved source quotes
- **Confidence Gating:** LOW confidence triggers stricter generation
- **Refusal Over Fabrication:** System refuses when evidence is insufficient
- **No Secrets:** No API keys or tokens in codebase
- **XSS Protection:** All user content rendered as text (no raw HTML)

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for full audit report.

---

## Deployment

### Cloud (GPU)

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Docker deployment with GPU
- RunPod instructions
- AWS EC2 GPU setup

### Frontend

See [ui/DEPLOYMENT.md](ui/DEPLOYMENT.md) for Vercel/Netlify deployment.

---

## API Documentation

See [API_CONTRACT.md](API_CONTRACT.md) for endpoint specifications.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/system_status` | GET | System state |
| `/ask_question` | POST | Ask a question |
| `/upload_file` | POST | Upload PDF/TXT |
| `/upload_material` | POST | Upload raw text |
| `/generate_study_guide` | POST | Generate questions |

---

## Disclaimer

Lucent is an **educational portfolio project** demonstrating:
- Retrieval-Augmented Generation (RAG)
- Hallucination prevention techniques
- Source-grounded question answering

**Not intended for production educational use.** Answers are limited to uploaded materials and may be incomplete or incorrect. Always verify information from authoritative sources.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Author

Built as a demonstration of grounded AI systems.
