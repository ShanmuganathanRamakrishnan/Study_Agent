# Lucent - Cloud Deployment Guide

## Prerequisites

- Docker with NVIDIA Container Toolkit
- GPU with CUDA 12.1 support (minimum 8GB VRAM recommended)
- ~10GB disk space for model cache

---

## Local Docker Build

```bash
# Build the image
docker build -t lucent-backend .

# Run with GPU support
docker run --gpus all -p 8000:8000 lucent-backend
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/.cache/huggingface` | HuggingFace cache directory |
| `TRANSFORMERS_CACHE` | `/app/.cache/huggingface` | Model cache path |

---

## RunPod Deployment

1. **Create Pod:**
   - Template: `runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04`
   - GPU: RTX 4090 or A100 (8GB+ VRAM)
   - Disk: 50GB

2. **Upload Code:**
   ```bash
   scp -r *.py runpod:/workspace/
   scp requirements.txt Dockerfile runpod:/workspace/
   ```

3. **Install & Run:**
   ```bash
   cd /workspace
   pip install -r requirements.txt
   python -m uvicorn server:app --host 0.0.0.0 --port 8000
   ```

4. **Expose Port:**
   - Configure RunPod proxy to expose port 8000

---

## AWS EC2 GPU Deployment

1. **Launch Instance:**
   - AMI: Deep Learning AMI (Ubuntu 22.04) with CUDA 12.1
   - Instance: `g4dn.xlarge` (T4 GPU) or `g5.xlarge` (A10G)
   - Storage: 50GB EBS

2. **Install Docker:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Build & Run:**
   ```bash
   git clone <your-repo>
   cd StudyAgent
   docker build -t lucent-backend .
   docker run --gpus all -p 8000:8000 -d lucent-backend
   ```

4. **Security Group:**
   - Allow inbound TCP 8000

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy","rag_ready":true,"model_ready":true}
```

---

## Verification Checklist

- [ ] Health endpoint returns `{"status":"healthy"}`
- [ ] `/system_status` shows indexed domains
- [ ] PDF upload via `/upload_file` succeeds
- [ ] `/ask_question` retrieves and answers correctly
- [ ] GPU utilization visible during inference

---

## Files Required

```
├── Dockerfile
├── requirements.txt
├── server.py
├── study_agent.py
├── pdf_extractor.py
├── math_support.py
├── intent_router.py
└── conversation_context.py
```
