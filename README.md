# LongevityAI-Microbiome 🧬

**ML + KNN + RAG pipeline for gut microbiome-based longevity analysis**

Live API: `https://microbiome.srv1424731.hstgr.cloud` (port 8030)

---

## Overview

This service extends the [LongevityAI](https://github.com/smelesa/longevityai-health-platform) platform with a new **microbiome layer**:

- **ML Inference** — LightGBM/GradientBoosting longevity score from microbiome features
- **KNN Profile Matching** — find similar profiles in the HGMA (Human Gut Microbiome Atlas) dataset
- **RAG Synthesis** — Groq-powered LLM answering microbiome questions with PubMed citations

---

## Architecture

```
User Input (species abundances, symptoms, geography)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              LongevityAI-Microbiome API             │
│                  Port 8030 (VPS)                   │
├─────────────┬──────────────────┬────────────────────┤
│   ML Layer  │   KNN Layer      │   RAG Layer        │
│ GradientBoost│ HGMA similarity │ ChromaDB + Groq    │
│ longevity   │ profile matches  │ PubMed evidence     │
│ score       │ + recommendations│ + citations        │
└─────────────┴──────────────────┴────────────────────┘
        │
        ▼
Combined response: score + profile matches + evidence
```

---

## Input Features (Microbiome)

| Field | Type | Description |
|-------|------|-------------|
| `akkermansia_pct` | float 0-100 | *A. muciniphila* relative abundance |
| `bifidobacterium_pct` | float 0-100 | *Bifidobacterium* spp. |
| `butyrate_producers_pct` | float 0-100 | Combined butyrate-producing species |
| `alpha_diversity` | float 0-10 | Shannon diversity index |
| `enterotype` | int 1-3 | 1=Bacteroides, 2=Prevotella, 3=Ruminococcaceae |
| `country` | str | Country/region for geography factor |
| `known_diseases` | list[str] | Disease flags from HGMA (23 conditions) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | ML longevity score prediction |
| `POST` | `/knn/match` | Find K=7 similar HGMA profiles |
| `POST` | `/rag/query` | Query PubMed-backed RAG |
| `POST` | `/api/rag/import` | Reload RAG collections |
| `GET` | `/api/analyze` | Full ML+KNN+RAG pipeline |

---

## Data Sources

- **HGMA** (Human Gut Microbiome Atlas) — species × disease matrix, 20 countries, 23 diseases
- **PubMed** (NCBI E-utilities) — microbiome-longevity literature via ESearch/EFetch
- **ChromaDB** — persistent vector store for RAG

---

## Setup on VPS

```bash
# 1. SSH to VPS
ssh root@187.77.161.49

# 2. Create working dir
mkdir -p /root/longevity-microbiome/src
mkdir -p /root/longevity-microbiome/data/processed
mkdir -p /root/longevity-microbiome/data/chroma_microbiome

# 3. Copy files (from local project)
scp -r src/ microbiome_api.py scripts/ data/ root@187.77.161.49:/root/longevity-microbiome/

# 4. Install Python deps
python3 -m venv /root/longevity-rag-venv
source /root/longevity-rag-venv/bin/activate
pip install fastapi uvicorn pydantic scikit-learn chromadb numpy pandas joblib requests

# 5. Install systemd service
scp src/longevityai_microbiome.service root@187.77.161.49:/etc/systemd/system/
systemctl daemon-reload
systemctl enable longevityai_microbiome
systemctl start longevityai_microbiome

# 6. Set Groq API key (get from Vault/1Password)
echo "export GROQ_API_KEY=your_key_here" >> /root/longevity-rag-venv/bin/activate
# Or edit the service file directly:
nano /etc/systemd/system/longevityai_microbiome.service
# Add: Environment="GROQ_API_KEY=gsk_your_key"

# 7. Configure nginx subdomain
# Add to /etc/nginx/conf.d/microbiome.srv1424731.hstgr.cloud.conf:
# location / { proxy_pass http://127.0.0.1:8030; }

# 8. Test
curl http://localhost:8030/health
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/01_clean_hgma_data.py` | Parse and clean HGMA CSV data |
| `scripts/02_scrape_pubmed.py` | Fetch PubMed articles via E-utilities |
| `scripts/03_train_ml_knn_rag.py` | Train ML model, build KNN index, populate RAG |

---

## Related Repos

- [LongevityAI Health Platform](https://github.com/smelesa/longevityai-health-platform) — ML v5 + v6 biomarker models
- [LongevityAI RAG](https://github.com/smelesa/longevityai-rag) — RAG pipeline + narrative generator

---

*🦞 Jarvis — 2026-05-05*
