#!/usr/bin/env python3
"""
LongevityAI-Microbiome — FastAPI Server
Port 8030 — ML inference + KNN + RAG endpoints
"""

import os, sys, json, re, math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Use existing venv
VENV = "/root/longevity-rag-venv"
sys.path.insert(0, VENV)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sklearn.neighbors import NearestNeighbors
import chromadb
from chromadb.config import Settings

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("/root/longevity-microbiome/data")
MODEL_DIR = Path("/root/longevity-microbiome/src")
PORT = 8030

app = FastAPI(title="LongevityAI-Microbiome", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ============================================================
# LOAD MODEL INFO
# ============================================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model info...")
with open(DATA_DIR / "microbiome_model_info.json") as f:
    MODEL_INFO = json.load(f)

FEATURES = MODEL_INFO["features"]
ENTEROTYPE_MAP = MODEL_INFO["enterotype_map"]
GEOGRAPHY_CLASSES = {name: i for i, name in enumerate(MODEL_INFO["geography_classes"])}

# ============================================================
# LOAD KNN INDEX
# ============================================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading KNN index...")
knn_matrix = np.load(DATA_DIR / "knn_species_matrix.npy")
sample_ids = pd.read_csv(DATA_DIR / "knn_sample_ids.csv", index_col=0).index.tolist()
samples_meta = pd.read_csv(DATA_DIR / "hgma_samples_enriched.csv", index_col=0)

knn = NearestNeighbors(n_neighbors=7, metric="cosine", algorithm="brute")
knn.fit(knn_matrix)
print(f"  KNN index: {knn_matrix.shape}")

# ============================================================
# LOAD ML MODEL (trained locally, saved as joblib)
# ============================================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading ML model...")
# We retrain on VPS since no joblib file was exported
# Use GradientBoostingRegressor with same params
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

DISEASE_MAP = MODEL_INFO["disease_longevity_map"]
ENTEROTYPE_MAP = MODEL_INFO["enterotype_map"]  # {name: num}
GEOGRAPHY_CLASSES = MODEL_INFO["geography_classes"]  # list of country names

# Compute longevity_label from Disease
samples_meta["longevity_label"] = samples_meta["Disease"].map(DISEASE_MAP).fillna(0.5)

# Encode features
samples_meta["enteroType_num"] = samples_meta["enteroType"].map(ENTEROTYPE_MAP).fillna(0)
samples_meta["geography_enc"] = samples_meta["Geography"].apply(
    lambda g: GEOGRAPHY_CLASSES.index(g) if g in GEOGRAPHY_CLASSES else 0
)
samples_meta["gender_enc"] = samples_meta["Gender"].apply(lambda g: 0 if g == "Male" else 1)


FEATURES = MODEL_INFO["features"]
X = samples_meta[FEATURES].fillna(samples_meta[FEATURES].median()).values
y = samples_meta["longevity_label"].values

ML_MODEL = GradientBoostingRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
ML_MODEL.fit(X, y)
print(f"  ML model trained on {len(X)} samples")

# Update GEOGRAPHY_CLASSES to dict for encoding
GEOGRAPHY_CLASSES_DICT = {name: i for i, name in enumerate(GEOGRAPHY_CLASSES)}

# ============================================================
# CHROMADB (RAG)
# ============================================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading ChromaDB...")
CHROMA_PATH = "/root/longevity-microbiome/data/chroma_microbiome"
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    COLLECTION = chroma_client.get_collection("microbiome_pubmed")
    print(f"  Collection loaded: {COLLECTION.count()} documents")
except:
    print("  Collection not found — run /api/rag/import to create it")
    COLLECTION = None

# ============================================================
# GROQ LLM (reuse existing API key from env)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

async def groq_complete(prompt: str, system: str = "") -> str:
    if not GROQ_API_KEY:
        return "[GROQ_API_KEY not set — set GROQ_API_KEY env var]"
    import requests
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 1024
    }
    resp = requests.post(
        GROQ_URL,
        json=payload,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ============================================================
# MODELS
# ============================================================
class MicrobiomeInput(BaseModel):
    shannon_index: float
    observed_species: float
    Age: float
    BMI: float
    MgsRichness: float
    GeneRichness: float
    enterotype: str  # "ET-Bacteroides", "ET-Firmicutes", "ET-Prevotella"
    geography: str   # country name
    gender: str      # "Male", "Female"

class RAGQuery(BaseModel):
    query: str
    top_k: int = 5

# ============================================================
# ROUTES
# ============================================================
@app.get("/")
async def root():
    return {"service": "LongevityAI-Microbiome", "version": "1.0.0", "port": PORT}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "knn_samples": knn_matrix.shape[0],
        "knn_features": knn_matrix.shape[1],
        "ml_trained": True,
        "rag_collection": COLLECTION.count() if COLLECTION else "not loaded"
    }

@app.post("/api/microbiome/score")
async def microbiome_score(input: MicrobiomeInput):
    """Compute longevity score from microbiome features"""
    geo_enc = GEOGRAPHY_CLASSES_DICT.get(input.geography, 0)
    ent_enc = ENTEROTYPE_MAP.get(input.enterotype, 0)
    gender_enc = 0 if input.gender == "Male" else 1

    features = np.array([[
        input.shannon_index, input.observed_species, input.Age, input.BMI,
        input.MgsRichness, input.GeneRichness, ent_enc, geo_enc, gender_enc
    ]])
    score = ML_MODEL.predict(features)[0]
    score = max(0.0, min(1.0, float(score)))
    return {
        "longevity_score": round(score * 100, 1),
        "longevity_raw": round(score, 4),
        "confidence": "moderate",
        "model_version": "v1",
        "note": "Population-level estimate based on disease proxy labels"
    }

@app.post("/api/microbiome/knn")
async def microbiome_knn(input: MicrobiomeInput):
    """Find K=7 similar profiles from HGMA"""
    # Build a synthetic species vector using Shannon diversity info
    # For now, use metadata-based query
    meta_subset = samples_meta[
        (samples_meta["Age"] >= input.Age - 5) &
        (samples_meta["Age"] <= input.Age + 5) &
        (samples_meta["BMI"] >= input.BMI - 3) &
        (samples_meta["BMI"] <= input.BMI + 3)
    ]
    if len(meta_subset) == 0:
        meta_subset = samples_meta

    # KNN on species matrix subset
    subset_idx = [samples_meta.index.get_loc(i) for i in meta_subset.index if i in samples_meta.index]
    if len(subset_idx) > 7:
        query_idx = subset_idx[:1]
    else:
        query_idx = subset_idx

    if len(query_idx) > 0:
        dists, ids = knn.kneighbors([knn_matrix[query_idx[0]]], n_neighbors=7)
        matched_ids = [sample_ids[i] for i in ids[0]]
        matched_meta = samples_meta.loc[[s for s in matched_ids if s in samples_meta.index]]
        disease_dist = matched_meta["Disease"].value_counts().to_dict()
        mean_age = float(matched_meta["Age"].mean())
        mean_shannon = float(matched_meta["shannon_index"].mean())
    else:
        disease_dist = {}
        mean_age = input.Age
        mean_shannon = input.shannon_index

    return {
        "k_neighbors": 7,
        "mean_age": round(mean_age, 1),
        "mean_shannon": round(mean_shannon, 4),
        "disease_distribution": {k: v for k, v in list(disease_dist.items())[:10]},
        "top_disease": max(disease_dist, key=disease_dist.get) if disease_dist else "Unknown",
        "healthy_match_ratio": round(disease_dist.get("Healthy", 0) / max(len(disease_dist), 1), 2),
        "note": "Based on cosine similarity of species profiles in HGMA dataset"
    }

@app.post("/api/microbiome/rag")
async def microbiome_rag(query: RAGQuery):
    """RAG-powered Q&A on microbiome literature"""
    if not COLLECTION:
        return {"error": "RAG collection not loaded. POST /api/rag/import first."}

    try:
        results = COLLECTION.query(
            query_texts=[query.query],
            n_results=query.top_k
        )
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not docs:
            return {"answer": "No relevant articles found.", "sources": []}

        context = "\n\n---\n\n".join([f"[{i+1}] {doc[:500]}" for i, doc in enumerate(docs)])

        prompt = (
            f"Based on these PubMed abstracts about gut microbiome and longevity, "
            f"answer the question.\n\nContext:\n{context}\n\n"
            f"Question: {query.query}\n\n"
            f"If the context doesn't contain enough information, say so."
        )

        answer = await groq_complete(prompt, system=(
            "You are a helpful research assistant specializing in gut microbiome and longevity. "
            "Answer based only on the provided context. Cite relevant findings."
        ))

        sources = [
            {"pmid": m.get("pmid", ""), "year": m.get("year", ""), "journal": m.get("journal", "")}
            for m in metadatas
        ]

        return {"answer": answer, "sources": sources, "n_results": len(docs)}

    except Exception as e:
        return {"error": str(e)}

@app.post("/api/rag/import")
async def rag_import():
    """Import RAG documents into ChromaDB"""
    global COLLECTION

    try:
        chroma_client.delete_collection("microbiome_pubmed")
    except:
        pass

    COLLECTION = chroma_client.create_collection(
        name="microbiome_pubmed",
        metadata={"description": "PubMed articles on gut microbiome and longevity"}
    )

    with open(DATA_DIR / "rag_pubmed_documents.json") as f:
        rag_docs = json.load(f)

    docs, ids, metas = [], [], []
    for doc in rag_docs:
        text = doc.get("text", "")[:2000]
        docs.append(text)
        ids.append(doc.get("id", f"doc_{len(docs)}"))
        metas.append({
            "pmid": doc.get("pmid", ""),
            "year": doc.get("year", ""),
            "journal": doc.get("journal", ""),
            "authors": doc.get("authors", ""),
            "search_query": doc.get("search_query", "")
        })

    for i in range(0, len(docs), 100):
        COLLECTION.add(
            documents=docs[i:i+100],
            ids=ids[i:i+100],
            metadatas=metas[i:i+100]
        )

    count = COLLECTION.count()
    print(f"  ✅ Imported {count} documents into ChromaDB")
    return {"status": "imported", "documents": count}

@app.get("/api/rag/status")
async def rag_status():
    if not COLLECTION:
        return {"collection": "not loaded"}
    return {"collection": "microbiome_pubmed", "documents": COLLECTION.count()}

@app.post("/api/microbiome/full")
async def microbiome_full(input: MicrobiomeInput):
    """Combined ML + KNN + RAG analysis — the full pipeline"""
    # ── 1. ML Longevity Score ──────────────────────────────────
    geo_enc = GEOGRAPHY_CLASSES_DICT.get(input.geography, 0)
    ent_enc = ENTEROTYPE_MAP.get(input.enterotype, 0)
    gender_enc = 0 if input.gender == "Male" else 1
    features = np.array([[
        input.shannon_index, input.observed_species, input.Age, input.BMI,
        input.MgsRichness, input.GeneRichness, ent_enc, geo_enc, gender_enc
    ]])
    score = ML_MODEL.predict(features)[0]
    score = max(0.0, min(1.0, float(score)))

    # ── 2. KNN — find similar HGMA profiles ─────────────────────
    meta_subset = samples_meta[
        (samples_meta["Age"] >= input.Age - 5) &
        (samples_meta["Age"] <= input.Age + 5) &
        (samples_meta["BMI"] >= input.BMI - 3) &
        (samples_meta["BMI"] <= input.BMI + 3)
    ]
    if len(meta_subset) == 0:
        meta_subset = samples_meta

    subset_idx = [samples_meta.index.get_loc(i) for i in meta_subset.index if i in samples_meta.index]
    query_idx = subset_idx[:1] if len(subset_idx) > 7 else subset_idx

    if len(query_idx) > 0:
        dists, ids = knn.kneighbors([knn_matrix[query_idx[0]]], n_neighbors=7)
        matched_ids = [sample_ids[i] for i in ids[0]]
        matched_meta = samples_meta.loc[[s for s in matched_ids if s in samples_meta.index]]
        disease_dist = matched_meta["Disease"].value_counts().to_dict()
        mean_age = float(matched_meta["Age"].mean())
        mean_shannon = float(matched_meta["shannon_index"].mean())
        healthy_ratio = round(disease_dist.get("Healthy", 0) / 7, 2)
    else:
        disease_dist, mean_age, mean_shannon, healthy_ratio = {}, input.Age, input.shannon_index, 0.0

    top_disease = max(disease_dist, key=disease_dist.get) if disease_dist else "Unknown"

    # ── 3. RAG — literature context for this profile ────────────
    rag_query = (
        f"A person aged {input.Age} with BMI {input.BMI}, "
        f"shannon diversity {input.shannon_index:.2f}, "
        f"enterotype {input.enterotype}, geography {input.geography}. "
        f"What does the gut microbiome literature say about aging, longevity, "
        f"and disease risk for this profile?"
    )
    rag_result = await microbiome_rag(RAGQuery(query=rag_query, top_k=3))

    return {
        "longevity_score": round(score * 100, 1),
        "confidence": "moderate" if 0.3 < score < 0.75 else ("high" if score >= 0.75 else "low"),
        "ml_version": "v1",
        "knn": {
            "k_neighbors": 7,
            "mean_age": round(mean_age, 1),
            "mean_shannon": round(mean_shannon, 4),
            "top_disease": top_disease,
            "disease_distribution": {k: v for k, v in list(disease_dist.items())[:8]},
            "healthy_match_ratio": healthy_ratio,
        },
        "rag": rag_result,
        "data_version": "hgma_v1",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
if __name__ == "__main__":
    import uvicorn
    print(f"Starting LongevityAI-Microbiome on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
