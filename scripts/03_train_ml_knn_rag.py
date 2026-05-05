#!/usr/bin/env python3
"""
LongevityAI-Microbiome — Phase 3: ML Training + KNN Index + RAG Setup
Builds longevity prediction model + KNN similarity index + ChromaDB RAG
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
import os

# Try to import ML libraries; fall back gracefully
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not available — will build data structures only")

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("⚠️ chromadb not available — RAG will be set up on VPS")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

print("=" * 60)
print("LongevityAI-Microbiome — Phase 3: ML + KNN + RAG")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# Load processed data
# ============================================================
print("\n[1/5] Loading processed data...")
samples = pd.read_csv(PROCESSED_DIR / "hgma_samples_enriched.csv", index_col=0)
species_matrix = pd.read_csv(PROCESSED_DIR / "hgma_species_matrix.csv", index_col=0)
articles_df = pd.read_csv(PROCESSED_DIR / "pubmed_microbiome_articles.csv")
print(f"  Samples: {len(samples)}")
print(f"  Species: {species_matrix.shape[1]}")
print(f"  PubMed articles: {len(articles_df)}")

# ============================================================
# FEATURE ENGINEERING for ML
# ============================================================
print("\n[2/5] Feature engineering...")

# Create longevity proxy label:
# Healthy samples = high longevity
# Disease samples = reduced longevity
# Based on: healthy→1, NGT→0.9, IGT→0.7, T2D→0.5, CRC→0.4, etc.

disease_longevity_map = {
    "Healthy": 1.0,
    "NGT": 0.95,
    "IGT": 0.75,
    "T2D": 0.65,
    "T1D": 0.55,
    "NAFLD": 0.65,
    "LC": 0.50,
    "ACVD": 0.45,
    "atherosclerosis": 0.45,
    "RA": 0.70,
    "CD": 0.55,
    "UC": 0.65,
    "CRC": 0.40,
    "Adenoma": 0.70,
    "RCC": 0.45,
    "NSCLC": 0.40,
    "melanoma": 0.50,
    "SPA": 0.70,
    "VKH": 0.55,
    "BD": 0.65,
    "PD": 0.55,
    "ME/CFS": 0.60,
    "GDM": 0.75
}

samples["longevity_label"] = samples["Disease"].map(disease_longevity_map)
samples["longevity_label"] = samples["longevity_label"].fillna(0.5)

# Features: diversity metrics + enterotype + age + BMI
# We'll use aggregated species data via diversity metrics
features_df = samples[[
    "shannon_index", "observed_species", "Age", "BMI", 
    "MgsRichness", "GeneRichness"
]].copy()

# Encode enterotype
entero_map = {"ET-Bacteroides": 0, "ET-Firmicutes": 1, "ET-Prevotella": 2}
samples["enteroType_num"] = samples["enteroType"].map(entero_map).fillna(0)
features_df["enteroType_num"] = samples["enteroType_num"]

# Geography encoding
geo_encoder = LabelEncoder()
features_df["geography_enc"] = geo_encoder.fit_transform(samples["Geography"].fillna("Unknown"))

# Gender encoding
features_df["gender_enc"] = samples["Gender"].map({"Male": 0, "Female": 1}).fillna(0)

# Handle missing
features_df = features_df.fillna(features_df.median())

print(f"  Features: {list(features_df.columns)}")
print(f"  Feature matrix shape: {features_df.shape}")

# ============================================================
# ML MODEL: Gradient Boosting for Longevity Score
# ============================================================
if HAS_SKLEARN and len(samples) > 100:
    print("\n[3/5] Training ML model...")
    
    X = features_df.values
    y = samples["longevity_label"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"  Model: GradientBoostingRegressor")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²:  {test_score:.4f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    feat_names = list(features_df.columns)
    importances = model.feature_importances_
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    print(f"\n  Feature importances:")
    for name, imp in feat_imp[:6]:
        print(f"    {name}: {imp:.4f}")
    
    # Save model metadata (actual model joblib on VPS)
    model_info = {
        "model_type": "GradientBoostingRegressor",
        "features": feat_names,
        "n_features": len(feat_names),
        "train_r2": round(train_score, 4),
        "test_r2": round(test_score, 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "feature_importances": {name: round(imp, 4) for name, imp in feat_imp},
        "n_samples_trained": len(X_train),
        "disease_longevity_map": disease_longevity_map,
        "enterotype_map": entero_map,
        "geography_classes": list(geo_encoder.classes_)
    }
    
    with open(PROCESSED_DIR / "microbiome_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n  ✅ Model info saved to microbiome_model_info.json")

# ============================================================
# KNN INDEX: Build from HGMA data
# ============================================================
print("\n[4/5] Building KNN index...")

if HAS_SKLEARN and len(samples) > 100:
    # Use species matrix for KNN (high-dimensional)
    # Scale species data
    species_vals = species_matrix.values
    
    # Normalize
    row_sums = species_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    species_norm = species_vals / row_sums
    
    # Replace NaN/Inf
    species_norm = np.nan_to_num(species_norm, nan=0, posinf=0, neginf=0)
    
    # Fit KNN (K=7, cosine-like with normalized vectors)
    knn = NearestNeighbors(n_neighbors=7, metric="cosine", algorithm="brute")
    knn.fit(species_norm)
    
    print(f"  KNN model fitted: K=7, metric=cosine")
    print(f"  Index shape: {species_norm.shape}")
    
    # Save KNN data + metadata
    np.save(PROCESSED_DIR / "knn_species_matrix.npy", species_norm)
    species_matrix.index.to_series().to_csv(PROCESSED_DIR / "knn_sample_ids.csv", index=False)
    
    # Create healthy profiles for KNN recommendations
    healthy_samples = samples[samples["Disease"] == "Healthy"].copy()
    healthy_meta = healthy_samples[["Age", "Gender", "BMI", "Geography", 
                                    "shannon_index", "observed_species", "enteroType"]].copy()
    healthy_meta.to_csv(PROCESSED_DIR / "knn_healthy_reference.csv")
    
    print(f"  ✅ KNN index saved")
    print(f"     - knn_species_matrix.npy")
    print(f"     - knn_sample_ids.csv")
    print(f"     - knn_healthy_reference.csv")
    print(f"     - {len(healthy_samples)} healthy reference samples")

# ============================================================
# RAG: Create ChromaDB collection from PubMed articles
# ============================================================
print("\n[5/5] Setting up RAG knowledge base...")

if HAS_CHROMA and len(articles_df) > 0:
    try:
        # Create ChromaDB client
        chroma_path = PROCESSED_DIR / "chroma_microbiome"
        chroma_path.mkdir(exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(chroma_path))
        
        # Drop existing collection if any
        try:
            client.delete_collection("microbiome_pubmed")
        except:
            pass
        
        collection = client.create_collection(
            name="microbiome_pubmed",
            metadata={"description": "PubMed articles on gut microbiome and longevity"}
        )
        
        # Chunk articles and add to collection
        docs = []
        ids = []
        metadatas = []
        
        for idx, row in articles_df.iterrows():
            pmid = str(row["pmid"])
            title = str(row.get("title", "")) or ""
            abstract = str(row.get("abstract", "")) or ""
            
            # Combine title + abstract as document
            text = f"Title: {title}\n\nAbstract: {abstract}"
            
            docs.append(text[:2000])  # Chunk limit
            ids.append(f"pmid_{pmid}")
            metadatas.append({
                "pmid": pmid,
                "year": str(row.get("year", "")) or "unknown",
                "journal": str(row.get("journal", "")) or "",
                "authors": str(row.get("authors", "")) or "",
                "search_query": str(row.get("search_query", "")) or ""
            })
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            try:
                collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)
            except Exception as e:
                print(f"  ⚠️ Batch error at {i}: {e}")
        
        count = collection.count()
        print(f"  ✅ ChromaDB collection 'microbiome_pubmed' created")
        print(f"     Documents: {count}")
        print(f"     Path: {chroma_path}")
    
    except Exception as e:
        print(f"  ⚠️ ChromaDB setup failed: {e}")
        print(f"     → RAG will be set up on VPS")

else:
    print(f"  ℹ️ ChromaDB library not available locally")
    print(f"     → RAG setup deferred to VPS deployment")

# ============================================================
# Create RAG-ready JSON (for VPS import)
# ============================================================
print("\n[Extra] Creating RAG-ready data package...")

rag_data = []
for _, row in articles_df.iterrows():
    rag_data.append({
        "id": f"pmid_{row['pmid']}",
        "text": f"Title: {row.get('title','')}\n\nAbstract: {row.get('abstract','')}",
        "pmid": str(row.get('pmid', '')),
        "year": str(row.get('year', '')),
        "journal": str(row.get('journal', '')),
        "authors": str(row.get('authors', '')),
        "search_query": str(row.get('search_query', ''))
    })

with open(PROCESSED_DIR / "rag_pubmed_documents.json", "w") as f:
    json.dump(rag_data, f, indent=2)

print(f"  ✅ RAG data package: rag_pubmed_documents.json ({len(rag_data)} docs)")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)
print(f"✅ ML Model: GradientBoostingRegressor trained on {len(samples)} samples")
print(f"✅ KNN Index: {species_matrix.shape[0]} samples × {species_matrix.shape[1]} species")
print(f"✅ RAG: {len(articles_df)} PubMed articles ready for ChromaDB")
print(f"\nAll files in: {PROCESSED_DIR}/")

print("\n📋 File inventory:")
for f in sorted(PROCESSED_DIR.iterdir()):
    size = f.stat().st_size / 1024
    print(f"  {f.name:45} {size:8.1f} KB")

print(f"\n✅ Phase 3 COMPLETE at {datetime.now().strftime('%H:%M:%S')}")