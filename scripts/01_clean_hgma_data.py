#!/usr/bin/env python3
"""
LongevityAI-Microbiome — Phase 1 Data Processing
Cleans and structures HGMA (Human Gut Microbiome Atlas) data
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LongevityAI-Microbiome — Phase 1: Data Cleaning")
print("=" * 60)

# ============================================================
# 1. Load sample metadata
# ============================================================
print("\n[1/4] Loading sample metadata...")
sample_df = pd.read_csv(RAW_DIR / "sampleID.csv")
print(f"  Samples: {len(sample_df)}")
print(f"  Columns: {list(sample_df.columns)}")
print(f"  Diseases: {sample_df['Disease'].nunique()} unique values")
print(f"  Countries: {sample_df['Geography'].nunique()} unique")

# ============================================================
# 2. Identify key longevity-associated species
# Based on PubMed research:
#   - Akkermansia muciniphila → longevity marker
#   - Bifidobacterium → healthy aging
#   - Butyrate producers → gut health
#   - Christensenellaceae → metabolic health
#   - Faecalibacterium → anti-inflammatory
# ============================================================
print("\n[2/4] Loading species abundance matrix...")
vect_df = pd.read_csv(RAW_DIR / "vect_atlas.csv", index_col=0)
print(f"  Species (MGS): {vect_df.shape[0]}")
print(f"  Samples: {vect_df.shape[1]}")

# Transpose: rows = samples, cols = species
species_matrix = vect_df.T
species_matrix.index.name = "sample.ID"
print(f"  Matrix shape: {species_matrix.shape} (samples × species)")

# ============================================================
# 3. Extract key species for longevity model
# ============================================================
print("\n[3/4] Extracting longevity-associated species...")

# Strategy: compute per-sample aggregates for known beneficial/detrimental groups
# We don't have species names mapping yet (msp_0001 etc.), so we'll use
# aggregate statistics as features + build a species lookup

# Calculate per-sample diversity metrics
sample_diversity = pd.DataFrame(index=species_matrix.index)
sample_diversity["shannon_index"] = species_matrix.apply(
    lambda row: -np.sum(row * np.log(row + 1e-12)), axis=1
)
sample_diversity["observed_species"] = (species_matrix > 0).sum(axis=1)
sample_diversity["total_abundance"] = species_matrix.sum(axis=1)

# Merge with metadata
merged = sample_diversity.join(sample_df.set_index("sample.ID"), how="inner")
print(f"  Merged dataset: {merged.shape[0]} samples × {merged.shape[1]} features")

# ============================================================
# 4. Create disease risk profile (23 diseases)
# ============================================================
print("\n[4/4] Building disease risk profiles...")

# One-hot encode disease status
disease_dummies = pd.get_dummies(merged["Disease"], prefix="disease")
merged = pd.concat([merged, disease_dummies], axis=1)

# Save processed datasets
merged.to_csv(PROCESSED_DIR / "hgma_samples_enriched.csv")
species_matrix.to_csv(PROCESSED_DIR / "hgma_species_matrix.csv")
sample_df.to_csv(PROCESSED_DIR / "hgma_sample_metadata.csv", index=False)

print(f"\n✅ Saved to {PROCESSED_DIR}/")
print(f"   - hgma_samples_enriched.csv ({merged.shape[0]} rows)")
print(f"   - hgma_species_matrix.csv ({species_matrix.shape[0]} samples × {species_matrix.shape[1]} species)")
print(f"   - hgma_sample_metadata.csv")

# ============================================================
# 5. Create species name mapping (from row index)
# ============================================================
species_list = pd.DataFrame({
    "mgs_id": vect_df.index,
    "mgs_index": range(len(vect_df))
})
species_list.to_csv(PROCESSED_DIR / "species_list.csv", index=False)
print(f"   - species_list.csv ({len(species_list)} species)")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total samples: {len(merged)}")
print(f"Total species: {species_matrix.shape[1]}")
print(f"Diseases in dataset: {sorted(merged['Disease'].unique())}")
print(f"\nDataset breakdown:")
print(merged['Disease'].value_counts().to_string())
print(f"\nAge range: {merged['Age'].min()} - {merged['Age'].max()} years")
print(f"Countries: {sorted(merged['Geography'].unique())}")

# ============================================================
# For KNN: build healthy reference profiles
# ============================================================
print("\n[Extra] Building healthy reference profiles for KNN...")
healthy = merged[merged["Disease"] == "Healthy"]
print(f"  Healthy samples: {len(healthy)}")
print(f"  Mean Shannon index (healthy): {healthy['shannon_index'].mean():.3f}")

healthy_profiles = healthy.groupby("Geography").agg({
    "shannon_index": "mean",
    "observed_species": "mean",
    "total_abundance": "mean",
    "Age": "mean"
}).round(3)
healthy_profiles.to_csv(PROCESSED_DIR / "healthy_reference_by_country.csv")
print(f"  Saved: healthy_reference_by_country.csv")

print("\n✅ Phase 1 (HGMA data cleaning) COMPLETE")