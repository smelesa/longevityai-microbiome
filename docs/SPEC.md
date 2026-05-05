# LongevityAI-Microbiome — Solution Architecture & Design
**Version:** 1.0 | **Date:** 2026-05-05 | **Status:** Draft for review

---

## 1. Executive Summary

LongevityAI-Microbiome estende la piattaforma LongevityAI esistente con un **nuovo layer dati sul microbioma intestinale**. L'obiettivo: integrare dati del microbioma (species abundance, disease associations) con i modelli ML esistenti (v5 lifestyle, v6 biomarker) per creare un sistema multimodale di predizione della longevità.

**Dati disponibili:**
| Fonte | Tipo | Formato | Accesso |
|-------|------|---------|---------|
| MicrobiomeAtlas.org | Species × Disease (23 malattie, 20 paesi) | CSV + ZIP (download diretto) | `vect_atlas.csv.gz`, `sampleID.csv` |
| PubMed (E-utilities) | Articoli su microbiome-longevity | JSON via API | ESearch + EFetch (gratuito, rate limit 3/sec) |
| HGMA esistente | RAG knowledge base (1,876 chunks) | ChromaDB | Già nel sistema |

---

## 2. Data Sources — Deep Dive

### 2.1 MicrobiomeAtlas.org (HGMA)

**Cosa contiene:**
- **vect_atlas.csv.gz** — Matrice species abundance (MGS = Metagenomic Species)
- **sampleID.csv** — Metadata dei campioni (paese, salute/malattia)
- **MSP_GEM_models.zip** — Genome-scale metabolic models
- **23 malattie** associate a specie (disease-associated species)
- **20 paesi** con region-enriched species
- Specie chiave: *Akkermansia muciniphila*, *Bifidobacterium*, butyrate-producers, *Christensenellaceae*

**Approccio consigliato: ML (KNN + Gradient Boosting)**
- I dati strutturati (species × disease matrix) si prestano a **KNN** per trovare profile simili
- **XGBoost/LightGBM** per predire longevity score basato su microbial features
- Feature set: abbondanza relativa delle specie chiave + alpha-diversity + enterotype
- KNN trova i "vicini" più simili nel dataset HGMA → suggerisce interventi basati su chi ha il profilo simile

**Limiti:**
- Dati aggregati per paese/malattia, non a livello individuale
- Non include dati longitudinali

---

### 2.2 PubMed (NCBI E-utilities API)

**Cosa contiene:**
- Articoli su: microbiome aging, Akkermansia, butyrate, FMT, probiotic interventions
- Dati clinici su: centenarios, gut-brain axis, inflammation
- Aging clocks basati su microbiome

**Come accedere:**
```
ESearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=microbiome+longevity&datetype=pdat&mindate=2020
EFetch:  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=PMID1,PMID2&rettype=abstract
```

**Approccio consigliato: RAG**
- E-utilities restituisce articoli → chunking → vector embedding → ChromaDB
- RAG Synthesizer risponde a domande utente con citazioni da letteratura
- Pipeline: ESearch → EFetch → chunk(abstract+fulltext) → embed → store → query

**Quando usare ML vs RAG su PubMed:**
- **ML:** Non adatto — PubMed è letteratura, non dataset strutturato
- **RAG:** Perfetto — cerca risposte in abstract, genera insight con citazioni

**Limiti:**
- Rate limit NCBI: max 3 req/sec (pausa 1 secondo tra chiamate)
- Abstract non sempre contengono tutti i dettagli necessari

---

## 3. Architecture — Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                       │
│  - Input form: microbiome species abundances (slider/input) │
│  - Visualizzazione: radar chart, longevity score            │
│  - Suggerimenti personalizzati (KNN profile matches)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  API GATEWAY (VPS :8000-8502)               │
│  - /api/microbiome     → ML inference + KNN               │
│  - /api/microbiome-rag → RAG synthesis                     │
│  - /api/analyze        → Full pipeline (v5+v6+microbiome)  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ML LAYER      │  │   RAG LAYER     │  │   KNN LAYER     │
│  XGBoost/LightGBM│  │  ChromaDB       │  │  Profile Match  │
│  - longevity score│  │  - PubMed chunks│  │  - Find similar │
│  - bio age delta │  │  - HGMA data    │  │  - Recommend    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│               DATA LAYER (VPS storage)                      │
│  - MicrobiomeAtlas CSV (species × disease matrix)          │
│  - PubMed chunked articles                                 │
│  - HGMA vector store                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Component Approach

### 4.1 ML Component (XGBoost/LightGBM)

**Quando:** Predire longevity score o biologica età da bacterial features

**Input features (approccio microbiome-first):**
| Feature | Tipo | Fonte |
|---------|------|-------|
| `akkermansia_ratio` | float | Relative abundance di A. muciniphila |
| `bifidobacterium_ratio` | float | Relative abundance |
| `butyrate_producers_ratio` | float | Sum of butyrate-producing species |
| `alpha_diversity` | float | Shannon/Simpson index |
| `proteobacteria_ratio` | float | Inflammatory markers |
| `enterotype` | categorical | 1= Bacteroides, 2= Prevotella, 3= Ruminococcaceae |
| `disease_risk_profile` | array[23] | HGMA disease species match |
| `geography_factor` | categorical | Country/region encoding |

**Output:**
- `longevity_score` (0-100)
- `biological_age_delta` (years ahead/behind)
- `microbiome_age` prediction

**Modello preferito:** LightGBM (veloce, gestisce missing values, feature importance nativa)

**Perché non KNN per predizione:** KNN è ottimo per trovare profili simili, ma per predizione numerica XGBoost è più accurato su feature engineering.

---

### 4.2 KNN Component

**Quando:** Raccomandazioni personalizzate basate su profile simili

**Come funziona:**
1. Costruisci vettore microbioma utente: `[akke=0.05, bifido=0.02, butyrate=0.15, ...]`
2. Trova i K campioni più simili nel dataset HGMA (cosine similarity)
3. Estrai gli interventi/outcomes associati ai profili simili
4. Genera raccomandazioni: "Profili simili al tuo hanno risposto positivamente a..."

**KNN applicazioni:**
- **Raccomandazione interventi:** "Akkermansia basso + Bifidobacterium basso → Integratore specifico"
- **Food recommendations:** "Profili simili al tuo consumano più fibre fermentabili"
- **Supplementi:** "Centenarians con il tuo profilo assumono questi probiotici"

**Parametri:**
- K = 5-10 (troppo basso = overfitting, troppo alto = too generic)
- Distance: cosine similarity (ottimo per vettori di abbondanza)
- Weighting: inverse distance weighting per dare più peso ai profili più simili

---

### 4.3 RAG Component

**Quando:** Domande scientifiche, spiegazioni, citazioni da letteratura

**Pipeline:**
```
User question → Embed → ChromaDB similarity search → Context → LLM → Answer + Citations
```

**Collections:**
1. `microbiome_pubmed` — articoli PubMed chunkati (abstract + key findings)
2. `microbiome_hgma` — HGMA disease-species associations, geographic data

**Esempi d'uso:**
- "Cosa dice la scienza su Akkermansia e longevità?"
- "Quali interventi sul microbioma sono supportati dalla ricerca?"
- "Come influisce la geografia sul microbioma intestinale?"

**LLM:** Groq llama-3.3-70b (stesso usato nel RAG esistente) con temp=0.3, max_tokens=1024

---

## 5. Data Flow

### 5.1 Data Ingestion

```
MicrobiomeAtlas.org ──download──> vect_atlas.csv.gz ──parse──> ChromaDB (microbiome_hgma)
                                          │
                                          └──> species_disease_matrix.csv (structured)

PubMed E-utilities ──ESearch──> PMID list ──EFetch──> articles.json ──chunk──> ChromaDB (microbiome_pubmed)
```

### 5.2 Inference Pipeline

```
User: "Ho Akkermansia 3%, Bifidobacterium 1%, come posso migliorare?"

Step 1: KNN Match
         → Find similar profiles in HGMA
         → "Profili con Akkermansia 2-5% + Bifidobacterium <2% ->干预 X"

Step 2: ML Score (se utente fornisce più dati)
         → longevity_score = model.predict(microbial_features)

Step 3: RAG Synthesis
         → Query: "Akkermansia supplementation longevity evidence"
         → Return: "Studies show A. muciniphila supplementation increases Mucosal barrier..."
         → Citations: [PMID1, PMID2]

Step 4: Combine & Respond
         → KNN recommendation + RAG evidence + ML score (if available)
```

---

## 6. Frontend Requirements

**Input fields per microbioma:**
| Campo | Tipo | Note |
|-------|------|------|
| `akkermansia_pct` | slider 0-100% | Abbondanza relativa stimata |
| `bifidobacterium_pct` | slider 0-100% | |
| `butyrate_producers_pct` | slider 0-100% | Combinazione specie multiple |
| `alpha_diversity` | slider 0-10 | Shannon index stimato |
| `enterotype` | select | Bacteroides/Prevotella/Ruminococcaceae |
| `country` | select | Per geografia |
| `known_diseases` | multi-select | 23 malattie HGMA |
| `symptoms` | text | Optional context |

**Output display:**
- **Radar Chart:** Microbiome composition vs healthy range
- **Longevity Score:** 0-100 con benchmark per età
- **KNN Matches:** Top 3 profili simili con interventi
- **RAG Insights:** Evidence-based recommendations
- **Bio Age Delta:** "Il tuo microbioma suggerisce +3 anni di vita residua rispetto a età cronologica"

**Consistency check (FRONTEND VALIDATION):**
Prima di mostrare output, valida che i dati richiesti dal frontend siano effettivamente disponibili:
- ❌ Se frontend chiede `alpha_diversity` ma il modello ML non lo usa → allineare
- ❌ Se RAG chiede articoli PubMed non ancora scaricati → avvisare utente
- ✅ Definire contract: "Questo campo → questo modello/knowledge base"

---

## 7. Integration with Existing LongevityAI Platform

```
LongevityAI v5 (lifestyle)     → /api/analyze?type=lifestyle
LongevityAI v6 (biomarker)     → /api/analyze?type=biomarker  
LongevityAI-Microbiome (NEW)   → /api/analyze?type=microbiome
                                       │
                                       ▼
                              Multi-modal analysis
                              (v5 lifestyle + v6 biomarkers + microbiome)
```

**API endpoint unificato:**
```
POST /api/analyze/full
{
  "lifestyle": { "work_hours": 45, "sleep_hours": 6.5, ... },
  "biomarkers": { "hba1c": 5.8, "hdl": 65, ... },
  "microbiome": { "akkermansia_pct": 3.2, "bifidobacterium_pct": 1.1, ... }
}
→ Returns: { longevity_score, bio_age_delta, recommendations, citations }
```

---

## 8. Technology Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Frontend | Next.js 16 (existing) | Add microbiome input form |
| API Server | Python/FastAPI (new service on VPS) | Port 8030 |
| ML | LightGBM + XGBoost | Train on HGMA + PubMed data |
| KNN | scikit-learn | Use HGMA species matrix |
| RAG | ChromaDB + Groq | Extend existing RAG pipeline |
| Data | CSV (HGMA) + JSON (PubMed) | Local storage on VPS |
| Deploy | Docker on VPS | New container |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| MicrobiomeAtlas data is aggregate, not individual | Use as population-level priors; KNN maps to closest profile |
| PubMed rate limit (3/sec) | Batch requests with delays; cache results |
| User can't easily get their microbiome data | Provide manual input (slider/select) + link to tests |
| Model overfitting | Cross-validation; keep KNN K reasonable (5-10) |
| Missing ground truth labels | Use proxy labels: centenarian microbiome studies → "high longevity" |

---

## 10. Next Steps

1. **Download HGMA data** → Parse `vect_atlas.csv.gz` → Build species × disease matrix
2. **Scrape PubMed** → Query E-utilities per "microbiome longevity" → chunk → store
3. **Train ML model** → LightGBM su species features → longevity prediction
4. **Build KNN index** → Use HGMA profiles → cosine similarity
5. **Extend RAG** → Add microbiome collections → test queries
6. **Frontend** → Add microbiome form → validate input-output contract
7. **Integrate** → Connect new service to existing `/api/analyze`

---

**Decision required:** Vuoi procedere con tutte le fasi in parallelo o una alla volta partendo dallo scraping HGMA + PubMed?