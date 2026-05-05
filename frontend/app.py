"""
LongevityAI Microbiome — Streamlit Frontend
Complete ML + KNN + RAG + LLM pipeline with developer view
"""
import streamlit as st
import requests
import time
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────

API_BASE = "https://microbiome.srv1424731.hstgr.cloud"

# ── Presets ──────────────────────────────────────────────────────────────────

PRESETS = {
    "Italian Male 40 — Healthy": {
        "shannon_index": 4.5, "observed_species": 420, "Age": 40,
        "BMI": 23.5, "MgsRichness": 195, "GeneRichness": 13500,
        "enterotype": "ET-Bacteroides", "geography": "Switzerland", "gender": "Male",
    },
    "Swiss Female 55 — Average": {
        "shannon_index": 3.8, "observed_species": 310, "Age": 55,
        "BMI": 26.0, "MgsRichness": 155, "GeneRichness": 11000,
        "enterotype": "ET-Bacteroides", "geography": "Switzerland", "gender": "Female",
    },
    "European Male 65 — At Risk": {
        "shannon_index": 2.9, "observed_species": 220, "Age": 65,
        "BMI": 29.5, "MgsRichness": 120, "GeneRichness": 8500,
        "enterotype": "ET-Prevotella", "geography": "Europe", "gender": "Male",
    },
    "Research Profile — Max Diversity": {
        "shannon_index": 5.2, "observed_species": 580, "Age": 35,
        "BMI": 22.0, "MgsRichness": 230, "GeneRichness": 15500,
        "enterotype": "ET-Bacteroides", "geography": "Switzerland", "gender": "Male",
    },
    "Custom": None,
}

PRESET_DESCRIPTIONS = {
    "Italian Male 40 — Healthy": "High diversity, healthy BMI, Swiss residence — optimal microbiome profile",
    "Swiss Female 55 — Average": "Moderate diversity, slightly elevated BMI — typical middle-aged profile",
    "European Male 65 — At Risk": "Low diversity, higher BMI, Prevotella-dominant — inflammatory enterotype",
    "Research Profile — Max Diversity": "Exceptional diversity, optimal metrics — research-grade baseline",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def call_full_pipeline(payload: dict, question: str) -> dict:
    payload["question"] = question
    start = time.time()
    r = requests.post(f"{API_BASE}/api/microbiome/full", json=payload, timeout=30)
    r.raise_for_status()
    elapsed_ms = (time.time() - start) * 1000
    result = r.json()
    result["_inference_ms"] = round(elapsed_ms, 1)
    return result


def score_label(score: float) -> tuple:
    if score >= 75: return ("Excellent", "🟢")
    if score >= 60: return ("Good", "🟡")
    if score >= 45: return ("Moderate", "🟠")
    return ("Low", "🔴")


def knn_confidence(healthy_ratio: float) -> tuple:
    if healthy_ratio >= 0.6: return ("High", "🟢")
    if healthy_ratio >= 0.3: return ("Moderate", "🟡")
    if healthy_ratio >= 0.15: return ("Low", "🟠")
    return ("Very Low", "🔴")


# ── Scorecard components ──────────────────────────────────────────────────────

def scorecard(title: str, value: str, subtitle: str, emoji: str, color: str):
    """Renders a formatted scorecard block."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.07);
    ">
        <div style="font-size:13px; text-transform:uppercase; letter-spacing:1.5px;
                    color: #888; margin-bottom:6px;">{title}</div>
        <div style="font-size:42px; font-weight:700; color:{color}; line-height:1;">
            {emoji} {value}
        </div>
        <div style="font-size:13px; color:#aaa; margin-top:8px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(emoji: str, title: str, description: str):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin:24px 0 12px 0;">
        <span style="font-size:22px;">{emoji}</span>
        <div>
            <div style="font-size:16px; font-weight:600; color:#e0e0e0;">{title}</div>
            <div style="font-size:12px; color:#777;">{description}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def dev_metric(label: str, value: str, help_text: str = ""):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**{label}**")
    with col2:
        st.markdown(f"`{value}`")
    if help_text:
        st.caption(help_text)


def pill(text: str, color: str):
    st.markdown(f"<span style='background:{color}; padding:2px 10px; "
                f"border-radius:12px; font-size:11px; color:white;'>{text}</span>",
                unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LongevityAI — Microbiome",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "dev_view" not in st.session_state:
    st.session_state["dev_view"] = False
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "inference_start" not in st.session_state:
    st.session_state["inference_start"] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
    border-radius: 0 0 24px 24px;
    padding: 20px 32px 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <div style="font-size:28px; font-weight:700; color:white;">
        🧬 LongevityAI — Microbiome Analysis
    </div>
    <div style="font-size:13px; color:#aaa; margin-top:6px;">
        HGMA-powered · ML + KNN + RAG + Groq LLM · 6,014 reference microbiomes
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Preset selector
    preset_names = list(PRESETS.keys())
    selected_preset = st.selectbox("Profile Preset", preset_names)

    st.divider()

    # Parameter inputs (if Custom or always editable)
    st.markdown("**Parameters**")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=40)
        BMI = st.number_input("BMI", min_value=15.0, max_value=50.0, value=23.5, step=0.5)
        Shannon = st.number_input("Shannon Index", min_value=0.0, max_value=7.0, value=4.5, step=0.1)
    with col2:
        ObsSp = st.number_input("Observed Species", min_value=50, max_value=1000, value=420, step=10)
        MgsR = st.number_input("MGS Richness", min_value=50, max_value=400, value=195, step=5)
        GeneR = st.number_input("Gene Richness", min_value=5000, max_value=25000, value=13500, step=500)

    gender = st.selectbox("Gender", ["Male", "Female"])
    enterotype = st.selectbox("Enterotype", [
        "ET-Bacteroides", "ET-Prevotella", "ET-Firmicutes", "ET-Alistipes"
    ])
    geography = st.selectbox("Geography", [
        "Switzerland", "Europe", "USA", "Asia", "Other"
    ])

    st.divider()

    # Developer mode toggle
    dev_mode = st.toggle("🔧 Developer View", value=st.session_state["dev_view"])
    st.session_state["dev_view"] = dev_mode

    st.divider()

    # Question
    question = st.text_area(
        "Question to LLM",
        value="What does my microbiome tell me about my health and longevity?",
        height=80,
    )

    # Run button
    run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    if dev_mode:
        st.caption("Developer view enabled — intermediate results visible below.")

# ── Payload assembly ──────────────────────────────────────────────────────────

payload = {
    "shannon_index": Shannon,
    "observed_species": ObsSp,
    "Age": Age,
    "BMI": BMI,
    "MgsRichness": MgsR,
    "GeneRichness": GeneR,
    "enterotype": enterotype,
    "geography": geography,
    "gender": gender,
}

# ── Run pipeline ─────────────────────────────────────────────────────────────

if run:
    st.session_state["inference_start"] = time.time()
    try:
        with st.spinner("Running ML + KNN + RAG + LLM pipeline..."):
            result = call_full_pipeline(payload, question)
            result["_inference_ms"] = round(
                (time.time() - st.session_state["inference_start"]) * 1000, 1
            )
        st.session_state["last_result"] = result
        st.success("Analysis complete!")
    except Exception as e:
        st.error(f"API error: {e}")
        st.session_state["last_result"] = None

# ── Render results ────────────────────────────────────────────────────────────

result = st.session_state.get("last_result")

if not result:
    st.info("Configure parameters and click **Run Full Analysis** to begin.")
    st.stop()

# ─── MAIN OUTPUT ─────────────────────────────────────────────────────────────

st.markdown("---")
section_header("🧬", "LLM Interpretation", "Groq LLM response based on ML + KNN + RAG pipeline")

# Scorecard row
col_sc = st.columns(3)

# Longevity score
ls = result.get("longevity_score", 0)
label, color = ("Excellent", "#00c853") if ls >= 75 else \
               ("Good", "#76ff03") if ls >= 60 else \
               ("Moderate", "#ffab00") if ls >= 45 else ("Low", "#ff5252")

with col_sc[0]:
    scorecard(
        "Longevity Score",
        f"{ls:.1f}/100",
        f"{label} — based on ML disease-proxy model",
        "📊", color
    )

# KNN healthy match
knn = result.get("knn", {})
hm = knn.get("healthy_match_ratio", 0)
hml = "High" if hm >= 0.6 else "Moderate" if hm >= 0.3 else "Low" if hm >= 0.15 else "Very Low"
hmc = "#00c853" if hm >= 0.6 else "#76ff03" if hm >= 0.3 else "#ffab00" if hm >= 0.15 else "#ff5252"

with col_sc[1]:
    scorecard(
        "KNN Healthy Match",
        f"{hm:.0%}",
        f"{hml} — {knn.get('k_neighbors', '?')} similar profiles in HGMA",
        "🔬", hmc
    )

# Inference time
inf_ms = result.get("_inference_ms", 0)
inf_label = "⚡ Fast" if inf_ms < 2000 else "🐢 Slow"

with col_sc[2]:
    scorecard(
        "Inference Time",
        f"{inf_ms:.0f}ms",
        "End-to-end ML+KNN+RAG+LLM",
        inf_label, "#42a5f5"
    )

# LLM response
st.markdown("#### 💬 LLM Response")
llm_bg = """
<div style="
    background: linear-gradient(135deg, #1e2a3a 0%, #0f1923 100%);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border-left: 4px solid #42a5f5;
    line-height:1.7;
    font-size:15px;
    color: #e8eaf0;
">
{}
</div>
"""
answer = result.get("rag", {}).get("answer", "No answer returned.")
st.markdown(llm_bg.format(answer), unsafe_allow_html=True)

# RAG sources
rag_data = result.get("rag", {})
sources = rag_data.get("sources", [])
if sources:
    st.markdown("**📚 References**")
    for s in sources:
        st.markdown(f"  • PMID `{s.get('pmid', '?')}` — {s.get('journal', '?')} ({s.get('year', '?')})")

# ─── DEVELOPER VIEW ──────────────────────────────────────────────────────────

if st.session_state["dev_view"]:
    st.markdown("---")
    st.markdown("## 🔧 Developer View")

    # ML section
    with st.expander("📊 ML Model — Longevity Score", expanded=True):
        col_ml = st.columns(2)
        with col_ml[0]:
            st.metric("Longevity Score", f"{result.get('longevity_score', '?')}/100")
            st.metric("Model Version", result.get("ml_version", "unknown"))
            st.metric("Confidence", result.get("confidence", "unknown"))
        with col_ml[1]:
            st.markdown("**How it is calculated**")
            st.capsule("""
The ML model (GradientBoostingRegressor) is trained on the Human Gut Microbiome Atlas (HGMA) — 6,014 samples × 1,990 species.
It uses disease proxy labels (CRC, Adenoma, T2D, etc.) as the target variable. The longevity score (0–100) is a population-relative ranking:
scores above 60 indicate a microbiome profile similar to healthy individuals in the HGMA dataset.
            """)
        st.markdown(f"**Model info:** `{result.get('data_version', '?')}`")
        st.caption("ML pipeline: normalize species vector → GradientBoostingRegressor → map to 0-100 scale")

    # KNN section
    knn = result.get("knn", {})
    with st.expander("🔬 KNN — Nearest Microbiome Neighbors", expanded=True):
        col_knn = st.columns(2)
        with col_knn[0]:
            st.metric("K Neighbors", knn.get("k_neighbors", "?"))
            st.metric("Mean Age of Neighbors", f"{knn.get('mean_age', '?')}")
            st.metric("Healthy Match Ratio", f"{knn.get('healthy_match_ratio', '?'):.2%}")
            st.metric("Top Disease in Neighbors", knn.get("top_disease", "?"))
        with col_knn[1]:
            st.markdown("**How it is calculated**")
            st.capsule("""
Cosine similarity is computed between the input species vector and all 6,014 HGMA profiles.
The K=7 nearest neighbors (by cosine similarity) are selected.
Each neighbor carries a disease label. The healthy_match_ratio is the fraction of neighbors labeled 'Healthy'.
Lower ratios indicate the input profile is more similar to diseased populations in HGMA.
            """)
        st.markdown("**Disease distribution of neighbors:**")
        dd = knn.get("disease_distribution", {})
        if dd:
            cols_d = st.columns(len(dd))
            for i, (disease, count) in enumerate(dd.items()):
                with cols_d[i]:
                    st.metric(disease, f"{count}/7")
        st.caption(f"Mean Shannon diversity of neighbors: {knn.get('mean_shannon', '?')}")

    # RAG section
    rag_data = result.get("rag", {})
    with st.expander("📚 RAG — PubMed Literature Retrieval", expanded=True):
        col_rag = st.columns(2)
        with col_rag[0]:
            st.metric("Retrieved Docs", rag_data.get("n_results", "?"))
            st.markdown("**Sources**")
            for s in rag_data.get("sources", []):
                st.code(f"PMID {s.get('pmid')} — {s.get('journal', '?')} ({s.get('year', '?')})")
        with col_rag[1]:
            st.markdown("**How it works**")
            st.capsule("""
The query is embedded using all-MiniLM-L6-v2 and cosine-similarity searched against a ChromaDB
collection of 327 PubMed abstracts covering microbiome-longevity-biomarker literature.
Top-3 most similar abstracts are retrieved and injected as context into the Groq LLM prompt.
            """)
        st.markdown(f"**Retrieved answer:**\n\n{rag_data.get('answer', 'No answer')}")
        st.caption("RAG confidence: based on cosine similarity score of top-1 document (higher = more relevant context)")

    # LLM section
    with st.expander("🧠 LLM Inference", expanded=True):
        col_llm = st.columns(2)
        with col_llm[0]:
            st.metric("Total Inference Time", f"{result.get('_inference_ms', '?')}ms")
            st.metric("Groq Model", "llama-3.3-70b-versatile")
            st.metric("Temperature", "0.3 (focused, factual)")
            st.metric("Max Tokens", "1024")
        with col_llm[1]:
            st.markdown("**How inference works**")
            st.capsule("""
The LLM receives a structured prompt containing:
1. The user's question
2. ML longevity score + KNN summary as JSON
3. Top-3 RAG abstracts with PMID citations

The Groq model (llama-3.3-70b-versatile) generates a narrative response with citations.
Temperature=0.3 keeps output focused and less hallucinatory.
            """)
        st.caption(f"LLM output reflects ML+RAG context only — it does not access external data beyond the RAG corpus of 327 PubMed articles.")

    # Raw JSON
    with st.expander("📄 Raw JSON Response"):
        st.json(result)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "LongevityAI Microbiome · HGMA (6,014 samples) · "
    "Groq llama-3.3-70b-versatile · ChromaDB 327 PubMed articles · "
    f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
)