"""
app.py — Interface Streamlit IA Sentiment Analysis
Thème: IA moderne, bleu deep-tech, futuriste & époustouflant
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import time
import random
from huggingface_hub import hf_hub_download
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentAI · Analyse de Sentiment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — THÈME IA BLEU FUTURISTE
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Exo+2:wght@100;200;300;400;600;800&display=swap');

/* ── Racine ── */
:root {
    --bg-void:      #020814;
    --bg-deep:      #060f24;
    --bg-panel:     #0a1628;
    --bg-card:      #0d1e38;
    --blue-core:    #0d6efd;
    --blue-glow:    #1a8fff;
    --blue-bright:  #4db8ff;
    --blue-ice:     #a8d8ff;
    --cyan-neon:    #00d4ff;
    --cyan-dim:     #0099cc;
    --accent-teal:  #00f5d4;
    --accent-violet:#7c3aed;
    --text-primary: #e8f4ff;
    --text-muted:   #6a9ab8;
    --text-dim:     #3a6a8a;
    --positive:     #00e676;
    --negative:     #ff4060;
    --neutral:      #ffb300;
    --border-glow:  rgba(0, 212, 255, 0.25);
    --border-dim:   rgba(13, 110, 253, 0.2);
}

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--bg-void) !important;
    color: var(--text-primary);
}

.stApp {
    background: 
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(13,110,253,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(0,212,255,0.08) 0%, transparent 50%),
        radial-gradient(ellipse 30% 20% at 10% 90%, rgba(124,58,237,0.06) 0%, transparent 50%),
        var(--bg-void);
    min-height: 100vh;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--blue-core); border-radius: 2px; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1rem 2.5rem 2rem 2.5rem;
    max-width: 1200px;
}

/* ══════════════════════════════════
   HERO HEADER
══════════════════════════════════ */
.hero-wrapper {
    text-align: center;
    padding: 3rem 0 2rem 0;
    position: relative;
    overflow: hidden;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(0,212,255,0.12), rgba(13,110,253,0.12));
    border: 1px solid var(--cyan-dim);
    color: var(--cyan-neon);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    padding: 0.35rem 1.2rem;
    border-radius: 2px;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    animation: fadeSlideDown 0.8s ease both;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.01em;
    margin: 0 0 0.5rem 0;
    animation: fadeSlideDown 0.9s ease both;
    animation-delay: 0.1s;
}
.hero-title .word-sentiment {
    background: linear-gradient(135deg, var(--blue-bright) 0%, var(--cyan-neon) 50%, var(--accent-teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    filter: drop-shadow(0 0 30px rgba(0,212,255,0.4));
}
.hero-subtitle {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.05rem;
    font-weight: 200;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    margin: 0.5rem 0 0 0;
    animation: fadeSlideDown 1s ease both;
    animation-delay: 0.2s;
}
.hero-divider {
    width: 180px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan-neon), transparent);
    margin: 2rem auto 0 auto;
    animation: expandLine 1.2s ease both;
    animation-delay: 0.4s;
}
.scan-line {
    position: absolute;
    top: 0; left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.03), transparent);
    animation: scanPass 4s ease-in-out infinite;
    pointer-events: none;
}

/* ══════════════════════════════════
   STATS ROW
══════════════════════════════════ */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    animation: fadeUp 1s ease both;
    animation-delay: 0.5s;
}
.stat-chip {
    flex: 1;
    background: linear-gradient(135deg, var(--bg-panel), var(--bg-card));
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 0.9rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.3s;
}
.stat-chip::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan-neon), transparent);
    opacity: 0.6;
}
.stat-chip:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.stat-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--cyan-neon);
    display: block;
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-family: 'Space Mono', monospace;
    margin-top: 0.3rem;
    display: block;
}

/* ══════════════════════════════════
   INPUT PANEL
══════════════════════════════════ */
.input-panel {
    background: linear-gradient(145deg, var(--bg-panel) 0%, var(--bg-card) 100%);
    border: 1px solid var(--border-dim);
    border-radius: 6px;
    padding: 1.8rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: fadeUp 1s ease both;
    animation-delay: 0.6s;
}
.input-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--blue-core), var(--cyan-neon), var(--accent-teal));
}
.input-panel::after {
    content: 'INPUT_VECTOR';
    position: absolute;
    top: 0.6rem; right: 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
}
.panel-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--cyan-neon);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.panel-label::before {
    content: '';
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--cyan-neon);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--cyan-neon);
    animation: pulse 2s ease infinite;
}

/* Streamlit textarea override */
.stTextArea textarea {
    background: rgba(2, 8, 20, 0.8) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    resize: vertical !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: var(--cyan-neon) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.12), 0 0 20px rgba(0,212,255,0.08) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: var(--text-dim) !important;
    font-style: italic;
}
.stTextArea label { display: none !important; }

/* ══════════════════════════════════
   BUTTONS
══════════════════════════════════ */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0a47c9 0%, #0d6efd 50%, #0099cc 100%) !important;
    border: none !important;
    border-radius: 4px !important;
    color: white !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(13,110,253,0.35), 0 0 0 1px rgba(0,212,255,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(13,110,253,0.5), 0 0 40px rgba(0,212,255,0.15) !important;
    background: linear-gradient(135deg, #1458d9 0%, #1a8fff 50%, #00b8e6 100%) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══════════════════════════════════
   RESULT CARDS
══════════════════════════════════ */
.result-wrapper {
    animation: fadeUp 0.6s ease both;
}
.result-card {
    border-radius: 6px;
    padding: 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    border: 1px solid;
}
.result-card.positive {
    background: linear-gradient(145deg, rgba(0,230,118,0.06), rgba(0,230,118,0.02));
    border-color: rgba(0,230,118,0.3);
}
.result-card.negative {
    background: linear-gradient(145deg, rgba(255,64,96,0.06), rgba(255,64,96,0.02));
    border-color: rgba(255,64,96,0.3);
}
.result-card.neutral {
    background: linear-gradient(145deg, rgba(255,179,0,0.06), rgba(255,179,0,0.02));
    border-color: rgba(255,179,0,0.3);
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
}
.result-card.positive::before { background: linear-gradient(90deg, transparent, var(--positive), transparent); }
.result-card.negative::before { background: linear-gradient(90deg, transparent, var(--negative), transparent); }
.result-card.neutral::before { background: linear-gradient(90deg, transparent, var(--neutral), transparent); }

.result-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.result-icon {
    font-size: 2.5rem;
    filter: drop-shadow(0 0 12px currentColor);
    animation: floatIcon 3s ease infinite;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--text-dim);
}
.result-sentiment {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    line-height: 1;
}
.result-sentiment.positive { color: var(--positive); text-shadow: 0 0 30px rgba(0,230,118,0.4); }
.result-sentiment.negative { color: var(--negative); text-shadow: 0 0 30px rgba(255,64,96,0.4); }
.result-sentiment.neutral  { color: var(--neutral);  text-shadow: 0 0 30px rgba(255,179,0,0.4); }

/* Confidence bar */
.conf-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.conf-track {
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 0.4rem;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1.5s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}
.conf-fill::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 2s ease infinite;
}
.conf-fill.positive { background: linear-gradient(90deg, #00a84a, var(--positive)); }
.conf-fill.negative { background: linear-gradient(90deg, #cc0033, var(--negative)); }
.conf-fill.neutral  { background: linear-gradient(90deg, #cc8800, var(--neutral)); }
.conf-pct {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
}
.conf-pct.positive { color: var(--positive); }
.conf-pct.negative { color: var(--negative); }
.conf-pct.neutral  { color: var(--neutral);  }

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.5rem 0;
}
.prob-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    min-width: 80px;
    text-transform: uppercase;
}
.prob-bar {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 1.2s cubic-bezier(0.16,1,0.3,1);
}
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    min-width: 40px;
    text-align: right;
}

/* ══════════════════════════════════
   INFO PANEL / EXAMPLES
══════════════════════════════════ */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1rem 0;
    animation: fadeUp 1s ease both;
    animation-delay: 0.7s;
}
.info-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 1.2rem;
    position: relative;
    cursor: pointer;
    transition: border-color 0.3s, transform 0.2s, background 0.3s;
}
.info-card:hover {
    border-color: var(--border-glow);
    background: var(--bg-card);
    transform: translateY(-2px);
}
.info-card-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    color: var(--cyan-neon);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.8;
}
.info-card-text {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.6;
    font-style: italic;
}

/* ══════════════════════════════════
   HISTORY
══════════════════════════════════ */
.hist-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.1rem;
    background: var(--bg-panel);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    margin: 0.5rem 0;
    transition: border-color 0.3s;
}
.hist-item:hover { border-color: var(--border-glow); }
.hist-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.hist-text {
    flex: 1;
    font-size: 0.85rem;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hist-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    font-weight: 700;
    letter-spacing: 0.1em;
    flex-shrink: 0;
}

/* ══════════════════════════════════
   SECTION HEADERS
══════════════════════════════════ */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.5rem 0 0.8rem 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--text-dim);
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-dim), transparent);
}

/* ══════════════════════════════════
   LOADING ANIMATION
══════════════════════════════════ */
.loading-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2.5rem;
    gap: 1.2rem;
}
.loading-rings {
    position: relative;
    width: 64px; height: 64px;
}
.ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid transparent;
    animation: spin 1.2s linear infinite;
}
.ring-1 { border-top-color: var(--cyan-neon); animation-duration: 0.9s; }
.ring-2 { inset: 8px; border-right-color: var(--blue-glow); animation-duration: 1.2s; animation-direction: reverse; }
.ring-3 { inset: 16px; border-bottom-color: var(--accent-teal); animation-duration: 1.5s; }
.loading-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--cyan-neon);
    text-transform: uppercase;
    animation: blink 1.2s ease infinite;
}

/* ══════════════════════════════════
   KEYFRAMES
══════════════════════════════════ */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes expandLine {
    from { width: 0; opacity: 0; }
    to   { width: 180px; opacity: 1; }
}
@keyframes scanPass {
    0%   { left: -60%; }
    100% { left: 160%; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--cyan-neon); }
    50%       { opacity: 0.4; box-shadow: 0 0 3px var(--cyan-neon); }
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-5px); }
}
@keyframes shimmer {
    0%   { left: -100%; }
    100% { left: 200%; }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Streamlit selectbox & misc ── */
.stSelectbox > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}
.stMarkdown p { color: var(--text-muted); }
div[data-testid="stHorizontalBlock"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Télécharge et charge modèle + vectorizer depuis HF Hub."""
    HF_REPO = os.environ.get("HF_REPO", "")
    if not HF_REPO:
        return None, None
    try:
        model_path = hf_hub_download(repo_id=HF_REPO, filename="model.pkl")
        vec_path   = hf_hub_download(repo_id=HF_REPO, filename="vectorizer.pkl")
        model      = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None, None

def predict_sentiment(text: str, model, vectorizer):
    """Prédit le sentiment et retourne label + probabilités."""
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        prob_dict = {str(c): float(p) for c, p in zip(classes, proba)}
    else:
        prob_dict = {pred: 1.0}

    return str(pred), prob_dict

def demo_predict(text: str):
    """Simulateur de prédiction quand le modèle n'est pas disponible."""
    positive_kw = ["excellent", "great", "love", "best", "amazing", "wonderful", "super",
                   "génial", "parfait", "bien", "bravo", "super", "bon", "merci", "top",
                   "recommend", "good", "happy", "satisfied", "nice", "recommend"]
    negative_kw = ["terrible", "awful", "bad", "worst", "hate", "horrible", "poor",
                   "nul", "mauvais", "problème", "bug", "lent", "horrible",
                   "disappointed", "broken", "useless", "waste", "never", "avoid"]

    text_lower = text.lower()
    pos_score  = sum(1 for k in positive_kw if k in text_lower)
    neg_score  = sum(1 for k in negative_kw if k in text_lower)

    if pos_score > neg_score:
        conf   = min(0.65 + pos_score * 0.07 + random.uniform(-0.05, 0.05), 0.97)
        neg_p  = (1 - conf) * random.uniform(0.3, 0.7)
        neu_p  = 1 - conf - neg_p
        return "POSITIVE", {"POSITIVE": conf, "NEGATIVE": neg_p, "NEUTRAL": neu_p}
    elif neg_score > pos_score:
        conf   = min(0.60 + neg_score * 0.07 + random.uniform(-0.05, 0.05), 0.97)
        pos_p  = (1 - conf) * random.uniform(0.3, 0.7)
        neu_p  = 1 - conf - pos_p
        return "NEGATIVE", {"POSITIVE": pos_p, "NEGATIVE": conf, "NEUTRAL": neu_p}
    else:
        conf   = random.uniform(0.45, 0.62)
        rem    = 1 - conf
        return "NEUTRAL", {"POSITIVE": rem * 0.5, "NEGATIVE": rem * 0.5, "NEUTRAL": conf}

# ─────────────────────────────────────────────
# SENTIMENT META
# ─────────────────────────────────────────────
SENTIMENT_META = {
    "POSITIVE": {"icon": "✦", "class": "positive", "color": "#00e676", "bg": "rgba(0,230,118,0.08)"},
    "NEGATIVE": {"icon": "✦", "class": "negative", "color": "#ff4060", "bg": "rgba(255,64,96,0.08)"},
    "NEUTRAL":  {"icon": "✦", "class": "neutral",  "color": "#ffb300", "bg": "rgba(255,179,0,0.08)"},
}

EXAMPLES = [
    {"tag": "AVIS CLIENT", "text": "Ce produit est absolument incroyable ! La qualité dépasse toutes mes attentes. Je le recommande vivement à tout le monde."},
    {"tag": "SUPPORT TECH", "text": "Terrible experience. The app keeps crashing and customer service was completely useless. Never buying again."},
    {"tag": "REVUE PRODUIT", "text": "Le produit est correct. Ni fantastique, ni décevant. Livraison dans les délais, emballage standard."},
    {"tag": "FEEDBACK USER", "text": "I've been using this for 3 months. Some bugs exist but the core functionality works well. Mixed feelings overall."},
]

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

# ── Hero ──
st.markdown("""
<div class="hero-wrapper">
    <div class="scan-line"></div>
    <div class="hero-badge">⬡ Neural Sentiment Engine · v2.0</div>
    <h1 class="hero-title">
        <span class="word-sentiment">SENTIMENT</span> ANALYSIS
    </h1>
    <p class="hero-subtitle">Intelligence artificielle · Analyse émotionnelle en temps réel</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Stats ──
pos_count = sum(1 for h in st.session_state.history if h["label"] == "POSITIVE")
neg_count = sum(1 for h in st.session_state.history if h["label"] == "NEGATIVE")
neu_count = sum(1 for h in st.session_state.history if h["label"] == "NEUTRAL")

st.markdown(f"""
<div class="stats-row">
    <div class="stat-chip">
        <span class="stat-value">{st.session_state.total_analyzed}</span>
        <span class="stat-label">Textes analysés</span>
    </div>
    <div class="stat-chip">
        <span class="stat-value" style="color:#00e676">{pos_count}</span>
        <span class="stat-label">Positifs</span>
    </div>
    <div class="stat-chip">
        <span class="stat-value" style="color:#ff4060">{neg_count}</span>
        <span class="stat-label">Négatifs</span>
    </div>
    <div class="stat-chip">
        <span class="stat-value" style="color:#ffb300">{neu_count}</span>
        <span class="stat-label">Neutres</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ──
col_main, col_side = st.columns([3, 2], gap="large")

with col_main:

    # ── Input panel ──
    st.markdown("""
    <div class="input-panel">
        <div class="panel-label">Texte à analyser</div>
    </div>
    """, unsafe_allow_html=True)

    user_text = st.text_area(
        "text",
        height=160,
        placeholder="Entrez votre texte ici... (avis client, commentaire, tweet, message...)",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        analyze_btn = st.button("⬡  ANALYSER LE SENTIMENT", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("✕  RESET", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.session_state.total_analyzed = 0
        st.rerun()

    # ── Analysis ──
    if analyze_btn:
        if not user_text.strip():
            st.markdown("""
            <div style="border:1px solid rgba(255,64,96,0.3); background:rgba(255,64,96,0.05);
                        padding:1rem 1.2rem; border-radius:4px; margin:1rem 0;
                        font-family:'Space Mono',monospace; font-size:0.75rem; color:#ff4060;
                        letter-spacing:0.1em;">
                ⚠ INPUT_EMPTY — Veuillez entrer un texte à analyser.
            </div>""", unsafe_allow_html=True)
        else:
            # Loading
            loading_ph = st.empty()
            with loading_ph.container():
                st.markdown("""
                <div class="loading-wrap">
                    <div class="loading-rings">
                        <div class="ring ring-1"></div>
                        <div class="ring ring-2"></div>
                        <div class="ring ring-3"></div>
                    </div>
                    <div class="loading-text">traitement du signal neuronal...</div>
                </div>""", unsafe_allow_html=True)
            time.sleep(1.2)
            loading_ph.empty()

            # Load model (first time)
            if st.session_state.model is None:
                m, v = load_model()
                st.session_state.model = m
                st.session_state.vectorizer = v

            if st.session_state.model and st.session_state.vectorizer:
                label, probs = predict_sentiment(
                    user_text, st.session_state.model, st.session_state.vectorizer
                )
            else:
                label, probs = demo_predict(user_text)

            meta = SENTIMENT_META.get(label, SENTIMENT_META["NEUTRAL"])
            confidence = probs.get(label, 0.5)
            conf_pct   = int(confidence * 100)

            # ── Result card ──
            # Build probability bars HTML
            labels_order = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            colors_map   = {"POSITIVE": "#00e676", "NEGATIVE": "#ff4060", "NEUTRAL": "#ffb300"}
            prob_bars_html = ""
            for lbl in labels_order:
                p     = probs.get(lbl, 0)
                pct_v = int(p * 100)
                col   = colors_map[lbl]
                prob_bars_html += f"""
                <div class="prob-row">
                    <span class="prob-name">{lbl}</span>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width:{pct_v}%; background:{col};"></div>
                    </div>
                    <span class="prob-pct" style="color:{col}">{pct_v}%</span>
                </div>"""

            emoji = '😊' if label == 'POSITIVE' else '😞' if label == 'NEGATIVE' else '😐'

            # Border / glow colors per sentiment
            border_colors = {"positive": "rgba(0,230,118,0.3)", "negative": "rgba(255,64,96,0.3)", "neutral": "rgba(255,179,0,0.3)"}
            top_colors    = {"positive": "#00e676", "negative": "#ff4060", "neutral": "#ffb300"}
            bg_colors     = {"positive": "rgba(0,230,118,0.06)", "negative": "rgba(255,64,96,0.06)", "neutral": "rgba(255,179,0,0.06)"}
            sent_colors   = {"positive": "#00e676", "negative": "#ff4060", "neutral": "#ffb300"}
            fill_grad     = {"positive": "linear-gradient(90deg,#00a84a,#00e676)", "negative": "linear-gradient(90deg,#cc0033,#ff4060)", "neutral": "linear-gradient(90deg,#cc8800,#ffb300)"}

            cls = meta['class']

            card_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Space+Mono:wght@400;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'Exo 2', sans-serif;
    padding: 0;
    margin: 0;
  }}
  .result-card {{
    background: {bg_colors[cls]};
    border: 1px solid {border_colors[cls]};
    border-radius: 6px;
    padding: 1.8rem;
    position: relative;
    overflow: hidden;
  }}
  .result-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, {top_colors[cls]}, transparent);
  }}
  .result-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .result-icon {{
    font-size: 2.5rem;
    color: {meta['color']};
  }}
  .result-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #3a6a8a;
    margin-bottom: 0.3rem;
  }}
  .result-sentiment {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: {sent_colors[cls]};
    text-shadow: 0 0 30px {sent_colors[cls]}66;
  }}
  .conf-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #3a6a8a;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }}
  .conf-track {{
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 0.4rem;
  }}
  .conf-fill {{
    height: 100%;
    border-radius: 3px;
    background: {fill_grad[cls]};
    width: {conf_pct}%;
  }}
  .conf-pct {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: {sent_colors[cls]};
  }}
  .prob-section {{ margin-top: 1.5rem; }}
  .prob-row {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.6rem 0;
  }}
  .prob-name {{
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #6a9ab8;
    min-width: 80px;
    text-transform: uppercase;
  }}
  .prob-bar {{
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
  }}
  .prob-fill {{
    height: 100%;
    border-radius: 2px;
    opacity: 0.85;
  }}
  .prob-pct {{
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    min-width: 40px;
    text-align: right;
  }}
</style>
</head>
<body>
<div class="result-card">
  <div class="result-header">
    <div class="result-icon">{emoji}</div>
    <div>
      <div class="result-label">VERDICT SENTIMENT</div>
      <div class="result-sentiment">{label}</div>
    </div>
  </div>

  <div class="conf-label">Niveau de confiance</div>
  <div class="conf-track"><div class="conf-fill"></div></div>
  <div class="conf-pct">{conf_pct}%</div>

  <div class="prob-section">
    <div class="conf-label" style="margin-bottom:0.8rem;">Distribution de probabilité</div>
    {prob_bars_html}
  </div>
</div>
</body>
</html>"""

            components.html(card_html, height=340, scrolling=False)

            # Save to history
            st.session_state.history.insert(0, {
                "text": user_text[:80] + ("…" if len(user_text) > 80 else ""),
                "label": label,
                "conf": conf_pct,
                "color": meta["color"],
            })
            st.session_state.total_analyzed += 1

            if not st.session_state.model:
                st.markdown("""
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:rgba(255,179,0,0.6);
                            padding:0.5rem 0; letter-spacing:0.15em; text-align:center;">
                    ⚠ MODE DÉMONSTRATION · Configurez HF_REPO pour charger le modèle réel
                </div>""", unsafe_allow_html=True)


with col_side:

    # ── Exemples ──
    st.markdown('<div class="section-head">Exemples rapides</div>', unsafe_allow_html=True)

    for ex in EXAMPLES:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-tag">⬡ {ex['tag']}</div>
            <div class="info-card-text">"{ex['text'][:100]}{'…' if len(ex['text'])>100 else ''}"</div>
        </div>""", unsafe_allow_html=True)

    # ── Model info ──
    st.markdown('<div class="section-head">Informations modèle</div>', unsafe_allow_html=True)
    hf_repo = os.environ.get("HF_REPO", "Non configuré")
    model_status = "✅ Chargé" if st.session_state.model else "⚙ Démonstration"
    st.markdown(f"""
    <div style="background:var(--bg-panel); border:1px solid var(--border-dim); border-radius:4px; padding:1.1rem;">
        <div style="display:flex; flex-direction:column; gap:0.7rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--text-dim); letter-spacing:0.15em; text-transform:uppercase;">Statut</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--cyan-neon);">{model_status}</span>
            </div>
            <div style="height:1px; background:var(--border-dim);"></div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--text-dim); letter-spacing:0.15em; text-transform:uppercase;">Dépôt HF</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--blue-bright); word-break:break-all; max-width:60%; text-align:right;">{hf_repo}</span>
            </div>
            <div style="height:1px; background:var(--border-dim);"></div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--text-dim); letter-spacing:0.15em; text-transform:uppercase;">Algorithme</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--text-muted);">ML Classif.</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── History ──
    if st.session_state.history:
        st.markdown('<div class="section-head">Historique récent</div>', unsafe_allow_html=True)
        for h in st.session_state.history[:6]:
            st.markdown(f"""
            <div class="hist-item">
                <div class="hist-dot" style="background:{h['color']}; box-shadow:0 0 6px {h['color']};"></div>
                <span class="hist-text">{h['text']}</span>
                <span class="hist-badge" style="background:{h['color']}22; color:{h['color']}; border:1px solid {h['color']}44;">
                    {h['label'][:3]} {h['conf']}%
                </span>
            </div>""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1rem 0; margin-top:2rem;
            border-top:1px solid rgba(13,110,253,0.1);">
    <div style="font-family:'Space Mono',monospace; font-size:0.55rem; letter-spacing:0.3em;
                color:var(--text-dim); text-transform:uppercase;">
        ⬡ SentimentAI · Propulsé par Hugging Face Hub &amp; Scikit-learn · 2024
    </div>
</div>""", unsafe_allow_html=True)
