from pathlib import Path
import os
import random
import urllib.error
import urllib.request

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")


def interpret_probability(probability: float) -> str:
    if probability >= 0.75:
        return "Very likely to buy the next release."
    if probability >= 0.55:
        return "Likely to buy, especially if the release matches their favorite series."
    if probability >= 0.35:
        return "Mixed intent; interest may depend on price, rarity, and design."
    return "Unlikely to buy the next release."


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_api_status() -> tuple[str, str]:
    try:
        with urllib.request.urlopen(f"{FASTAPI_URL}/health", timeout=0.8) as response:
            if response.status == 200:
                return "🟢 API connected", f"REST endpoints are available at {FASTAPI_URL}."
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return (
        "🟡 API optional",
        "The dashboard still works with the saved model. Start FastAPI with `python -m uvicorn app_api.main:app --reload` to demo REST endpoints.",
    )


CHARACTER_EMOJIS = {
    "Labubu": "🐰",
    "Molly": "🌸",
    "Skullpanda": "💀",
    "Dimoo": "🌙",
    "Hirono": "🧸",
    "Crybaby": "💧",
}

MYSTERY_BOX_INSIGHTS = [
    "You'd probably open 8 blind boxes before finding your favorite figure.",
    "Collectors similar to you usually chase Secret Editions.",
    "Your collecting style is balanced between fun and rarity.",
    "Your shelf is giving curated chaos in the best way.",
    "You seem like the kind of collector who remembers every pull story.",
]


st.set_page_config(page_title="POP MART Collector Predictor", page_icon="🎁", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #1f1c2b;
        --muted: #706b7d;
        --soft-muted: #9a94a8;
        --pink: #f7a9c7;
        --blue: #a8d4f7;
        --mint: #b8eadb;
        --lemon: #f6e3a4;
        --lavender: #d7cdf8;
        --card: rgba(255, 255, 255, 0.82);
        --solid-card: #ffffff;
        --line: rgba(31, 28, 43, 0.075);
        --shadow: 0 18px 50px rgba(31, 28, 43, 0.08);
        --soft-shadow: 0 8px 28px rgba(31, 28, 43, 0.055);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        letter-spacing: 0;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(247, 169, 199, 0.18), transparent 32rem),
            radial-gradient(circle at top right, rgba(168, 212, 247, 0.16), transparent 30rem),
            linear-gradient(180deg, #ffffff 0%, #fbf9fc 54%, #f7fbff 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1160px;
        padding-top: 2.4rem;
        padding-bottom: 3.4rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.82);
        border-right: 1px solid var(--line);
        box-shadow: 12px 0 36px rgba(31, 28, 43, 0.045);
        backdrop-filter: blur(22px);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-radius: 26px;
        box-shadow: var(--soft-shadow);
        padding: 1.1rem;
        margin: 0 0.3rem 1rem;
    }

    .sidebar-title {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.18;
        margin-bottom: 0.85rem;
    }

    .sidebar-link {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.72rem 0.8rem;
        margin: 0.36rem 0;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.64);
        color: var(--ink);
        font-weight: 720;
        border: 1px solid rgba(31, 28, 43, 0.055);
        transition: background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }

    .sidebar-link:hover {
        background: rgba(247, 169, 199, 0.12);
        transform: translateX(2px);
        box-shadow: 0 8px 18px rgba(31, 28, 43, 0.045);
    }

    .sidebar-footer {
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 650;
        line-height: 1.55;
        padding: 0.9rem 0.2rem 0;
    }

    .sidebar-note {
        margin-top: 0.95rem;
        padding: 0.85rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.84rem;
        font-weight: 650;
        line-height: 1.5;
    }

    .sidebar-status {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 820;
        margin-bottom: 0.35rem;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
        gap: 1.35rem;
        align-items: stretch;
        margin-bottom: 1.65rem;
    }

    .hero-card,
    .stat-card,
    .result-card {
        background: var(--card);
        border: 1px solid rgba(255, 255, 255, 0.92);
        border-radius: 30px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(20px);
    }

    .hero-card {
        padding: clamp(1.55rem, 3.4vw, 2.55rem);
        position: relative;
        overflow: hidden;
    }

    .hero-card:after {
        content: "";
        position: absolute;
        inset: auto -6rem -7rem auto;
        width: 18rem;
        height: 18rem;
        background: radial-gradient(circle, rgba(247, 169, 199, 0.22), rgba(168, 212, 247, 0.10) 58%, transparent 72%);
        border-radius: 999px;
        opacity: 1;
        filter: blur(0);
    }

    .eyebrow {
        color: #c45f89;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero h1 {
        color: var(--ink);
        font-size: clamp(2.45rem, 5vw, 4.6rem);
        line-height: 0.98;
        margin: 0 0 0.9rem;
        font-weight: 800;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 48rem;
        margin: 0;
    }

    .emoji-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.35rem;
        font-size: 2.15rem;
    }

    .emoji-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 3.4rem;
        height: 3.4rem;
        border-radius: 1.2rem;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(36, 31, 51, 0.08);
    }

    .collector-lineup {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.45rem;
        position: relative;
        z-index: 1;
    }

    .collector-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.72rem 0.95rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        box-shadow: var(--soft-shadow);
        color: var(--ink);
        font-weight: 800;
        white-space: nowrap;
    }

    .stat-card {
        padding: 1.45rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 100%;
    }

    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1;
    }

    .stat-label {
        color: var(--muted);
        font-weight: 650;
        margin-top: 0.45rem;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: var(--ink);
        font-size: 1.5rem;
        font-weight: 800;
        margin: 1.8rem 0 0.85rem;
    }

    .info-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(260px, 0.75fr);
        gap: 1rem;
        margin: 1.35rem 0 1.7rem;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--line);
        border-radius: 28px;
        box-shadow: var(--soft-shadow);
        padding: 1.25rem;
    }

    .info-title {
        color: var(--ink);
        font-size: 1.12rem;
        font-weight: 850;
        margin-bottom: 0.65rem;
    }

    .info-list {
        display: grid;
        gap: 0.48rem;
        color: var(--muted);
        font-weight: 650;
        line-height: 1.52;
    }

    .disclaimer-card {
        background: rgba(246, 227, 164, 0.18);
        border: 1px solid rgba(31, 28, 43, 0.07);
        border-radius: 28px;
        box-shadow: var(--soft-shadow);
        padding: 1.25rem;
        color: var(--muted);
        font-weight: 650;
        line-height: 1.58;
    }

    .input-section-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.54rem 0.84rem;
        border-radius: 999px;
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 850;
        margin: 0.55rem 0 0.95rem;
        border: 1px solid var(--line);
        box-shadow: 0 6px 16px rgba(31, 28, 43, 0.045);
    }

    .input-section-header.profile {
        background: rgba(215, 205, 248, 0.28);
    }

    .input-section-header.habits {
        background: rgba(184, 234, 219, 0.30);
    }

    .input-section-header.budget {
        background: rgba(246, 227, 164, 0.32);
    }

    .input-section-header.preferences {
        background: rgba(247, 169, 199, 0.25);
    }

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--line);
        border-radius: 30px;
        box-shadow: var(--shadow);
        padding: clamp(1.2rem, 2.5vw, 1.8rem);
        backdrop-filter: blur(18px);
    }

    div[data-testid="stForm"] label,
    div[data-testid="stMetricLabel"] {
        color: var(--ink) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stForm"] [data-testid="column"] {
        padding: 0 0.35rem;
    }

    div[data-testid="stSlider"] {
        padding: 0.14rem 0 0.5rem;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 18px;
        border-color: rgba(31, 28, 43, 0.11);
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 5px 16px rgba(31, 28, 43, 0.035);
    }

    .stSlider [data-baseweb="slider"] > div {
        color: #ff6fa8;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        border: 0;
        border-radius: 999px;
        padding: 0.94rem 1rem;
        color: #ffffff;
        font-weight: 800;
        background: linear-gradient(90deg, #d36b95 0%, #7d7bd1 100%);
        box-shadow: 0 14px 28px rgba(125, 123, 209, 0.22);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(125, 123, 209, 0.28);
    }

    .result-card {
        padding: 1.45rem;
        margin-top: 1.2rem;
    }

    .result-heading {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.6rem;
    }

    .premium-result-card {
        margin-top: 1.25rem;
        padding: clamp(1.35rem, 2.8vw, 1.9rem);
        border-radius: 32px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        background:
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.92), transparent 13rem),
            linear-gradient(135deg, rgba(247, 169, 199, 0.17), rgba(246, 227, 164, 0.15), rgba(168, 212, 247, 0.15));
        overflow: hidden;
    }

    .character-emoji-card {
        margin-top: 1.25rem;
        padding: 1.45rem;
        border-radius: 32px;
        text-align: center;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--line);
        box-shadow: var(--soft-shadow);
    }

    .character-emoji {
        font-size: clamp(4rem, 12vw, 7rem);
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .character-emoji-label {
        color: var(--muted);
        font-size: 0.95rem;
        font-weight: 800;
    }

    .premium-result-kicker {
        color: #c45f89;
        font-size: 0.82rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .premium-result-title {
        color: var(--ink);
        font-size: clamp(1.75rem, 4vw, 2.65rem);
        font-weight: 850;
        line-height: 1.08;
        margin-bottom: 0.75rem;
    }

    .premium-result-copy {
        color: var(--muted);
        font-size: 1.06rem;
        font-weight: 600;
        line-height: 1.65;
        max-width: 48rem;
        margin-bottom: 1.05rem;
    }

    .premium-result-meta {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--ink);
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(36, 31, 51, 0.08);
        border-radius: 999px;
        padding: 0.58rem 0.84rem;
        font-weight: 800;
        box-shadow: 0 10px 22px rgba(36, 31, 51, 0.07);
    }

    .mystery-insight-card {
        margin-top: 1.15rem;
        padding: 1.25rem 1.35rem;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.80);
        border: 1px solid var(--line);
        box-shadow: var(--soft-shadow);
    }

    .mystery-insight-title {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 850;
        margin-bottom: 0.45rem;
    }

    .mystery-insight-copy {
        color: var(--muted);
        font-size: 1.02rem;
        font-weight: 650;
        line-height: 1.6;
    }

    .personality-bars {
        margin-top: 1.15rem;
        padding: clamp(1.2rem, 2.5vw, 1.55rem);
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--line);
        box-shadow: var(--soft-shadow);
    }

    .bar-label {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 850;
        margin-top: 0.65rem;
    }

    .bar-percent {
        color: var(--muted);
        font-size: 0.95rem;
        font-weight: 800;
        margin: -0.35rem 0 0.65rem;
    }

    .app-footer {
        margin-top: 2.6rem;
        padding: 1.45rem;
        text-align: center;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        box-shadow: var(--soft-shadow);
        color: var(--muted);
        font-weight: 650;
        line-height: 1.78;
    }

    .footer-rule {
        color: rgba(36, 31, 51, 0.24);
        letter-spacing: 0.06em;
        margin: 0.1rem 0;
    }

    .footer-title {
        color: var(--ink);
        font-size: 1.22rem;
        font-weight: 850;
        margin: 0.35rem 0 0.15rem;
    }

    @keyframes candyStripe {
        0% { background-position: 0 0; }
        100% { background-position: 2rem 0; }
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(247, 169, 199, 0.14), rgba(168, 212, 247, 0.14));
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1rem 1.15rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 2.4rem;
        font-weight: 800;
    }

    div[data-testid="stAlert"] {
        border-radius: 20px;
        border: 1px solid rgba(36, 31, 51, 0.08);
        box-shadow: 0 12px 26px rgba(36, 31, 51, 0.06);
    }

    .stProgress > div > div > div > div {
        background:
            repeating-linear-gradient(
                45deg,
                rgba(255, 255, 255, 0.20) 0,
                rgba(255, 255, 255, 0.20) 0.45rem,
                transparent 0.45rem,
                transparent 0.9rem
            ),
            linear-gradient(90deg, #d36b95, #d8bf73, #79cbb4, #79b5df);
        background-size: 2rem 2rem, 100% 100%;
        animation: candyStripe 1.2s linear infinite;
    }

    footer,
    #MainMenu,
    header {
        visibility: hidden;
    }

    @media (max-width: 820px) {
        .hero {
            grid-template-columns: 1fr;
        }

        .info-grid {
            grid-template-columns: 1fr;
        }

        .hero-card {
            padding: 1.4rem;
        }

        .hero h1 {
            font-size: 2.45rem;
        }

        .stat-number {
            font-size: 2.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

api_status_title, api_status_detail = get_api_status()

st.sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <div class="sidebar-title">🧸 Collector Dashboard</div>
        <div class="sidebar-link">📊 Dataset</div>
        <div class="sidebar-link">🤖 Machine Learning</div>
        <div class="sidebar-link">⚡ FastAPI</div>
        <div class="sidebar-link">🎨 Streamlit</div>
        <div class="sidebar-link">🧪 Testing</div>
        <div class="sidebar-link">📦 Model</div>
        <div class="sidebar-note">
            <div class="sidebar-status">{api_status_title}</div>
            {api_status_detail}
        </div>
        <div class="sidebar-footer">Built with ❤️ using FastAPI + Streamlit + Scikit-learn</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-card">
            <div class="eyebrow">Premium collector dashboard</div>
            <h1>🎁 POP MART Collector Predictor</h1>
            <p>Discover your collector personality based on your collecting habits.</p>
            <div class="collector-lineup">
                <span class="collector-chip">✨ Labubu</span>
                <span class="collector-chip">🌸 Molly</span>
                <span class="collector-chip">💀 Skullpanda</span>
                <span class="collector-chip">🌙 Dimoo</span>
                <span class="collector-chip">🧸 Hirono</span>
            </div>
        </div>
        <div class="stat-card">
            <div>
                <div class="emoji-pill">📦</div>
                <div class="stat-number">2,000</div>
                <div class="stat-label">synthetic collector profiles</div>
            </div>
            <div class="stat-label">Random forest prediction pipeline</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-grid">
        <div class="info-card">
            <div class="info-title">🧭 How this works</div>
            <div class="info-list">
                <div>👤 User enters collector habits.</div>
                <div>🎨 Streamlit sends the collector profile through the prediction workflow.</div>
                <div>⚡ FastAPI provides REST endpoints for the saved model.</div>
                <div>🤖 FastAPI loads the saved Random Forest model.</div>
                <div>📊 The model returns collector level and probabilities for display.</div>
            </div>
        </div>
        <div class="disclaimer-card">
            <div class="info-title">📌 Model disclaimer</div>
            This is an educational mini project using synthetic data. Predictions are for demonstration purposes only.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python scripts/train_model.py` from the project root.")
    st.stop()

st.markdown('<div class="section-title">🛍️ Collector Profile</div>', unsafe_allow_html=True)

with st.form("collector_profile"):
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="input-section-header profile">👤 Collector Profile</div>',
            unsafe_allow_html=True,
        )
        age = st.slider("🎂 Age", min_value=16, max_value=55, value=28)
        collector_type = st.selectbox(
            "🧸 Collector Type",
            ["casual", "display-focused", "completionist", "reseller", "trend-chaser"],
        )
        region = st.selectbox(
            "🌎 Region",
            ["North America", "Europe", "East Asia", "Southeast Asia", "Oceania"],
        )

        st.markdown(
            '<div class="input-section-header habits">🎁 Collecting Habits</div>',
            unsafe_allow_html=True,
        )
        monthly_purchases = st.slider("📦 Blind Boxes per Month", 0, 20, 4)
        collection_size = st.slider("🧸 Collection Size", 0, 180, 24)

    with right:
        st.markdown(
            '<div class="input-section-header budget">💰 Budget</div>',
            unsafe_allow_html=True,
        )
        monthly_budget_usd = st.slider("🛍️ Monthly Budget", 20, 350, 95, step=5)
        resale_interest = st.slider("📈 Resale Interest", 0, 10, 3)

        st.markdown(
            '<div class="input-section-header preferences">⭐ Preferences</div>',
            unsafe_allow_html=True,
        )
        favorite_series = st.selectbox(
            "🌟 Favorite Character",
            ["Labubu", "Molly", "Skullpanda", "Dimoo", "Hirono", "Crybaby"],
        )
        blind_box_risk_tolerance = st.slider("💎 Buys Secret Editions", 0, 10, 7)
        social_media_engagement = st.slider("🎉 Convention Visits", 0, 10, 6)

    series_col, type_col, region_col = st.columns(3)
    with series_col:
        st.caption("Series vibe ✨")
        st.markdown(f"### ✨ {favorite_series}")
    with type_col:
        st.caption("Collector style 🧸")
        st.markdown(f"### {collector_type}")
    with region_col:
        st.caption("Collector region 🌎")
        st.markdown(f"### {region}")

    submitted = st.form_submit_button("🔮 Predict purchase intent")

if submitted:
    model = load_model()
    profile = pd.DataFrame(
        [
            {
                "age": age,
                "monthly_budget_usd": monthly_budget_usd,
                "collection_size": collection_size,
                "monthly_purchases": monthly_purchases,
                "resale_interest": resale_interest,
                "social_media_engagement": social_media_engagement,
                "blind_box_risk_tolerance": blind_box_risk_tolerance,
                "favorite_series": favorite_series,
                "collector_type": collector_type,
                "region": region,
            }
        ]
    )
    probability = float(model.predict_proba(profile)[0][1])
    casual_probability = max(0.0, min(1.0, 1 - probability / 0.6))
    enthusiast_probability = max(0.0, min(1.0, 1 - abs(probability - 0.6) / 0.4))
    hardcore_probability = max(0.0, min(1.0, (probability - 0.4) / 0.6))

    if probability < 0.45:
        result_title = "🌱 Casual Collector"
        result_copy = (
            "You enjoy collecting occasionally and are beginning your POP MART journey."
        )
    elif probability < 0.75:
        result_title = "🎁 Enthusiast Collector"
        result_copy = (
            "You actively collect and regularly expand your collection with exciting new releases."
        )
    else:
        result_title = "👑 Hardcore Collector"
        result_copy = (
            "You are a passionate collector who frequently purchases blind boxes, "
            "hunts secret editions, and maintains an impressive collection."
        )
        st.balloons()

    selected_character_emoji = CHARACTER_EMOJIS.get(favorite_series, "✨")
    st.markdown(
        f"""
        <div class="character-emoji-card">
            <div class="character-emoji">{selected_character_emoji}</div>
            <div class="character-emoji-label">{favorite_series} energy selected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="premium-result-card">
            <div class="premium-result-kicker">Collector Personality</div>
            <div class="premium-result-title">{result_title}</div>
            <div class="premium-result-copy">{result_copy}</div>
            <div class="premium-result-copy">🎉 Prediction Complete!</div>
            <div class="premium-result-meta">🔮 Personality confidence: {probability:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mystery_insight = random.choice(MYSTERY_BOX_INSIGHTS)
    st.markdown(
        f"""
        <div class="mystery-insight-card">
            <div class="mystery-insight-title">🎁 Mystery Box Insight</div>
            <div class="mystery-insight-copy">"{mystery_insight}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="personality-bars">', unsafe_allow_html=True)
    st.markdown('<div class="result-heading">✨ Personality Probability Mix</div>', unsafe_allow_html=True)

    st.markdown('<div class="bar-label">🌱 Casual</div>', unsafe_allow_html=True)
    st.progress(casual_probability)
    st.markdown(f'<div class="bar-percent">{casual_probability:.0%}</div>', unsafe_allow_html=True)

    st.markdown('<div class="bar-label">🎁 Enthusiast</div>', unsafe_allow_html=True)
    st.progress(enthusiast_probability)
    st.markdown(f'<div class="bar-percent">{enthusiast_probability:.0%}</div>', unsafe_allow_html=True)

    st.markdown('<div class="bar-label">👑 Hardcore</div>', unsafe_allow_html=True)
    st.progress(hardcore_probability)
    st.markdown(f'<div class="bar-percent">{hardcore_probability:.0%}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-footer">
        <div class="footer-rule">━━━━━━━━━━━━━━━━━━━━━━</div>
        <div class="footer-title">🎁 POP MART Collector Predictor</div>
        <div>Random Forest Classifier</div>
        <div>FastAPI REST API</div>
        <div>Streamlit Interface</div>
        <div>Synthetic Dataset (2,000 Collectors)</div>
        <div>Made for MLOps Mini Project</div>
        <div class="footer-rule">━━━━━━━━━━━━━━━━━━━━━━</div>
    </div>
    """,
    unsafe_allow_html=True,
)
