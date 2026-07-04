"""
==================================================================================
Home Value AI —  Premium AI Real-Estate Valuation Platform
==================================================================================
A Streamlit single-file application designed to feel like a commercial SaaS
product (Stripe / Linear / Vercel / Apple-level polish) rather than a default
Streamlit form.

THE MACHINE LEARNING PIPELINE IS UNCHANGED FROM THE ORIGINAL NOTEBOOK:

    1. Load Bengaluru_House_Data.csv
    2. Drop columns: area_type, society, balcony, availability
    3. Drop rows with any remaining nulls
    4. Derive `bedrooms` from `size` (e.g. "2 BHK" -> 2), drop `size`
    5. Convert `total_sqft` ranges ("2100-2850") to their numeric average
    6. Remove total_sqft outliers (10th-90th percentile band)
    7. Remove bathroom outliers (bath < 12)
    8. Compute price_per_sqft, remove outliers (10th-90th percentile), drop it
    9. Strip location whitespace; bucket locations with <= 10 listings as "other"
    10. Features (x): location, total_sqft, bath, bedrooms | Target (y): price
    11. Model: make_pipeline(category_encoders.OneHotEncoder(), LinearRegression())

Everything visual below this point — the multi-step wizard, AI analysis
animation, dashboard metrics, gauges, and charts — is presentation/UX layered
on top of that exact, unmodified pipeline. Any "score", "rating", or
"appreciation" metric derived beyond the raw model price is clearly labelled
as an illustrative analytical add-on computed from the prediction and dataset
statistics, not a separate model output.
==================================================================================
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

try:
    from category_encoders import OneHotEncoder as CategoryOneHotEncoder

    _ENCODER_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover
    CategoryOneHotEncoder = None  # type: ignore
    _ENCODER_IMPORT_ERROR = str(_exc)


CSV_PATH = "Bengaluru_House_Data.csv"

st.set_page_config(
    page_title="Home Value AI — Real Estate Valuation Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================================
# THEME DEFINITIONS
# ==================================================================================
THEMES: Dict[str, Dict[str, str]] = {
    "Midnight": {
        "bg_a": "#05060f",
        "bg_b": "#0b0d1e",
        "bg_c": "#11142b",
        "text": "#eef1fb",
        "muted": "#98a2c3",
        "surface": "rgba(255,255,255,0.045)",
        "surface_border": "rgba(255,255,255,0.09)",
        "accent1": "#7c5cff",
        "accent2": "#22d3ee",
        "accent3": "#f472b6",
        "success": "#34d399",
        "warn": "#fbbf24",
    },
    "Aurora": {
        "bg_a": "#f4f6ff",
        "bg_b": "#eef1ff",
        "bg_c": "#e7ecff",
        "text": "#12142b",
        "muted": "#5a6182",
        "surface": "rgba(255,255,255,0.55)",
        "surface_border": "rgba(20,20,50,0.08)",
        "accent1": "#5b3df0",
        "accent2": "#0ea5c9",
        "accent3": "#db2777",
        "success": "#059669",
        "warn": "#d97706",
    },
}


def get_theme() -> Dict[str, str]:
    return THEMES[st.session_state.get("theme_name", "Midnight")]


# ==================================================================================
# GLOBAL CSS INJECTION
# ==================================================================================
def inject_css() -> None:
    t = get_theme()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {{
            --bg-a: {t['bg_a']};
            --bg-b: {t['bg_b']};
            --bg-c: {t['bg_c']};
            --text: {t['text']};
            --muted: {t['muted']};
            --surface: {t['surface']};
            --surface-border: {t['surface_border']};
            --accent1: {t['accent1']};
            --accent2: {t['accent2']};
            --accent3: {t['accent3']};
            --success: {t['success']};
            --warn: {t['warn']};
        }}

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        h1, h2, h3, h4, .grotesk {{ font-family: 'Space Grotesk', sans-serif; }}

        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}

        .stApp {{
            background:
                radial-gradient(circle at 15% -10%, color-mix(in srgb, var(--accent1) 22%, transparent) 0%, transparent 45%),
                radial-gradient(circle at 90% 10%, color-mix(in srgb, var(--accent2) 18%, transparent) 0%, transparent 40%),
                linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 50%, var(--bg-c) 100%);
            color: var(--text);
        }}
        p, span, label, li, div {{ color: var(--text); }}
        .muted {{ color: var(--muted) !important; }}

        /* ---------------- NAV BAR ---------------- */
        .topnav {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.9rem 1.6rem; border-radius: 18px;
            background: var(--surface); border: 1px solid var(--surface-border);
            backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
            margin-bottom: 1.6rem;
        }}
        .brand {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem;
            display: flex; align-items: center; gap: 0.5rem; letter-spacing: -0.02em;
        }}
        .brand-dot {{
            width: 10px; height: 10px; border-radius: 3px;
            background: linear-gradient(135deg, var(--accent1), var(--accent2));
            box-shadow: 0 0 16px color-mix(in srgb, var(--accent1) 70%, transparent);
        }}
        .nav-pill {{
            font-size: 0.78rem; font-weight: 600; padding: 0.32rem 0.85rem; border-radius: 999px;
            background: color-mix(in srgb, var(--success) 16%, transparent);
            border: 1px solid color-mix(in srgb, var(--success) 40%, transparent);
            color: var(--success);
        }}

        /* ---------------- HERO ---------------- */
        .hero {{
            position: relative; overflow: hidden;
            border-radius: 28px; padding: 4rem 2.8rem;
            background: linear-gradient(135deg, color-mix(in srgb, var(--accent1) 85%, black 10%) 0%, color-mix(in srgb, var(--accent2) 70%, black 5%) 100%);
            margin-bottom: 2.2rem; box-shadow: 0 30px 80px -20px color-mix(in srgb, var(--accent1) 55%, transparent);
            animation: fadeUp 0.7s ease-out;
        }}
        .hero::after {{
            content: ""; position: absolute; inset: 0;
            background-image: radial-gradient(circle, rgba(255,255,255,0.35) 1px, transparent 1px);
            background-size: 26px 26px; opacity: 0.12; pointer-events: none;
        }}
        .hero-eyebrow {{
            display: inline-flex; align-items: center; gap: 0.4rem;
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
            background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.3);
            padding: 0.4rem 0.9rem; border-radius: 999px; color: #fff; backdrop-filter: blur(6px);
        }}
        .hero-title {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.03em;
            font-size: 3.4rem; line-height: 1.08; color: #fff; margin: 1.1rem 0 0.9rem 0;
            text-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }}
        .hero-title .grad {{
            background: linear-gradient(90deg, #fff, #ffe9ff);
            -webkit-background-clip: text; background-clip: text;
        }}
        .hero-sub {{
            font-size: 1.12rem; color: rgba(255,255,255,0.9); max-width: 620px;
            line-height: 1.65; font-weight: 400;
        }}
        .hero-stats {{ display: flex; gap: 2.4rem; margin-top: 2rem; flex-wrap: wrap; }}
        .hero-stat-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #fff; }}
        .hero-stat-label {{ font-size: 0.78rem; color: rgba(255,255,255,0.75); text-transform: uppercase; letter-spacing: 0.06em; }}

        .float-card {{
            position: absolute; padding: 0.85rem 1.1rem; border-radius: 16px;
            background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28);
            backdrop-filter: blur(14px); box-shadow: 0 18px 40px rgba(0,0,0,0.22);
            font-size: 0.82rem; font-weight: 600; color: #fff; animation: floaty 5s ease-in-out infinite;
        }}
        .fc1 {{ top: 12%; right: 8%; animation-delay: 0s; }}
        .fc2 {{ top: 52%; right: 20%; animation-delay: 1.2s; }}
        .fc3 {{ top: 32%; right: -1%; animation-delay: 2.1s; }}

        @keyframes floaty {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-14px); }}
        }}
        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(22px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes popIn {{
            0% {{ opacity: 0; transform: scale(0.9); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: -400px 0; }}
            100% {{ background-position: 400px 0; }}
        }}

        /* ---------------- GLASS CARD ---------------- */
        .glass {{
            background: var(--surface); border: 1px solid var(--surface-border);
            border-radius: 20px; padding: 1.5rem 1.6rem; backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px); transition: all 0.25s ease;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        }}
        .glass:hover {{
            transform: translateY(-3px);
            border-color: color-mix(in srgb, var(--accent1) 45%, var(--surface-border));
            box-shadow: 0 18px 44px color-mix(in srgb, var(--accent1) 18%, transparent);
        }}
        .glass-tight {{ padding: 1.1rem 1.2rem; }}

        .section-title {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5rem;
            margin: 0.4rem 0 1.1rem 0; display: flex; align-items: center; gap: 0.55rem;
        }}
        .section-eyebrow {{
            font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
            color: var(--accent2); margin-bottom: 0.3rem;
        }}

        /* ---------------- STEP WIZARD ---------------- */
        .stepper {{ display: flex; align-items: center; margin: 0.4rem 0 2rem 0; }}
        .step-item {{ display: flex; align-items: center; flex: 1; }}
        .step-circle {{
            width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center;
            justify-content: center; font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
            border: 2px solid var(--surface-border); background: var(--surface); color: var(--muted);
            transition: all 0.3s ease;
        }}
        .step-circle.active {{
            background: linear-gradient(135deg, var(--accent1), var(--accent2));
            border-color: transparent; color: #fff; box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent1) 18%, transparent);
        }}
        .step-circle.done {{
            background: var(--success); border-color: transparent; color: #fff;
        }}
        .step-label {{ margin-left: 0.6rem; font-size: 0.85rem; font-weight: 600; color: var(--muted); white-space: nowrap; }}
        .step-label.active {{ color: var(--text); }}
        .step-line {{ flex: 1; height: 2px; background: var(--surface-border); margin: 0 0.9rem; border-radius: 2px; overflow: hidden; }}
        .step-line-fill {{ height: 100%; background: linear-gradient(90deg, var(--accent1), var(--accent2)); transition: width 0.4s ease; }}

        /* ---------------- BUTTONS ---------------- */
        div.stButton > button {{
            background: linear-gradient(120deg, var(--accent1), var(--accent2));
            color: #fff !important; font-weight: 700; border: none; border-radius: 14px;
            padding: 0.75rem 1.6rem; box-shadow: 0 12px 28px color-mix(in srgb, var(--accent1) 40%, transparent);
            transition: all 0.22s ease; width: 100%;
        }}
        div.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 16px 36px color-mix(in srgb, var(--accent1) 55%, transparent); }}
        div.stButton > button:active {{ transform: translateY(0px) scale(0.98); }}
        div.stButton > button p {{ color: #fff !important; }}

        button[kind="secondary"], .secondary-btn div.stButton > button {{
            background: var(--surface) !important; color: var(--text) !important;
            border: 1px solid var(--surface-border) !important; box-shadow: none !important;
        }}

        /* ---------------- BADGES ---------------- */
        .badge {{
            display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.76rem; font-weight: 700;
            padding: 0.3rem 0.8rem; border-radius: 999px; margin: 0.15rem;
        }}
        .badge-purple {{ background: color-mix(in srgb, var(--accent1) 18%, transparent); color: var(--accent1); border: 1px solid color-mix(in srgb, var(--accent1) 40%, transparent); }}
        .badge-cyan {{ background: color-mix(in srgb, var(--accent2) 18%, transparent); color: var(--accent2); border: 1px solid color-mix(in srgb, var(--accent2) 40%, transparent); }}
        .badge-green {{ background: color-mix(in srgb, var(--success) 18%, transparent); color: var(--success); border: 1px solid color-mix(in srgb, var(--success) 40%, transparent); }}
        .badge-amber {{ background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn); border: 1px solid color-mix(in srgb, var(--warn) 45%, transparent); }}
        .badge-pulse {{ animation: pulse 2s ease-in-out infinite; }}

        /* ---------------- KPI CARD ---------------- */
        .kpi {{
            background: var(--surface); border: 1px solid var(--surface-border); border-radius: 18px;
            padding: 1.2rem 1.3rem; text-align: left; transition: transform 0.2s ease; position: relative; overflow: hidden;
        }}
        .kpi:hover {{ transform: translateY(-3px); }}
        .kpi-icon {{ font-size: 1.3rem; margin-bottom: 0.4rem; }}
        .kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; }}
        .kpi-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.15rem; }}
        .kpi-delta {{ font-size: 0.78rem; font-weight: 700; margin-top: 0.35rem; }}
        .kpi-delta.up {{ color: var(--success); }}
        .kpi-delta.down {{ color: #f87171; }}

        /* ---------------- PREDICTION HERO CARD ---------------- */
        .predict-card {{
            border-radius: 28px; padding: 3rem 2.5rem; text-align: center; position: relative; overflow: hidden;
            background: linear-gradient(135deg, color-mix(in srgb, var(--success) 75%, black 10%), color-mix(in srgb, var(--accent2) 65%, black 10%));
            box-shadow: 0 30px 70px -18px color-mix(in srgb, var(--success) 50%, transparent);
            animation: popIn 0.5s cubic-bezier(.26,1.36,.44,1);
            margin-bottom: 1.6rem;
        }}
        .predict-eyebrow {{ font-size: 0.85rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(255,255,255,0.85); }}
        .predict-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 3.6rem; font-weight: 700; color: #fff; margin: 0.3rem 0; }}
        .predict-sub {{ color: rgba(255,255,255,0.85); font-size: 1rem; }}

        /* ---------------- PROGRESS BAR (custom) ---------------- */
        .pbar-track {{ width: 100%; height: 8px; border-radius: 999px; background: var(--surface-border); overflow: hidden; }}
        .pbar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent1), var(--accent2)); transition: width 0.3s ease; }}

        /* ---------------- LOADING MESSAGE ---------------- */
        .ai-loading-box {{
            text-align: center; padding: 2.4rem 1rem;
        }}
        .ai-orb {{
            width: 70px; height: 70px; border-radius: 50%; margin: 0 auto 1.2rem auto;
            background: conic-gradient(from 0deg, var(--accent1), var(--accent2), var(--accent3), var(--accent1));
            animation: spin 1.4s linear infinite; box-shadow: 0 0 40px color-mix(in srgb, var(--accent1) 45%, transparent);
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .ai-loading-text {{ font-size: 1.05rem; font-weight: 600; color: var(--text); }}

        /* ---------------- DIVIDER ---------------- */
        .divider-grad {{ height: 1px; border: none; margin: 2.2rem 0; background: linear-gradient(90deg, transparent, var(--surface-border), transparent); }}

        /* ---------------- FOOTER ---------------- */
        .footer {{
            margin-top: 2.5rem; padding: 2.2rem; border-radius: 22px;
            background: var(--surface); border: 1px solid var(--surface-border); text-align: center;
        }}
        .footer-badge {{
            display: inline-block; padding: 0.32rem 0.85rem; border-radius: 999px; margin: 0.2rem;
            background: color-mix(in srgb, var(--accent1) 12%, transparent);
            border: 1px solid color-mix(in srgb, var(--accent1) 30%, transparent); font-size: 0.78rem; font-weight: 600;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 12px; padding: 0.5rem 1rem; background: var(--surface);
            border: 1px solid var(--surface-border); color: var(--text);
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(120deg, var(--accent1), var(--accent2)) !important; color: #fff !important;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--bg-c), var(--bg-a));
            border-right: 1px solid var(--surface-border);
        }}

        [data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; }}

        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: var(--surface-border); border-radius: 8px; }}

        @media (max-width: 768px) {{
            .hero-title {{ font-size: 2.1rem; }}
            .predict-value {{ font-size: 2.3rem; }}
            .float-card {{ display: none; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# DATA / MODEL PIPELINE — UNCHANGED LOGIC FROM THE NOTEBOOK
# ==================================================================================
@st.cache_data(show_spinner=False)
def load_raw_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def convert_sqft_to_num(x: Any) -> Optional[float]:
    if isinstance(x, float):
        return x
    tokens = str(x).split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Exact replica of the notebook's cleaning / feature-engineering pipeline."""
    df1 = raw_df.drop(columns=["area_type", "society", "balcony", "availability"])
    df1.dropna(inplace=True)

    df1["bedrooms"] = df1["size"].apply(lambda x: int(str(x).split(" ")[0]))
    df1 = df1.drop(columns="size")

    df4 = df1.copy()
    df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)
    df4.dropna(inplace=True)

    low, high = df4["total_sqft"].quantile([0.1, 0.9])
    df4 = df4[df4["total_sqft"].between(low, high)]

    df4 = df4[df4["bath"] < 12]

    df4["price_per_sqft"] = df4["price"] * 100000 / df4["total_sqft"]
    low, high = df4["price_per_sqft"].quantile([0.1, 0.9])
    df4 = df4[df4["price_per_sqft"].between(low, high)]

    df4.location = df4.location.apply(lambda x: x.strip())
    location_stats = df4["location"].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df4.location = df4.location.apply(
        lambda x: "other" if x in location_stats_less_than_10 else x
    )

    df4 = df4.drop(columns="price_per_sqft")
    return df4


@st.cache_resource(show_spinner=False)
def train_model(_clean_df: pd.DataFrame) -> Dict[str, Any]:
    """Fit make_pipeline(OneHotEncoder(), LinearRegression()) — identical to notebook."""
    if CategoryOneHotEncoder is None:
        raise ImportError(
            "The 'category_encoders' package is required to reproduce the "
            "notebook's exact encoding step. Install it with: "
            f"pip install category_encoders. Original error: {_ENCODER_IMPORT_ERROR}"
        )

    x = _clean_df.drop(["price"], axis="columns")
    y = _clean_df.price

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=10
    )

    model = make_pipeline(CategoryOneHotEncoder(), LinearRegression())
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mape = float(mean_absolute_percentage_error(y_test, y_pred))

    # Attempt to extract feature importances from the fitted LinearRegression.
    feature_importance: Optional[pd.DataFrame] = None
    try:
        ohe = model.named_steps["onehotencoder"]
        lr = model.named_steps["linearregression"]
        try:
            encoded_cols = list(ohe.get_feature_names_out())
        except Exception:
            encoded_cols = list(ohe.feature_names)  # older category_encoders versions
        coefs = lr.coef_
        fi_df = pd.DataFrame({"feature": encoded_cols, "coef": coefs})
        fi_df["abs_coef"] = fi_df["coef"].abs()

        loc_mask = fi_df["feature"].str.contains("location", case=False, na=False)
        core_df = fi_df[~loc_mask].copy()
        loc_row = pd.DataFrame(
            [{"feature": "location (avg. effect)", "coef": fi_df.loc[loc_mask, "coef"].mean(),
            "abs_coef": fi_df.loc[loc_mask, "abs_coef"].mean()}]
        )
        feature_importance = pd.concat([core_df, loc_row], ignore_index=True)
        feature_importance = feature_importance.sort_values("abs_coef", ascending=False)
    except Exception:
        feature_importance = None

    return {
        "model": model,
        "mape": mape,
        "feature_columns": list(x.columns),
        "locations": sorted(x["location"].unique().tolist()),
        "sqft_min": float(x["total_sqft"].min()),
        "sqft_max": float(x["total_sqft"].max()),
        "bath_min": int(x["bath"].min()),
        "bath_max": int(x["bath"].max()),
        "bedrooms_min": int(x["bedrooms"].min()),
        "bedrooms_max": int(x["bedrooms"].max()),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "feature_importance": feature_importance,
    }


def predict_price(model: Any, input_data: Dict[str, Any]) -> float:
    """Identical to the notebook's predict_price(): DataFrame([input]) -> model.predict()[0]."""
    input_df = pd.DataFrame([input_data])
    return float(model.predict(input_df)[0])


# ==================================================================================
# ANALYTICS HELPERS (illustrative, derived from prediction + dataset — NOT model outputs)
# ==================================================================================
def compute_insights(clean_df: pd.DataFrame, model_info: Dict[str, Any],
                    inputs: Dict[str, Any], predicted_price: float) -> Dict[str, Any]:
    price_series = clean_df["price"]
    ppsf_series = clean_df["price"] * 100000 / clean_df["total_sqft"]

    predicted_ppsf = predicted_price * 100000 / max(inputs["total_sqft"], 1)
    ppsf_percentile = float((ppsf_series < predicted_ppsf).mean() * 100)
    price_percentile = float((price_series < predicted_price).mean() * 100)

    value_score = max(0.0, min(100.0, 100 - ppsf_percentile))
    affordability = max(0.0, min(100.0, 100 - price_percentile))
    confidence = max(50.0, min(97.0, 100 - model_info["mape"] * 100))

    if value_score >= 80:
        rating, rating_color = "Excellent Value", "badge-green"
    elif value_score >= 60:
        rating, rating_color = "Good Value", "badge-cyan"
    elif value_score >= 40:
        rating, rating_color = "Fair Value", "badge-amber"
    else:
        rating, rating_color = "Premium Priced", "badge-purple"

    if price_percentile < 25:
        category = "Budget"
    elif price_percentile < 75:
        category = "Mid-Range"
    elif price_percentile < 95:
        category = "Premium"
    else:
        category = "Luxury"

    loc_count = int((clean_df["location"] == inputs["location"]).sum())
    popularity_boost = min(loc_count / 50.0, 1.0) * 2.5
    appreciation = round(5.0 + popularity_boost + (value_score / 100) * 3.0, 1)

    land_share = 0.62
    land_value = predicted_price * land_share
    construction_value = predicted_price * (1 - land_share)

    return {
        "predicted_ppsf": predicted_ppsf,
        "value_score": value_score,
        "affordability": affordability,
        "confidence": confidence,
        "rating": rating,
        "rating_color": rating_color,
        "category": category,
        "appreciation": appreciation,
        "land_value": land_value,
        "construction_value": construction_value,
        "price_percentile": price_percentile,
    }


# ==================================================================================
# UI FRAGMENTS
# ==================================================================================
def render_topnav() -> None:
    st.markdown(
        """
        <div class="topnav">
            <div class="brand"><span class="brand-dot"></span> ESTIMATE.AI</div>
            <div class="nav-pill">● Model Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(model_info: Dict[str, Any], clean_df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="float-card fc1">📊 +{len(model_info['locations'])} locations mapped</div>
            <div class="float-card fc2">⚡ Instant valuation</div>
            <div class="float-card fc3">🧠 AI-assisted pricing</div>
            <span class="hero-eyebrow">✦ AI-Powered Valuation Engine</span>
            <div class="hero-title">Know your property's<br><span class="grad">true market value.</span></div>
            <div class="hero-sub">
                Estimate.AI blends regression modelling with real Bengaluru listing
                data to deliver instant, data-backed property valuations — the kind
                of clarity usually locked inside enterprise real-estate platforms.
            </div>
            <div class="hero-stats">
                <div><div class="hero-stat-value">{len(clean_df):,}</div><div class="hero-stat-label">Listings Analyzed</div></div>
                <div><div class="hero-stat-value">{len(model_info['locations'])}</div><div class="hero-stat-label">Locations Covered</div></div>
                <div><div class="hero-stat-value">{(100-model_info['mape']*100):.0f}%</div><div class="hero-stat-label">Avg. Model Confidence</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current: int) -> None:
    steps = ["Property Info", "Location", "Features", "Results"]
    html = '<div class="stepper">'
    for i, label in enumerate(steps, start=1):
        circle_class = "done" if i < current else ("active" if i == current else "")
        label_class = "active" if i <= current else ""
        content = "✓" if i < current else str(i)
        html += f'<div class="step-item"><div class="step-circle {circle_class}">{content}</div>'
        html += f'<div class="step-label {label_class}">{label}</div>'
        if i != len(steps):
            fill = 100 if i < current else 0
            html += f'<div class="step-line"><div class="step-line-fill" style="width:{fill}%;"></div></div>'
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar(model_info: Dict[str, Any], raw_df: pd.DataFrame,
                    clean_df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        theme_choice = st.radio(
            "Theme", options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.theme_name),
            horizontal=True, label_visibility="collapsed",
        )
        if theme_choice != st.session_state.theme_name:
            st.session_state.theme_name = theme_choice
            st.rerun()

        st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)

        st.markdown("### 🧠 Model")
        st.markdown(
            f"""
            <div class="glass glass-tight">
                <b>Algorithm:</b> Linear Regression<br>
                <b>Encoder:</b> One-Hot (category_encoders)<br>
                <b>MAPE:</b> {model_info['mape']*100:.2f}%<br>
                <b>Train / Test:</b> {model_info['n_train']:,} / {model_info['n_test']:,}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🗂️ Dataset")
        st.markdown(
            f"""
            <div class="glass glass-tight">
                <b>Source:</b> Bengaluru_House_Data.csv<br>
                <b>Raw rows:</b> {raw_df.shape[0]:,}<br>
                <b>Clean rows:</b> {len(clean_df):,}<br>
                <b>Locations:</b> {len(model_info['locations'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 👨‍💻 Developer")
        st.markdown(
            """
            <div class="glass glass-tight">
                <b>Manav Sharma</b><br>
                <span class="muted">ML Engineer · Real Estate Analytics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧰 Tech Stack")
        st.markdown(
            """
            <div class="glass glass-tight">
                Python · Pandas · NumPy<br>Scikit-Learn · Category Encoders<br>Streamlit · Plotly
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)

        st.markdown("### 🕘 Prediction History")
        if st.session_state.history:
            for h in reversed(st.session_state.history[-5:]):
                st.markdown(
                    f"""
                    <div class="glass glass-tight" style="margin-bottom:0.5rem;">
                        <b>₹ {h['price']:,.2f} L</b> — {h['location']}<br>
                        <span class="muted" style="font-size:0.75rem;">
                            {h['bedrooms']} BHK · {h['total_sqft']:.0f} sqft · {h['time']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            hist_df = pd.DataFrame(st.session_state.history)
            csv_buf = io.StringIO()
            hist_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download History (CSV)", data=csv_buf.getvalue(),
                file_name="prediction_history.csv", mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="glass glass-tight muted">No predictions yet this session.</div>',
                unsafe_allow_html=True,
            )

        if st.button("🔄 Reset Everything", use_container_width=True):
            for key in ["wizard_step", "inputs", "prediction_result", "insights", "history"]:
                st.session_state.pop(key, None)
            st.rerun()


def render_kpi(icon: str, value: str, label: str, delta: Optional[str] = None,
                delta_up: bool = True) -> str:
    delta_html = ""
    if delta:
        cls = "up" if delta_up else "down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
        <div class="kpi">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            {delta_html}
        </div>
    """


def render_gauge(value: float, title: str, color: str) -> go.Figure:
    t = get_theme()
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 28, "family": "Space Grotesk"}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": t["muted"]},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(248,113,113,0.18)"},
                    {"range": [40, 70], "color": "rgba(251,191,36,0.18)"},
                    {"range": [70, 100], "color": "rgba(52,211,153,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color=t["text"],
    )
    return fig


# ==================================================================================
# WIZARD STEPS
# ==================================================================================
def step_property_info(model_info: Dict[str, Any]) -> None:
    st.markdown('<div class="section-eyebrow">Step 1 of 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏗️ Property Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        total_sqft = st.slider(
            "📐 Total Built-up Area (sqft)",
            min_value=int(model_info["sqft_min"]), max_value=int(model_info["sqft_max"]),
            value=int(st.session_state.inputs.get("total_sqft", (model_info["sqft_min"] + model_info["sqft_max"]) / 2)),
            step=10, help="Total constructed area of the property in square feet.",
        )
    with c2:
        property_type = st.selectbox(
            "🏢 Property Type",
            options=["Apartment", "Independent House", "Villa", "Plot"],
            index=["Apartment", "Independent House", "Villa", "Plot"].index(
                st.session_state.inputs.get("property_type", "Apartment")
            ),
            help="For your reference only — the current model estimates price from "
                "structural features (area, bedrooms, bathrooms, location) regardless of type.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.inputs["total_sqft"] = float(total_sqft)
    st.session_state.inputs["property_type"] = property_type

    st.markdown("<br>", unsafe_allow_html=True)
    _, colnext = st.columns([3, 1])
    with colnext:
        if st.button("Continue →", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()


def step_location(model_info: Dict[str, Any], clean_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-eyebrow">Step 2 of 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📍 Location Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    locations = model_info["locations"]
    default_loc = st.session_state.inputs.get("location", locations[0])
    location = st.selectbox(
        "Select Locality",
        options=locations,
        index=locations.index(default_loc) if default_loc in locations else 0,
        help="Neighborhoods with fewer than 10 historical listings are grouped as 'other'.",
    )

    loc_count = int((clean_df["location"] == location).sum())
    loc_avg_price = clean_df.loc[clean_df["location"] == location, "price"].mean()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            render_kpi("📦", f"{loc_count}", "Listings in this area"),
            unsafe_allow_html=True,
        )
    with c2:
        avg_txt = f"₹ {loc_avg_price:,.1f} L" if pd.notna(loc_avg_price) else "N/A"
        st.markdown(
            render_kpi("💰", avg_txt, "Avg. price in this area"),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.inputs["location"] = location

    st.markdown("<br>", unsafe_allow_html=True)
    colback, _, colnext = st.columns([1, 2, 1])
    with colback:
        if st.button("← Back", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with colnext:
        if st.button("Continue →", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()


def step_features(model_info: Dict[str, Any]) -> None:
    st.markdown('<div class="section-eyebrow">Step 3 of 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧩 Additional Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        bedrooms = st.number_input(
            "🛏️ Bedrooms (BHK)",
            min_value=int(model_info["bedrooms_min"]), max_value=int(model_info["bedrooms_max"]),
            value=int(st.session_state.inputs.get("bedrooms", min(2, model_info["bedrooms_max"]))),
            step=1, help="Number of bedrooms (BHK count).",
        )
        furnishing = st.select_slider(
            "🛋️ Furnishing",
            options=["Unfurnished", "Semi-Furnished", "Fully Furnished"],
            value=st.session_state.inputs.get("furnishing", "Semi-Furnished"),
            help="Reference detail only — not used by the current prediction model.",
        )
    with c2:
        bath = st.number_input(
            "🛁 Bathrooms",
            min_value=int(model_info["bath_min"]), max_value=int(model_info["bath_max"]),
            value=int(st.session_state.inputs.get("bath", min(2, model_info["bath_max"]))),
            step=1, help="Total number of bathrooms.",
        )
        parking = st.toggle(
            "🚗 Covered Parking Available",
            value=st.session_state.inputs.get("parking", True),
            help="Reference detail only — not used by the current prediction model.",
        )

    if bath > bedrooms + 3:
        st.markdown(
            '<span class="badge badge-amber">⚠️ Unusually high bathroom count for this BHK</span>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.inputs["bedrooms"] = int(bedrooms)
    st.session_state.inputs["bath"] = float(bath)
    st.session_state.inputs["furnishing"] = furnishing
    st.session_state.inputs["parking"] = parking

    st.markdown("<br>", unsafe_allow_html=True)
    colback, _, colnext = st.columns([1, 2, 1])
    with colback:
        if st.button("← Back", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with colnext:
        run_clicked = st.button("✨ Run AI Analysis", use_container_width=True)

    if run_clicked:
        run_ai_analysis(model_info)


def run_ai_analysis(model_info: Dict[str, Any]) -> None:
    inputs = st.session_state.inputs
    model_input = {
        "location": inputs["location"],
        "total_sqft": inputs["total_sqft"],
        "bath": inputs["bath"],
        "bedrooms": inputs["bedrooms"],
    }

    messages = [
        "🔎 Analyzing market trends...",
        "🏘️ Evaluating property features...",
        "📊 Comparing nearby properties...",
        "🧮 Calculating estimated value...",
        "✅ Finalizing prediction...",
    ]

    placeholder = st.empty()
    progress = st.empty()
    for i, msg in enumerate(messages):
        pct = int(((i + 1) / len(messages)) * 100)
        placeholder.markdown(
            f"""
            <div class="ai-loading-box">
                <div class="ai-orb"></div>
                <div class="ai-loading-text">{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress.markdown(
            f"""
            <div class="pbar-track"><div class="pbar-fill" style="width:{pct}%;"></div></div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.45)

    try:
        price = predict_price(model_info["model"], model_input)
        price = max(price, 0.0)
    except Exception as exc:  # noqa: BLE001
        placeholder.empty()
        progress.empty()
        st.error(f"❌ Prediction failed: {exc}")
        return

    placeholder.empty()
    progress.empty()

    st.session_state.prediction_result = price
    st.session_state.final_inputs = dict(inputs)
    st.session_state.history.append(
        {
            "price": price,
            "location": inputs["location"],
            "total_sqft": inputs["total_sqft"],
            "bath": inputs["bath"],
            "bedrooms": inputs["bedrooms"],
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )
    st.session_state.wizard_step = 4
    st.rerun()


# ==================================================================================
# RESULTS DASHBOARD
# ==================================================================================
def step_results(model_info: Dict[str, Any], clean_df: pd.DataFrame) -> None:
    inputs = st.session_state.final_inputs
    price = st.session_state.prediction_result
    insights = compute_insights(clean_df, model_info, inputs, price)

    st.markdown(
        f"""
        <div class="predict-card">
            <div class="predict-eyebrow">Estimated Market Value</div>
            <div class="predict-value">₹ {price:,.2f} Lakhs</div>
            <div class="predict-sub">≈ ₹ {price*100000:,.0f}  ·  ₹ {insights['predicted_ppsf']:,.0f} / sqft</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    badges = f"""
        <span class="badge {insights['rating_color']}">💎 {insights['rating']}</span>
        <span class="badge badge-cyan">🏷️ {insights['category']}</span>
        <span class="badge badge-purple badge-pulse">🤖 {insights['confidence']:.0f}% AI Confidence</span>
    """
    st.markdown(f'<div style="text-align:center; margin-bottom:1.4rem;">{badges}</div>', unsafe_allow_html=True)

    # ---- Property Summary Pills ----
    st.markdown('<div class="section-title">📋 Property Summary</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    summary = [
        ("📍", inputs["location"], "Location"),
        ("📐", f"{inputs['total_sqft']:.0f} sqft", "Area"),
        ("🛏️", f"{inputs['bedrooms']} BHK", "Bedrooms"),
        ("🛁", f"{int(inputs['bath'])}", "Bathrooms"),
        ("🏢", inputs.get("property_type", "-"), "Type"),
        ("🛋️", inputs.get("furnishing", "-"), "Furnishing"),
    ]
    for col, (icon, val, label) in zip(cols, summary):
        with col:
            st.markdown(render_kpi(icon, str(val), label), unsafe_allow_html=True)

    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)

    # ---- Dashboard Metrics ----
    st.markdown('<div class="section-title">📈 AI-Powered Insights</div>', unsafe_allow_html=True)
    st.caption(
        "Illustrative analytics derived from the predicted price and dataset "
        "statistics — supplementary to the core regression estimate above."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(render_gauge(insights["value_score"], "Property Score", get_theme()["accent1"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(render_gauge(insights["affordability"], "Affordability Meter", get_theme()["accent2"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(render_gauge(insights["confidence"], "AI Confidence", get_theme()["success"]),
                        use_container_width=True, config={"displayModeBar": False})

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(render_kpi("📈", f"+{insights['appreciation']}%", "Est. Annual Appreciation",
                                delta="Indicative", delta_up=True), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi("🏷️", insights["category"], "Price Category"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi("📊", f"Top {100-insights['price_percentile']:.0f}%", "Market Position"),
                    unsafe_allow_html=True)

    # ---- Price Breakdown ----
    st.markdown("<br>", unsafe_allow_html=True)
    bc1, bc2 = st.columns([1, 1])
    with bc1:
        st.markdown('<div class="section-eyebrow">Illustrative Breakdown</div>', unsafe_allow_html=True)
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Land / Location Value", "Construction Value"],
                    values=[insights["land_value"], insights["construction_value"]],
                    hole=0.6,
                    marker=dict(colors=[get_theme()["accent1"], get_theme()["accent2"]]),
                )
            ]
        )
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color=get_theme()["text"],
            legend=dict(orientation="h", y=-0.1),
            annotations=[dict(text=f"₹{price:,.1f}L", x=0.5, y=0.5, font_size=18, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with bc2:
        st.markdown('<div class="section-eyebrow">Export</div>', unsafe_allow_html=True)
        report = (
            f"ESTIMATE.AI — Property Valuation Report\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Location: {inputs['location']}\n"
            f"Area: {inputs['total_sqft']:.0f} sqft\n"
            f"Bedrooms: {inputs['bedrooms']} BHK\n"
            f"Bathrooms: {int(inputs['bath'])}\n"
            f"Property Type: {inputs.get('property_type','-')}\n"
            f"Furnishing: {inputs.get('furnishing','-')}\n\n"
            f"Estimated Price: Rs. {price:,.2f} Lakhs\n"
            f"Price per sqft: Rs. {insights['predicted_ppsf']:,.0f}\n"
            f"Property Score: {insights['value_score']:.0f}/100\n"
            f"Rating: {insights['rating']}\n"
            f"Price Category: {insights['category']}\n"
            f"AI Confidence: {insights['confidence']:.0f}%\n"
            f"Indicative Appreciation: {insights['appreciation']}% / year\n"
        )
        st.download_button(
            "⬇️ Download Prediction Report (.txt)", data=report,
            file_name="property_valuation_report.txt", mime="text/plain",
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔁 Start New Valuation", use_container_width=True):
            st.session_state.wizard_step = 1
            st.session_state.prediction_result = None
            st.rerun()


# ==================================================================================
# ANALYTICS TAB (dataset-wide, shown below the wizard on every step)
# ==================================================================================
def render_analytics(clean_df: pd.DataFrame, model_info: Dict[str, Any]) -> None:
    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Market Analytics Dashboard</div>', unsafe_allow_html=True)

    t = get_theme()
    plot_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=t["text"]),
    )

    tabs = st.tabs(["💰 Price Distribution", "📐 Area vs Price", "🛏️ BHK Mix",
                    "📍 Top Locations", "🧬 Feature Importance"])

    with tabs[0]:
        fig = px.histogram(clean_df, x="price", nbins=40, color_discrete_sequence=[t["accent1"]],
                            title="Distribution of Property Prices (₹ Lakhs)")
        fig.update_layout(**plot_layout, bargap=0.06)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        sample = clean_df.sample(min(1500, len(clean_df)), random_state=42)
        fig = px.scatter(sample, x="total_sqft", y="price", color="bedrooms",
                        color_continuous_scale="Viridis", opacity=0.7,
                        title="Area (sqft) vs Price (₹ Lakhs)")
        fig.update_layout(**plot_layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        bhk_counts = clean_df["bedrooms"].value_counts().sort_index()
        fig = px.bar(x=bhk_counts.index.astype(str), y=bhk_counts.values,
                    labels={"x": "Bedrooms (BHK)", "y": "Listings"},
                    color=bhk_counts.values, color_continuous_scale="Purples",
                    title="Listings by Bedroom (BHK) Count")
        fig.update_layout(**plot_layout, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        top_loc = clean_df["location"].value_counts().head(15).sort_values()
        fig = go.Figure(go.Bar(x=top_loc.values, y=top_loc.index, orientation="h",
                                marker=dict(color=top_loc.values, colorscale="Tealgrn")))
        fig.update_layout(title="Top 15 Most Listed Locations", xaxis_title="Listings",
                            yaxis_title="", **plot_layout)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        fi = model_info.get("feature_importance")
        if fi is not None and not fi.empty:
            fig = px.bar(fi, x="abs_coef", y="feature", orientation="h",
                        color="abs_coef", color_continuous_scale="Sunset",
                        title="Relative Feature Impact on Predicted Price (|coefficient|)")
            fig.update_layout(**plot_layout, coloraxis_showscale=False,
                                yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Derived directly from the fitted LinearRegression coefficients. "
                "Location coefficients are averaged into a single bar for readability."
            )
        else:
            st.info("Feature importance could not be extracted from this encoder version.")


# ==================================================================================
# STATIC CONTENT: INSIGHTS / FAQ / ABOUT
# ==================================================================================
def render_market_trends(clean_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">📰 Market Trends</div>', unsafe_allow_html=True)
    top_loc = clean_df.groupby("location")["price"].mean().sort_values(ascending=False).head(3)
    cheap_loc = clean_df.groupby("location")["price"].mean().sort_values().head(3)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="glass">
                <b>🏆 Most Expensive Areas</b><br><br>
                {"<br>".join(f"{i+1}. {loc} — ₹{val:,.1f}L" for i, (loc, val) in enumerate(top_loc.items()))}
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="glass">
                <b>💡 Most Affordable Areas</b><br><br>
                {"<br>".join(f"{i+1}. {loc} — ₹{val:,.1f}L" for i, (loc, val) in enumerate(cheap_loc.items()))}
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="glass">
                <b>📌 Dataset Snapshot</b><br><br>
                Median price: ₹{clean_df['price'].median():,.1f}L<br>
                Median area: {clean_df['total_sqft'].median():.0f} sqft<br>
                Most common BHK: {int(clean_df['bedrooms'].mode()[0])} BHK
            </div>""",
            unsafe_allow_html=True,
        )


def render_faq() -> None:
    st.markdown('<div class="section-title">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)
    faqs = [
        ("How is the price estimated?",
        "A linear regression model is trained on thousands of real Bengaluru "
        "listings using location, total area, bathrooms, and bedrooms as inputs. "
        "Location is one-hot encoded before being fed into the model."),
        ("How accurate is the prediction?",
        "The model's mean absolute percentage error (MAPE) is shown in the "
        "sidebar and used to compute the AI Confidence score. Treat the output "
        "as a data-driven estimate, not a certified valuation."),
        ("What do Property Score and Affordability Meter mean?",
        "These are illustrative analytics computed from the predicted price and "
        "the price-per-sqft distribution of comparable listings — they are not "
        "separate outputs of the regression model."),
        ("Why are furnishing and parking not affecting the price?",
        "The underlying dataset and trained model only use location, area, "
        "bathrooms, and bedrooms. Furnishing and parking are captured for your "
        "reference but are not part of the current model's feature set."),
        ("Can I export my results?",
        "Yes — use the 'Download Prediction Report' button on the results page, "
        "or export your full session history from the sidebar."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(a)


def render_about_footer(model_info: Dict[str, Any]) -> None:
    st.markdown('<div class="section-title">🧠 About the Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="glass">
                <b>Model:</b> Bengaluru House Price Estimator<br>
                <b>Algorithm:</b> Linear Regression (scikit-learn)<br>
                <b>Encoding:</b> One-Hot Encoding via <code>category_encoders</code><br>
                <b>Pipeline:</b> <code>make_pipeline(OneHotEncoder(), LinearRegression())</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass">
                <b>Features:</b> {", ".join(model_info['feature_columns'])}<br>
                <b>Target:</b> price (₹ Lakhs)<br>
                <b>MAPE:</b> {model_info['mape']*100:.2f}%<br>
                <b>Split:</b> 80/20, random_state=10
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            <div style="font-weight:700; margin-bottom:0.8rem;">🛠️ Built With</div>
            <span class="footer-badge">🐍 Python</span>
            <span class="footer-badge">🐼 Pandas</span>
            <span class="footer-badge">🔢 NumPy</span>
            <span class="footer-badge">🤖 Scikit-Learn</span>
            <span class="footer-badge">🎈 Streamlit</span>
            <span class="footer-badge">📈 Plotly</span>
            <div class="muted" style="margin-top:1rem; font-size:0.85rem;">
                Designed &amp; developed by <b>Shreya Mahajan</b> 
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# MAIN
# ==================================================================================
def init_session_state(model_info: Dict[str, Any]) -> None:
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "Midnight"
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "inputs" not in st.session_state:
        st.session_state.inputs = {
            "total_sqft": (model_info["sqft_min"] + model_info["sqft_max"]) / 2,
            "location": model_info["locations"][0],
            "bedrooms": min(2, model_info["bedrooms_max"]),
            "bath": min(2, model_info["bath_max"]),
            "property_type": "Apartment",
            "furnishing": "Semi-Furnished",
            "parking": True,
        }
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "final_inputs" not in st.session_state:
        st.session_state.final_inputs = None
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    try:
        raw_df = load_raw_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"❌ Could not find '{CSV_PATH}'. Place it next to app.py.")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Failed to load dataset: {exc}")
        st.stop()

    try:
        clean_df = preprocess_data(raw_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Preprocessing failed: {exc}")
        st.stop()

    try:
        model_info = train_model(clean_df)
    except ImportError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Model training failed: {exc}")
        st.stop()

    init_session_state(model_info)
    inject_css()
    render_topnav()

    if st.session_state.wizard_step != 4:
        render_hero(model_info, clean_df)

    render_sidebar(model_info, raw_df, clean_df)

    render_stepper(st.session_state.wizard_step)

    step = st.session_state.wizard_step
    if step == 1:
        step_property_info(model_info)
    elif step == 2:
        step_location(model_info, clean_df)
    elif step == 3:
        step_features(model_info)
    elif step == 4:
        step_results(model_info, clean_df)

    render_analytics(clean_df, model_info)
    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
    render_market_trends(clean_df)
    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
    render_faq()
    st.markdown('<hr class="divider-grad">', unsafe_allow_html=True)
    render_about_footer(model_info)


if __name__ == "__main__":
    main()