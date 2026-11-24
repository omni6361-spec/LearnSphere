
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import html
import json

from pathlib import Path
from local_xai import lookup_local_explanations
from personalization import student_reason, recommend_strategy, FEATURE_GROUPS
from release_data import ensure_release_data, ensure_release_history, ReleaseDataError


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LearnSphere | Personalized Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
BASE = APP_DIR / "data"
DATA_PATH = BASE / "history.csv"
TEST_DIR = BASE
IMAGE_DIR = APP_DIR / "assets"
SNAPSHOT = pd.Timestamp("2013-08-01")

COURSE_IMAGES = {
    "HarvardX/CS50x/2012": "cs50x.jpg",
    "HarvardX/ER22x/2013_Spring": "er22x.jpg",
    "MITx/8.MReV/2013_Summer": "8mrev.jpg"
}


# =====================================================
# APPLICATION STYLE
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    background: #101f3d;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h2 {
    color: white;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] * {
    color: #172b4a !important;
}

h1, h2, h3 {
    color: #183153;
}

.hero {
    background: linear-gradient(
        120deg,
        #132e59,
        #2766ba
    );

    color: white;
    border-radius: 24px;
    padding: 43px;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 37px;
    margin-bottom: 10px;
}

.hero p {
    color: #dce9ff;
    font-size: 15px;
    max-width: 620px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #192e51;
    margin-top: 28px;
    margin-bottom: 8px;
}

.section-description {
    color: #7888a0;
    margin-bottom: 18px;
    font-size: 14px;
}

.history-card {
    background: white;
    padding: 19px 22px;
    border-radius: 17px;
    border: 1px solid #e5ebf3;
    margin-bottom: 13px;
}

.history-title {
    color: #18345c;
    font-size: 16px;
    font-weight: 700;
}

.history-date {
    color: #8290a5;
    margin-top: 7px;
    font-size: 12px;
}

.explanation-panel {
    background: white;
    border: 1px solid #e6edf5;
    border-radius: 19px;
    padding: 25px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.explanation-title {
    font-size: 20px;
    font-weight: 800;
    color: #153a6b;
    margin-bottom: 12px;
}

.explanation-text {
    line-height: 1.9;
    color: #4e607c;
    font-size: 15px;
}

.strategy-card {
    background: white;
    border: 1px solid #e6edf5;
    border-radius: 18px;
    padding: 23px;
    margin-bottom: 16px;
}

.strategy-name {
    color: #183b72;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 9px;
}

.strategy-dimension {
    color: #168e99;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 9px;
}

.strategy-text {
    color: #526581;
    line-height: 1.8;
    font-size: 14px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #e7edf5;
    border-radius: 17px;
}
.student-cover {
    width: 100%; height: 156px; object-fit: cover;
    border-radius: 14px; display: block; margin-bottom: 12px;
}
.student-course-name {font-size: 19px; font-weight: 800; color: #183153; margin-top: 5px;}
.course-provider-label {font-size: 12px; font-weight: 700; letter-spacing: .065em; color: #268797;}
.course-rank {background:#e6f3f6; color:#176878; border-radius:30px; display:inline-block; padding:4px 10px; font-weight:750; font-size:12px;}
.recommendation-reason {border-left:4px solid #259a8d; background:#edf8f5; border-radius:8px; padding:14px 15px; line-height:1.65; margin:10px 0; color:#1d4050; font-size:14px;}
.strategy-result {background:linear-gradient(120deg,#143360,#2d7a8b); border-radius:18px; color:white; padding:25px; margin-bottom:16px;}
.strategy-result h3 {color:white; margin: 8px 0;}
.strategy-result p {color:#e0f3f4;}
.audience-note {color:#64748b; font-size:13px;}

/* Keep text readable on Streamlit Cloud and mobile browsers. */
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > h1,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > h2,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > h3,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > h4,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > p,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > ul,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] > ol {
    color: #183153 !important;
}
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"],
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: #526581 !important;
}
.stApp [data-testid="stMain"] [data-testid="stMetricLabel"],
.stApp [data-testid="stMain"] [data-testid="stMetricLabel"] p,
.stApp [data-testid="stMain"] [data-testid="stMetricValue"] {
    color: #183153 !important;
}
.stApp [data-testid="stMain"] [data-testid="stExpander"] summary,
.stApp [data-testid="stMain"] [data-testid="stExpander"] summary p {
    color: #183153 !important;
}


/* Mobile theme-contrast repair:
   Some visitors use Streamlit's dark client theme while our course
   surfaces remain light. Explicitly color native Streamlit content
   instead of allowing white text to inherit from the client theme. */
.stApp [data-testid="stMain"] {
    color: #183153 !important;
    background-color: #f5f7fb !important;
    color-scheme: light;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
    color: #183153 !important;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
:is(h1, h2, h3, h4, h5, h6, p, li, strong, em) {
    color: #183153 !important;
}
.stApp [data-testid="stMain"]
:is([data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *) {
    color: #314b68 !important;
}
.stApp [data-testid="stMain"]
[data-testid="stExpander"] {
    background: #ffffff !important;
    color: #183153 !important;
}
.stApp [data-testid="stMain"]
[data-testid="stExpander"] :is(summary, summary *) {
    background: #ffffff !important;
    color: #183153 !important;
}
.stApp [data-testid="stMain"]
[data-testid="stExpanderDetails"] {
    color: #183153 !important;
    background: #ffffff !important;
}
/* Restore intentional white-on-blue hero and strategy elements. */
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
.hero :is(h1, h2, h3, h4, strong) {
    color: #ffffff !important;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
.hero p {
    color: #dce9ff !important;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
.strategy-result :is(h2, h3, h4, strong) {
    color: #ffffff !important;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
.strategy-result p {
    color: #e0f3f4 !important;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
.recommendation-reason :is(p, strong, span) {
    color: #1d4050 !important;
}
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"]
.course-rank {
    color: #176878 !important;
}

@media (max-width: 640px) {
    .hero { padding: 26px 22px; border-radius: 18px; }
    .hero h1 { font-size: 30px; line-height: 1.24; }
    .block-container { padding-top: 1rem; }
    .student-cover { height: auto; aspect-ratio: 16 / 9; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# DATA
# =====================================================

@st.cache_data(show_spinner=False)
def load_predictions():
    path = BASE / "recommendations.csv"
    df = pd.read_csv(path, dtype={"userid_DI": str, "course_id": str})
    required = {"userid_DI", "course_id", "rank", "recommendation_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Recommendations missing columns: {sorted(missing)}")
    if df.duplicated(["userid_DI", "course_id"]).any():
        raise ValueError("Repeated learner–course pairs in recommendations.csv")
    df["rank"] = df["rank"].astype(int)
    df["recommendation_score"] = pd.to_numeric(df["recommendation_score"], errors="raise")
    return df


@st.cache_data(show_spinner=False)
def load_history(source_version=None):
    # Include the source file version in the Streamlit cache key.
    # Otherwise a previous empty result can remain cached after the
    # on-demand GitHub Release history import completes.
    # A reviewed, pre-cutoff history_display.csv has only learner/course keys.
    # The complete history.csv is used locally when dates are available.
    display_path = BASE / "history_display.csv"
    source = display_path if display_path.exists() else DATA_PATH
    if not source.exists():
        return pd.DataFrame(columns=["userid_DI", "course_id", "start_time_DI", "precutoff_display"])
    df = pd.read_csv(source, dtype={"userid_DI": str, "course_id": str})
    if not {"userid_DI", "course_id"}.issubset(df.columns):
        raise ValueError("Learner history requires userid_DI and course_id")
    df["precutoff_display"] = source.name == "history_display.csv"
    if "start_time_DI" not in df.columns:
        df["start_time_DI"] = pd.NaT
    else:
        df["start_time_DI"] = pd.to_datetime(df["start_time_DI"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_explanations():
    path = BASE / "explanations.csv"
    if not path.exists():
        return pd.DataFrame(columns=["userid_DI", "course_id", "Feature", "SHAP_Value", "baseline"])
    return pd.read_csv(path, dtype={"userid_DI": str, "course_id": str})


@st.cache_data(show_spinner=False)
def optional_csv(filename):
    path = BASE / filename
    return pd.read_csv(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_public_analytics():
    path = BASE / "public_analytics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False, max_entries=180)
def cached_pair(learner, course):
    return lookup_local_explanations(BASE, learner, course)



# =====================================================
# OBSERVED LEARNER BEHAVIOR (NOT MODEL INPUTS)
# =====================================================
# These variables are read from the original edX records, rather
# than inferred from SHAP. When last_event_DI is later than the
# evaluation cutoff, whole-record engagement metrics may include
# future activity and are deliberately withheld.

BEHAVIOR_COLUMNS = [
    "userid_DI", "course_id", "start_time_DI", "last_event_DI",
    "viewed", "explored", "nevents", "ndays_act",
    "nchapters", "nforum_posts",
]
HORIZON_END = pd.Timestamp("2013-08-29")  # Aug 1–28 inclusive


@st.cache_data(show_spinner=False)
def load_observed_behavior():
    path = BASE / "behavior.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"userid_DI": str, "course_id": str})


def get_observed_behavior(user_id, course_id):
    records = load_observed_behavior()
    if records.empty:
        return None, "No observed behavioral records are included in this demo."
    rows = records.loc[
        (records["userid_DI"] == str(user_id))
        & (records["course_id"] == str(course_id))
    ]
    if rows.empty:
        return None, "No recorded course start for this learner–course pair."
    return rows.iloc[-1], None


def observed_behavior_status(record):
    """Accept only event records whose last activity is temporally auditable."""
    if record is None:
        return False, "No recorded course entry"
    if "start_time_DI" not in record.index:
        return False, "Start date unavailable"
    started = pd.to_datetime(record.get("start_time_DI"), errors="coerce")
    if pd.isna(started):
        return False, "Start date missing"
    if not (SNAPSHOT <= started < HORIZON_END):
        return False, "No course start recorded within Aug 1–28"
    if "last_event_DI" not in record.index:
        return False, "Last-event date unavailable; activity window cannot be verified"
    ended = pd.to_datetime(record.get("last_event_DI"), errors="coerce")
    if pd.isna(ended):
        return False, "Last-event date missing; activity window cannot be verified"
    if ended < started:
        return False, "Invalid event order; activity measures withheld"
    if ended >= HORIZON_END:
        return False, "Later activity after Aug 28; full-record measures withheld"
    return True, "Recorded events fall within Aug 1–28"


def show_behavioral_values(record, fields):
    if record is None:
        st.caption("No course-level outcome record available.")
        return
    entries = []
    for field, label in fields:
        if field not in record.index:
            continue
        val = record[field]
        if pd.isna(val):
            entries.append((label, "Not recorded"))
            continue
        if field in {"viewed", "explored"}:
            number = pd.to_numeric(val, errors="coerce")
            output = "Yes" if number == 1 else "No" if number == 0 else "Not recorded"
        else:
            number = pd.to_numeric(val, errors="coerce")
            output = f"{number:g}" if pd.notna(number) else "Not recorded"
        entries.append((label, output))
    if entries:
        cols = st.columns(min(len(entries), 3))
        for idx, (label, value) in enumerate(entries):
            cols[idx % len(cols)].metric(label, value)
    else:
        st.caption("The original CSV does not contain these indicators.")


# =====================================================
# FEATURE DESCRIPTIONS
# =====================================================

FEATURE_LABELS = {

    "days_since_last_start":
        "Timing of the last course start",

    "user_starts_30d":
        "Recent learning activity",

    "history_count":
        "Previous course selections",

    "history_span_days":
        "Learning history duration",

    "provider_history_count":
        "Previous courses from the provider",

    "provider_affinity":
        "Historical provider affinity",

    "course_code_history_count":
        "Previous selections of related course codes",

    "course_code_affinity":
        "Historical course-code affinity",

    "provider_diversity":
        "Provider diversity",

    "course_code_diversity":
        "Course selection diversity",

    "course_popularity_7d":
        "Recent course popularity",

    "course_popularity_prev7d":
        "Previous-week popularity",

    "course_popularity_30d":
        "Monthly course popularity",

    "course_age_days":
        "Course age",

    "course_trend_7d":
        "Recent demand trend",

    "course_share_30d":
        "Course share of recent demand",

    "provider_starts_30d":
        "Recent provider demand"
}


def friendly_feature(name):

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


BEHAVIOR_GROUPS = {

    "Selection History": [
        "history_count",
        "history_span_days",
        "provider_history_count",
        "course_code_history_count",
        "provider_affinity",
        "course_code_affinity"
    ],

    "Selection Timing": [
        "days_since_last_start",
        "user_starts_30d"
    ],

    "Selection Diversity": [
        "provider_diversity",
        "course_code_diversity"
    ]
}


STRATEGIES = {

    "Selection History": {
        "name": "History-Based Personalization",
        "description":
            "Personalize course discovery using learners' "
            "previous course-selection patterns."
    },

    "Selection Timing": {
        "name": "Timing-Aware Recommendations",
        "description":
            "Investigate personalized recommendations "
            "that consider the timing and recency "
            "of learners' course-selection activity."
    },

    "Selection Diversity": {
        "name": "Diversified Course Discovery",
        "description":
            "Investigate diversified course recommendations "
            "based on learners' historical selection diversity."
    }
}


# =====================================================
# SHAP
# =====================================================

def explain_row(row):
    # Every value was computed and validated offline with the original
    # fitted model. Streamlit Cloud never loads the training model.
    all_explanations = load_explanations()
    selected = all_explanations.loc[
        (all_explanations["userid_DI"] == str(row["userid_DI"]))
        & (all_explanations["course_id"] == str(row["course_id"]))
    ].copy()
    if selected.empty:
        offline = cached_pair(row["userid_DI"], row["course_id"])
        if offline is not None:
            return clean_local_shap(offline["shap"]), float(row["recommendation_score"]), float(offline["baseline"])
        return None, float(row["recommendation_score"]), None
    selected["Abs_SHAP"] = selected["SHAP_Value"].abs()
    score = float(row["recommendation_score"])
    baseline = float(selected["baseline"].iloc[0])
    if not np.isclose(
        baseline + selected["SHAP_Value"].sum(), score, atol=1e-5
    ):
        raise ValueError("Stored SHAP contributions do not match recommendation score.")
    return (
        selected[["Feature", "SHAP_Value", "Abs_SHAP"]]
        .sort_values("Abs_SHAP", ascending=False).reset_index(drop=True),
        score,
        baseline,
    )


# =====================================================
# COURSE IMAGES
# =====================================================

def course_code(course_id):

    parts = str(course_id).split("/")

    return parts[1] if len(parts) > 1 else str(course_id)


def course_provider(course_id):

    return str(course_id).split("/")[0]


def course_cover(course_id, index=0):

    filename = COURSE_IMAGES.get(course_id)

    if filename:

        path = IMAGE_DIR / filename

        if path.exists():

            mime = (
                "image/png"
                if path.suffix.lower() == ".png"
                else "image/jpeg"
            )

            encoded = base64.b64encode(
                path.read_bytes()
            ).decode("ascii")

            return f"data:{mime};base64,{encoded}"

    palettes = [
        ("#122f5c", "#4074c4"),
        ("#18394c", "#1c918f"),
        ("#392b66", "#7f65c1"),
        ("#453054", "#bd678c")
    ]

    first, second = palettes[index % len(palettes)]

    provider = html.escape(course_provider(course_id))
    code = html.escape(course_code(course_id))

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg"
     width="640" height="390" viewBox="0 0 640 390">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="{first}"/>
<stop offset="100%" stop-color="{second}"/>
</linearGradient>
</defs>
<rect width="640" height="390" fill="url(#bg)"/>
<circle cx="540" cy="85" r="140" fill="white" opacity=".06"/>
<circle cx="540" cy="85" r="95" fill="none"
        stroke="white" stroke-width="2" opacity=".3"/>
<path d="M40 310 L175 170 L280 250 L440 90"
      fill="none" stroke="white" stroke-width="4" opacity=".18"/>
<text x="45" y="75" font-family="Arial"
      font-size="24" fill="#dbe9ff">{provider}</text>
<text x="45" y="210" font-family="Arial"
      font-size="65" font-weight="bold"
      fill="white">{code}</text>
<rect x="45" y="245" width="72" height="5"
      rx="2" fill="#81e4da"/>
<text x="45" y="305" font-family="Arial"
      font-size="20" fill="#e4efff">
EXPLORE YOUR NEXT COURSE
</text>
</svg>
"""

    encoded = base64.b64encode(
        svg.encode("utf-8")
    ).decode("ascii")

    return f"data:image/svg+xml;base64,{encoded}"


# =====================================================
# COURSE CAROUSEL
# =====================================================

CAROUSEL_CSS = """
<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: transparent;
    font-family: Arial, sans-serif;
}

.carousel {
    display: flex;
    gap: 20px;
    overflow-x: auto;
    padding: 10px 8px 24px;
    scroll-snap-type: x mandatory;
}

.carousel::-webkit-scrollbar {
    height: 8px;
}

.carousel::-webkit-scrollbar-thumb {
    background: #cad8ed;
    border-radius: 12px;
}

.course-card {
    flex: 0 0 305px;
    scroll-snap-align: start;
    background: #fff;
    border: 1px solid #e3eaf4;
    border-radius: 19px;
    overflow: hidden;
    box-shadow: 0 6px 22px rgba(16,42,86,.09);
    transition: transform .2s, box-shadow .2s;
    color: #172b4c;
    outline-offset: 3px;
}

.course-card:hover,
.course-card:focus {
    transform: translateY(-4px);
    box-shadow: 0 13px 28px rgba(16,42,86,.17);
}

.photo {
    height: 188px;
    position: relative;
    overflow: hidden;
}

.photo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.rank {
    position: absolute;
    top: 13px;
    left: 13px;
    background: white;
    color: #173c70;
    padding: 7px 12px;
    border-radius: 25px;
    font-size: 11px;
    font-weight: bold;
    z-index: 1;
}

.hover-popup {
    position: absolute;
    inset: 0;
    background: rgba(12,29,57,.97);
    color: white;
    padding: 18px;
    opacity: 0;
    transform: translateY(12px);
    transition: .22s ease;
    overflow-y: auto;
    z-index: 2;
}

.course-card:hover .hover-popup,
.course-card:focus .hover-popup {
    opacity: 1;
    transform: translateY(0);
}

.hover-popup h3 {
    font-size: 17px;
    color: white;
    margin: 0 0 12px;
}

.hover-popup p {
    color: #e4efff;
    font-size: 12px;
    line-height: 1.5;
    margin: 8px 0;
}

.tag {
    color: #7ae5d0;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: bold;
    margin-top: 10px;
}

.content {
    padding: 19px;
}

.provider {
    color: #2e70bb;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 9px;
}

.course-title {
    color: #1b3052;
    font-size: 17px;
    font-weight: bold;
    line-height: 1.5;
    min-height: 50px;
    overflow-wrap: anywhere;
}

.score {
    border-top: 1px solid #ebf0f6;
    padding-top: 15px;
    margin-top: 15px;
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-size: 12px;
    color: #74859c;
}

.score strong {
    color: #173e78;
    font-size: 18px;
}

.hint {
    color: #929eae;
    font-size: 11px;
    margin-top: 12px;
}
</style>
"""


def render_course_carousel(top3):

    cards = []

    for index, (_, row) in enumerate(top3.iterrows()):

        explanation, score, _ = explain_row(row)

        if explanation is None:
            pos = "No validated per-course SHAP explanation is available for this case."
            neg = "The displayed score is a previously computed ranking score."
        else:
            positive = explanation[explanation["SHAP_Value"] > 0].head(2)
            negative = explanation[explanation["SHAP_Value"] < 0].head(1)
            pos = "; ".join(friendly_feature(f) for f in positive["Feature"])
            neg = "; ".join(friendly_feature(f) for f in negative["Feature"])
        cover = course_cover(
            row["course_id"],
            index
        )

        provider = html.escape(
            course_provider(row["course_id"])
        )

        title = html.escape(
            course_code(row["course_id"])
        )

        pos = html.escape(pos)
        neg = html.escape(neg)

        card = f"""
<article class="course-card" tabindex="0">
<div class="photo">
<img src="{cover}" alt="{title}">
<div class="rank">TOP {int(row['rank'])}</div>
<div class="hover-popup">
<h3>Why this course?</h3>
<div class="tag">POSITIVE CONTRIBUTIONS</div>
<p>{pos}</p>
<div class="tag">NEGATIVE CONTRIBUTION</div>
<p>{neg}</p>
<div class="tag">RECOMMENDATION SCORE</div>
<p><strong>{score:.4f}</strong></p>
</div>
</div>
<div class="content">
<div class="provider">{provider}</div>
<div class="course-title">{title}</div>
<div class="score">
<span>Recommendation score</span>
<strong>{score:.4f}</strong>
</div>
<div class="hint">Hover to discover why</div>
</div>
</article>
"""

        cards.append(card.strip())

    document = (
        '<!DOCTYPE html>'
        '<html lang="en">'
        '<head><meta charset="UTF-8">'
        + CAROUSEL_CSS
        + '</head><body>'
        '<div class="carousel">'
        + "".join(cards)
        + '</div></body></html>'
    )

    components.html(
        document,
        height=405,
        scrolling=False
    )


# =====================================================
# TEXT EXPLANATION
# =====================================================

def narrative_explanation(explanation, previous):

    paragraphs = []

    if not previous.empty:

        last = previous.sort_values(
            "start_time_DI"
        ).iloc[-1]

        paragraphs.append(
            "Your most recently started course was "
            f"{last['course_id']} on "
            f"{last['start_time_DI']:%B %d, %Y}."
        )

    positive = explanation[
        explanation["SHAP_Value"] > 0
    ].head(3)

    negative = explanation[
        explanation["SHAP_Value"] < 0
    ].head(2)

    if not positive.empty:

        factors = ", ".join(
            friendly_feature(f)
            for f in positive["Feature"]
        )

        paragraphs.append(
            "The recommendation score was mainly increased "
            "by " + factors + "."
        )

    if not negative.empty:

        factors = ", ".join(
            friendly_feature(f)
            for f in negative["Feature"]
        )

        paragraphs.append(
            "Factors that reduced the recommendation score "
            "included " + factors + "."
        )

    timing = explanation[
        explanation["Feature"] == "days_since_last_start"
    ]

    if not timing.empty and not previous.empty:

        value = float(
            timing.iloc[0]["SHAP_Value"]
        )

        if value > 0:

            paragraphs.append(
                "The timing of your most recent course start "
                "made a positive contribution to this "
                "recommendation."
            )

        elif value < 0:

            paragraphs.append(
                "The timing of your most recent course start "
                "made a negative contribution to this "
                "recommendation."
            )

    return " ".join(paragraphs)


# =====================================================
# SHAP CHART
# =====================================================

def shap_chart(explanation):

    # Do not show SHAP's synthetic additive tail as an individual feature.
    explanation = explanation.loc[
        explanation["Feature"] != "Other features (aggregated)"
    ]
    data = explanation.head(12).sort_values(
        "SHAP_Value"
    )

    colors = [
        "#19a68d" if x > 0 else "#ed6e78"
        for x in data["SHAP_Value"]
    ]

    fig = go.Figure(
        go.Bar(
            x=data["SHAP_Value"],
            y=[
                friendly_feature(f)
                for f in data["Feature"]
            ],
            orientation="h",
            marker_color=colors
        )
    )

    fig.update_layout(
        height=450,
        margin=dict(l=15, r=20, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="SHAP Contribution",
        yaxis_title=""
    )

    fig.add_vline(
        x=0,
        line_color="#9daec2"
    )

    return fig


# =====================================================
# BEHAVIOR ANALYSIS
# =====================================================

def analyze_behavior(explanation):

    rows = []

    for dimension, features in BEHAVIOR_GROUPS.items():

        subset = explanation[
            explanation["Feature"].isin(features)
        ]

        rows.append({
            "Dimension": dimension,
            "Importance": float(
                subset["Abs_SHAP"].sum()
            ),
            "Net_Contribution": float(
                subset["SHAP_Value"].sum()
            )
        })

    return pd.DataFrame(rows).sort_values(
        "Importance",
        ascending=False
    )




# =====================================================
# COMPLETE TWO-METHOD LOCAL COMPARISON
# =====================================================

def clean_local_shap(data):
    # The SHAP tail is an additive aggregate, not a standalone feature.
    return data.loc[data["Feature"] != "Other features (aggregated)"].copy().sort_values(
        "SHAP_Value", key=lambda x: x.abs(), ascending=False
    ).reset_index(drop=True)


def comparison_data(pair, top_k=5):
    shap_items = clean_local_shap(pair["shap"])
    lime_items = pair["lime"].copy().sort_values(
        "LIME_Weight", key=lambda x: x.abs(), ascending=False
    ).reset_index(drop=True)
    shap_top = shap_items.head(top_k).reset_index(drop=True)
    lime_top = lime_items.head(top_k).reset_index(drop=True)
    first = pd.DataFrame({
        "Rank": range(1, top_k + 1),
        "SHAP Feature": shap_top["Feature"].reindex(range(top_k)),
        "SHAP Value": shap_top["SHAP_Value"].reindex(range(top_k)),
        "LIME Feature": lime_top["Feature"].reindex(range(top_k)),
        "LIME Weight": lime_top["LIME_Weight"].reindex(range(top_k)),
    })
    sdict = dict(zip(shap_top.Feature, shap_top.SHAP_Value))
    ldict = dict(zip(lime_top.Feature, lime_top.LIME_Weight))
    intersection = set(sdict) & set(ldict)
    union = set(sdict) | set(ldict)
    matched_rows = []
    for feat in sorted(union):
        matched_rows.append({
            "Feature": feat,
            "SHAP Value (Top N)": sdict.get(feat, np.nan),
            "LIME Weight (Top N)": ldict.get(feat, np.nan),
            "In Both Top N": feat in intersection,
        })
    same_sign = [np.sign(sdict[f]) == np.sign(ldict[f]) for f in intersection]
    return first, pd.DataFrame(matched_rows), len(intersection), (
        len(intersection) / len(union) if union else np.nan
    ), (float(np.mean(same_sign)) if same_sign else np.nan)


def signed_bar(data, key, title, limit=10):
    subset = data.sort_values(key, key=lambda x: x.abs(), ascending=False).head(limit)
    subset = subset.sort_values(key)
    fig = go.Figure(go.Bar(
        x=subset[key],
        y=subset["Feature"].map(friendly_feature),
        orientation="h",
        marker_color=["#1a9b89" if x >= 0 else "#df6b73" for x in subset[key]],
        hovertemplate="%{y}<br>Weight: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(335, len(subset)*33+85),
        margin=dict(l=15,r=10,t=48,b=24),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="SHAP contribution to score" if key == "SHAP_Value" else "LIME local surrogate coefficient",
        yaxis_title="", font=dict(size=11),
    )
    fig.add_vline(x=0, line_color="#b4c4d4", line_width=1)
    return fig


def render_pair_comparison(learner, course, expected_score, key_prefix="comparison"):
    pair = cached_pair(learner, course)
    if pair is None:
        st.warning("This learner/course has no precomputed SHAP and LIME explanation in the published data.")
        return
    if not np.isclose(float(pair["score"]), float(expected_score), rtol=0, atol=1e-5):
        st.error("The stored explanation's model score does not match the saved recommendation. Please check the data export.")
        return
    shap_full = pair["shap"].copy()
    shap = clean_local_shap(shap_full)
    lime = pair["lime"].copy()
    shap_total = float(pair["baseline"]) + float(shap_full["SHAP_Value"].sum())
    if not np.isclose(shap_total, pair["score"], atol=1e-4, rtol=0):
        st.error("The stored SHAP values do not reconstruct the prediction score.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Model score (not calibrated)", f"{float(pair['score']):.4f}")
    c2.metric("SHAP reference score", f"{float(pair['baseline']):.4f}")
    c3.metric("LIME local fit R²", f"{float(pair['lime_fidelity']):.4f}")
    fidelity = float(pair["lime_fidelity"])
    if fidelity < 0.50:
        st.warning("Low LIME fidelity (R² < 0.50): its feature weights are a weak local approximation of the Extra Trees model for this case. Interpret with caution.")
    elif fidelity < 0.75:
        st.info("Moderate LIME fidelity: coefficient interpretation needs caution. The threshold is a display convention, not a validated quality standard.")
    st.caption(f"LIME uses {int(pair['samples']):,} local perturbations. SHAP contributions and LIME coefficients have different definitions and must not be compared directly by magnitude. Differences are expected and are not errors by themselves.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(signed_bar(shap, "SHAP_Value", "SHAP • Model contributions"), use_container_width=True)
    with right:
        st.plotly_chart(signed_bar(lime, "LIME_Weight", "LIME • Local surrogate"), use_container_width=True)

    top_n = st.radio("Compare top features", [5, 10], horizontal=True, key=f"{key_prefix}_topn")
    side, aligned, shared, jaccard, signed = comparison_data(pair, top_k=top_n)
    st.markdown(f"#### Top {top_n} feature ranking — two methods")
    st.dataframe(side, hide_index=True, use_container_width=True)
    a,b,c = st.columns(3)
    a.metric("Common features", f"{shared} / {top_n}")
    b.metric("Jaccard overlap", f"{jaccard:.3f}")
    c.metric("Shared sign agreement", f"{signed:.1%}" if np.isfinite(signed) else "N/A")
    st.caption("These are descriptive cross-method differences, not measures of truth. LIME's coefficient sign describes the fitted local surrogate; the SHAP sign describes contribution relative to the SHAP reference.")
    with st.expander("Same features aligned by name (missing means not in Top N, not zero)"):
        st.dataframe(aligned, hide_index=True, use_container_width=True)
    with st.expander("Full saved explanation data"):
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("**SHAP saved entries**")
            st.dataframe(shap_full, hide_index=True, use_container_width=True)
        with rcol:
            st.markdown("**LIME saved entries**")
            st.dataframe(lime, hide_index=True, use_container_width=True)
    downloadable = side.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download this comparison as CSV", downloadable,
                       file_name="LearnSphere_local_SHAP_LIME.csv", mime="text/csv",
                       key=f"{key_prefix}_download")


def render_cohort_comparison():
    summary = load_public_analytics()
    if not summary:
        st.info("The public cohort comparison has not been generated. Run 'python build_public_analytics.py --expected 177247' locally after copying the 12 complete SQLite files into data/local_xai/. The learner-level recommendations remain accessible if they are present.")
        return
    n = summary.get("total_recommendations", 0)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Explained recommendations", f"{n:,}")
    m2.metric("Mean LIME R²", f"{summary['mean_lime_fidelity']:.3f}")
    m3.metric("Mean Top-5 Jaccard", f"{summary['mean_jaccard']:.3f}")
    m4.metric("Mean shared Top-5", f"{summary['mean_top5_overlap']:.2f} / 5")
    st.caption("Every recommendation was explained offline with the original model. These numbers describe the complete validated export, not a randomly sampled subset. The model score is not a calibrated enrollment/purchase probability.")
    dist = pd.DataFrame(summary.get("overlap_distribution", []))
    if not dist.empty:
        fig = go.Figure(go.Bar(x=dist["Shared_Features"].astype(str), y=dist["Percentage"], marker_color="#2962aa"))
        fig.update_layout(height=310, margin=dict(l=10,r=10,t=20,b=35),
                          xaxis_title="Shared features (out of 5)",yaxis_title="% of recommendations")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dist, hide_index=True, use_container_width=True)
    groups = pd.DataFrame(summary.get("fidelity_groups", []))
    if not groups.empty:
        st.markdown("#### LIME local fidelity groups")
        st.dataframe(groups, hide_index=True, use_container_width=True)
    courses = pd.DataFrame(summary.get("courses", []))
    if not courses.empty:
        st.markdown("#### Comparison by course")
        course_filter = st.multiselect("Filter courses", sorted(courses["Course"].unique()), key="cohort_courses")
        visible = courses[courses["Course"].isin(course_filter)] if course_filter else courses
        st.dataframe(visible, hide_index=True, use_container_width=True)
        st.download_button("Download course comparison CSV", visible.to_csv(index=False).encode("utf-8-sig"),
                           file_name="LearnSphere_by_course.csv", mime="text/csv")
    a,b = st.columns(2)
    for column, metric, label in [(a, "most_frequent_shap_top1", "Most frequent #1 SHAP feature"),
                                  (b, "most_frequent_lime_top1", "Most frequent #1 LIME feature")]:
        with column:
            st.markdown(f"#### {label}")
            top = pd.DataFrame(summary.get(metric, []))
            if not top.empty:
                st.dataframe(top, hide_index=True, use_container_width=True)
    st.caption("The SHAP and LIME rankings use the largest absolute values from each saved explanation; SHAP's 'Other features (aggregated)' is excluded. LIME R² is an in-sample surrogate statistic, not an independent explanation-ground-truth score.")

# =====================================================
# STUDENT / PLATFORM VIEW COMPONENTS
# =====================================================

def render_student_card(row, position):
    """Only a plain-language explanation is exposed to the student."""
    course = str(row["course_id"])
    learner = str(row["userid_DI"])
    cover = course_cover(course, position)
    pair = cached_pair(learner, course)
    reason = student_reason(pair, float(row["recommendation_score"]), friendly_feature)
    with st.container(border=True):
        st.markdown(
            '<img class="student-cover" src="' + html.escape(cover, quote=True) + '" alt="Course illustration" />',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="course-provider-label">{html.escape(course_provider(course))}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="student-course-name">{html.escape(course_code(course))}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<span class="course-rank">Recommended #{int(row["rank"])}</span>',
                    unsafe_allow_html=True)
        st.caption(course)
        st.markdown("**Why this course?**")
        st.markdown('<div class="recommendation-reason">' + html.escape(reason["main"]) + '</div>',
                    unsafe_allow_html=True)
        st.caption(reason["secondary"])


def render_student_history(previous, has_history):
    st.markdown('### Your previously selected courses')
    if not has_history:
        st.info("Previous course history has not been included in this site. The saved recommendations are still available.")
        return
    if previous.empty:
        st.info("No earlier course starts were found for this learner ID before August 2013.")
        return
    st.caption(f"{len(previous)} distinct historical course selections, before the August 2013 prediction window.")
    def one_history(item):
        with st.container(border=True):
            provider = course_provider(item["course_id"])
            st.markdown(f"**{course_code(item['course_id'])}** · {provider}")
            when = item.get("start_time_DI")
            if pd.notna(when):
                st.caption("Started " + when.strftime("%B %d, %Y"))
            else:
                st.caption("Previous course selection")
    for _, item in previous.head(6).iterrows():
        one_history(item)
    if len(previous) > 6:
        with st.expander(f"View {len(previous)-6} more previous courses"):
            for _, item in previous.iloc[6:].iterrows():
                one_history(item)


def render_marketing_strategy(selected_user, top3):
    prepared = []
    for _, row in top3.iterrows():
        prepared.append((cached_pair(selected_user, str(row['course_id'])),
                         float(row['recommendation_score'])))
    proposal = recommend_strategy(prepared, friendly_feature)
    if not proposal["available"]:
        st.info(proposal["summary"])
        return
    st.markdown('<div class="strategy-result"><div>STRATEGY SUGGESTION FOR THIS LEARNER</div>'
                '<h3>' + html.escape(proposal['title']) + '</h3>'
                '<p>' + html.escape(proposal['summary']) + '</p></div>', unsafe_allow_html=True)
    left,right = st.columns([2,1])
    with left:
        st.markdown("#### Why was this suggested?")
        st.write(
            f"The **{proposal['group']}** feature group had the largest combined absolute "
            f"SHAP contribution among the three saved recommended courses "
            f"({proposal['valid_recommendations']} validated explanations used). "
            "This is a transparent rule, not a model predicting marketing effectiveness."
        )
        if proposal["top_factors"]:
            st.markdown("**Supporting historical factors:** " + ", ".join(proposal["top_factors"]))
        st.markdown("**Related domain:** " + proposal["dimension"])
        st.markdown("**Suggested next action:** " + proposal["action"])
    with right:
        table = pd.DataFrame([
            {"Feature group": k, "Total |SHAP|": v}
            for k, v in proposal["group_scores"].items()
        ])
        st.dataframe(table, hide_index=True, use_container_width=True)
    st.warning(
        "Research proposal only: SHAP explains the historical course-start score. "
        "It does not establish preferences, engagement outcomes, purchase intent, "
        "or that this marketing strategy will work. Test strategies prospectively."
    )


def render_behavior_dimensions(selected_user, user_history, previous, selected_course):
    """Show domain indicators where observable; link historical inputs only to participation."""
    st.caption("These are operational research domains, not a validated unified Consumer Behavior scale.")
    observed, issue = get_observed_behavior(selected_user, selected_course)
    valid, status = observed_behavior_status(observed)
    pair = cached_pair(selected_user, selected_course)
    historical_features = clean_local_shap(pair['shap']) if pair is not None else None
    has_history = (BASE / "history_display.csv").exists() or DATA_PATH.exists()

    with st.container(border=True):
        st.markdown("#### 1. Course Participation")
        st.caption("Course selections and historical starts before the prediction snapshot (Ho et al., 2015).")
        c1,c2 = st.columns(2)
        c1.metric("Earlier courses selected", len(previous) if has_history else "Not published")
        if has_history and not user_history["precutoff_display"].any():
            starts = user_history.loc[(user_history.start_time_DI >= SNAPSHOT) &
                                      (user_history.start_time_DI < HORIZON_END), "course_id"].nunique()
            c2.metric("Starts in Aug 1–28", int(starts))
        else:
            c2.metric("Starts in Aug 1–28", "Not published")
        if historical_features is None:
            st.caption("Saved local historical-feature SHAP values are unavailable.")
        else:
            participation_features = set().union(*FEATURE_GROUPS.values())
            related = historical_features.loc[historical_features.Feature.isin(participation_features)]
            total = float(historical_features["Abs_SHAP"].sum())
            amount = float(related["Abs_SHAP"].sum())
            st.metric("Share of displayed |SHAP| from past selection features",
                      f"{amount / total:.1%}" if total else "N/A")
            with st.expander("Historical input features affecting this recommendation"):
                display = related[["Feature", "SHAP_Value"]].copy()
                display["Feature"] = display["Feature"].map(friendly_feature)
                st.dataframe(display, hide_index=True, use_container_width=True)
            st.caption("This attribution concerns the course-start model; it does not explain later learner behavior.")

    with st.container(border=True):
        st.markdown("#### 2. Behavioral Engagement")
        st.caption("Observed activity such as event counts, active days and forum posts, where within-window data are verifiable.")
        if valid:
            show_behavioral_values(observed, [("nevents", "Logged events"),
                                              ("ndays_act", "Active days"),
                                              ("nforum_posts", "Forum posts")])
        else:
            st.info("No safely time-verified engagement measures for this learner/course: " + status)
        st.caption("Engagement values are not inputs to this trained course-start model. Do not assign them SHAP values.")

    with st.container(border=True):
        st.markdown("#### 3. Course Exploration")
        st.caption("Observed chapter access and explorer status, if recorded within the evaluation period.")
        if valid:
            show_behavioral_values(observed, [("viewed", "Content accessed"),
                                              ("explored", "Explorer status"),
                                              ("nchapters", "Chapters accessed")])
        else:
            st.info("No safely time-verified exploration measures for this learner/course: " + status)
        st.caption("These are historical outcome indicators, not predictions of future browsing or satisfaction.")

    if issue:
        st.caption(issue)
    st.caption(
        "Relationship to the strategy: only prior-selection inputs can support the SHAP-based strategy heuristic. "
        "Engagement and exploration are displayed as separate observed outcomes when their timestamps can be audited; "
        "they were not used to train the current course-start recommender."
    )


# =====================================================
# LOAD DATA
# =====================================================

try:
    if not (BASE / "recommendations.csv").is_file():
        with st.spinner("Loading verified public LearnSphere recommendations and explanations..."):
            ensure_release_data(BASE)
    predictions = load_predictions()
    # Import only pseudonymous, pre-cutoff course selections for public display.
    # Fail gracefully if the history asset is missing or fails review.
    history_warning = None
    try:
        with st.spinner("Loading earlier course selections..."):
            ensure_release_history(BASE)
    except ReleaseDataError as history_error:
        history_warning = str(history_error)
    history_path = BASE / "history_display.csv"
    if not history_path.is_file():
        history_path = DATA_PATH
    history_version = (
        (history_path.name, history_path.stat().st_size,
         history_path.stat().st_mtime_ns)
        if history_path.is_file() else ("no-history", 0, 0)
    )
    history = load_history(history_version)
except ReleaseDataError as error:
    st.title("LearnSphere — Research Results")
    st.warning("Individual learner recommendations are not yet available on this public server. The complete data release is awaiting publication.")
    st.caption("The verified aggregate results below are based on the 177,247 historical recommendations; they contain no learner-level records.")
    public_metrics_path = BASE / "public_analytics.json"
    if public_metrics_path.is_file():
        stats = json.loads(public_metrics_path.read_text(encoding="utf-8"))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recommendations explained", f"{stats['total_recommendations']:,}")
        col2.metric("Learners in study", f"{stats['unique_learners']:,}")
        col3.metric("Mean LIME local R²", f"{stats['mean_lime_fidelity']:.3f}")
        col4.metric("Mean Top-5 Jaccard", f"{stats['mean_jaccard']:.3f}")
        st.subheader("Recommendation counts by course")
        course_summary = pd.DataFrame(stats.get("courses", []))
        if not course_summary.empty:
            st.bar_chart(course_summary.set_index("Course")["Recommendations"])
        st.subheader("SHAP vs LIME: shared Top-5 features")
        overlap = pd.DataFrame(stats.get("overlap_distribution", []))
        if not overlap.empty:
            st.bar_chart(overlap.set_index("Shared_Features")["Recommendations"])
        st.caption(stats.get("methodology", ""))
    else:
        st.info("The aggregate research results are also being prepared for public release.")
    with st.expander("When will learner recommendations become available?"):
        st.write("The site will download the approved recommendations.csv and 12 SHAP/LIME SQLite shards when the public GitHub Release tagged learnsphere-data-v1 contains all required files. The site does not retrain the model.")
        st.code(str(error), language=None)
    st.stop()
except Exception as error:
    st.error("Deployment data could not be loaded: " + str(error))
    st.stop()

if history_warning:
    st.sidebar.warning(
        "Course history could not be loaded automatically: " + history_warning
    )

users = sorted(predictions["userid_DI"].dropna().unique().tolist(),
               key=lambda v: (0, int(v.split()[-1])) if v.startswith('Learner ') and v.split()[-1].isdigit()
               else (1, str(v)))
if not users:
    st.error("No learner IDs are present in recommendations.csv")
    st.stop()

st.sidebar.markdown("## 🎓 LearnSphere")
st.sidebar.caption("Personalized learning | Research platform")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", ["Explore Courses", "Platform Intelligence", "Research Methods"],
                        label_visibility="collapsed")
st.sidebar.divider()
selected_user = st.sidebar.selectbox(
    "Select Learner ID", users,
    index=0,
    help="Searchable dropdown of the learner IDs already stored in recommendations.csv. No registration is needed.",
)
st.sidebar.caption("Historical dataset IDs • no login or account registration")

user_predictions = predictions.loc[predictions.userid_DI == selected_user].sort_values("rank")
top3 = user_predictions.head(3)
user_history = history.loc[history.userid_DI == selected_user].copy()
if not user_history.empty and user_history["precutoff_display"].all():
    previous = user_history.drop_duplicates("course_id")
else:
    previous = user_history.loc[user_history.start_time_DI < SNAPSHOT].drop_duplicates("course_id")
previous = previous.sort_values("start_time_DI", na_position="last", ascending=False)
if top3.empty:
    st.warning("No saved course recommendations for this learner ID.")
    st.stop()

if page == "Explore Courses":
    st.markdown('<div class="hero"><h1>Discover your next learning opportunity.</h1>'
                '<p>View your course history and three personalized course suggestions. '
                'Every explanation below is based on the original saved model decision.</p></div>',
                unsafe_allow_html=True)
    st.markdown(f"### Welcome, {selected_user}!")
    st.caption("This is a historical 2013 course recommendation study, not a live registration or course catalog.")
    left,right = st.columns(2)
    left.metric("Previous course selections", len(previous) if (DATA_PATH.exists() or (BASE/'history_display.csv').exists()) else "Not published")
    right.metric("Recommended courses", len(top3))
    with st.expander("My previous courses", expanded=True):
        render_student_history(previous, DATA_PATH.exists() or (BASE/'history_display.csv').exists())

    st.markdown("### Courses recommended for you")
    st.caption("Each card tells you why the original model ranked that course. Technical SHAP/LIME tables are available in the platform view, not here.")
    cols = st.columns(3, gap="medium")
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            render_student_card(row, i)
    st.caption("Suggestions are precomputed historical model rankings. They do not guarantee enrollment, suitability, or course availability.")

elif page == "Platform Intelligence":
    st.markdown('<div class="hero"><h1>Platform Intelligence</h1><p>'
                'Understand each learner’s recommendations, compare model explanations, '
                'examine observable behavior domains, and develop research-based marketing hypotheses.'
                '</p></div>', unsafe_allow_html=True)
    st.markdown(f"### Learner analysis · {selected_user}")
    st.caption("Use the Learner ID dropdown in the sidebar to review a different historical record. No login is required in this research viewer.")
    recommended = top3[["course_id", "rank", "recommendation_score"]].copy()
    recommended.columns = ["Course", "Rank", "Saved model score"]
    st.dataframe(recommended, hide_index=True, use_container_width=True)
    selected_course = st.selectbox("Choose one of this learner's recommended courses", top3.course_id.tolist(),
                                   format_func=lambda c: f"{course_code(c)} · {course_provider(c)}", key="platform_course")
    chosen_row = top3.loc[top3.course_id == selected_course].iloc[0]
    tabs = st.tabs(["Personalized Strategy", "Consumer Behavior", "SHAP vs LIME", "Cohort Analytics"])
    with tabs[0]:
        st.markdown("### A marketing suggestion linked to this learner's model factors")
        render_marketing_strategy(selected_user, top3)
    with tabs[1]:
        st.markdown("### Behavior dimensions and what can actually be measured")
        st.caption(f"Selected recommendation: {selected_course}")
        render_behavior_dimensions(selected_user, user_history, previous, selected_course)
    with tabs[2]:
        st.markdown("### Two explanation methods for the same recommendation")
        render_pair_comparison(selected_user, selected_course,
                               float(chosen_row["recommendation_score"]), key_prefix="platform_view")
    with tabs[3]:
        st.markdown("### Results across the saved historical recommendations")
        render_cohort_comparison()
        with st.expander("Additional earlier analytical exports (if supplied)"):
            for name in ["ExtraTrees_SHAP_Group_Importance.csv", "ExtraTrees_XAI_20_Case_Metrics.csv",
                         "ExtraTrees_LIME_20_Case_Stability.csv", "ExtraTrees_XAI_Statistical_Tests.csv"]:
                table = optional_csv(name)
                if table is not None:
                    st.markdown(f"**{name}**")
                    st.dataframe(table.head(250), hide_index=True, use_container_width=True)
    st.caption("All platform comparisons are descriptive. The proposed strategy has not been prospectively tested or shown to change consumer behavior.")

elif page == "Research Methods":
    st.markdown('<div class="hero"><h1>Research methodology & interpretability</h1>'
                '<p>Historical HarvardX/MITx course-start recommendations, model transparency, '
                'consumer behavior domains and limitations.</p></div>', unsafe_allow_html=True)
    st.markdown("### What the recommender does")
    st.write("A frozen Extra Trees model ranked learner–course candidates by a saved score related to an observed course start during August 1–28, 2013. Each learner is shown their top three ranked courses. Scores are not calibrated enrollment or purchase probabilities, and do not represent current course availability.")
    st.markdown("### Why SHAP and LIME are shown separately")
    st.write("SHAP attributes the model score relative to its saved baseline, using the top 15 individual features and an additive tail. 'Other features (aggregated)' is **not** a feature in any ranking. LIME fits a local linear surrogate using 300 perturbations and a batch-sampled reference; its weights are not directly comparable to SHAP in magnitude. LIME's R² measures fit on its sampled neighborhood, not independent model correctness.")
    st.markdown("### Evaluation of the two explainers")
    st.write("Top-5/Top-10 shared features, Jaccard similarity and direction agreement describe differences between the methods. Agreement is not an optimization target and disagreement by itself is not evidence of an error. Each explanation is loaded from stored SQLite files; the web server does not rerun SHAP, LIME, or model fitting.")
    st.markdown("### Linking behavioral dimensions and marketing hypotheses")
    st.write("Course Participation is related to historical selection features used by the model. Behavioral Engagement and Course Exploration can only be described using separate observed records with verifiable timing; their values are not inputs to this course-start model and must not be assigned invented SHAP contributions. The three dimensions are operational constructs inspired by literature, not a validated joint psychological scale. Proposed marketing strategies are transparent heuristics based on historical-feature SHAP magnitude, not causal claims or predictors of intervention success.")
    st.markdown("### Learner ID selection and public deployment")
    st.write("The displayed IDs come from the existing research export; no names, logins, new users, or identity inference are introduced. A public dropdown with course histories can nonetheless raise data-licensing and indirect re-identification issues. Publishing learner histories or per-case explanations requires research authorization and a licensing/privacy review. The public release tool excludes full histories by default and can optionally create a reduced previous-course display without dates after explicit approval.")

st.sidebar.divider()
st.sidebar.caption("2013 historical research data · precomputed results")
