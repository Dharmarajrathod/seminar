import calendar
import json
import re
import io
import os
from zipfile import ZipFile
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pydeck as pdk
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Mango Flowering and Fruiting Seasonality Across India",
    layout="wide",
)


DATA2_PATH = Path(__file__).resolve().parent / "data2.csv"
KOBO_BASE_URL = "https://eu.kobotoolbox.org"
KOBO_ASSET_UID = os.getenv("KOBO_ASSET_UID", "a4fnScVYw99a9fUv4kL48W")
KOBO_API_TOKEN = os.getenv(
    "KOBO_API_TOKEN",
    "67d9b9b9c5276d1dd7cf717573df7ae1598468e1",
)
MAP_CENTER = {"lat": 22, "lon": 78}
ZONE_BINS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
ZONE_LABELS = [
    "Zone 1", "Zone 2", "Zone 3", "Zone 4",
    "Zone 5", "Zone 6", "Zone 7", "Zone 8",
]
ZONE_BOUNDARIES = [8, 12, 16, 20, 24, 28, 32]
ZONE_EXTENTS = [
    ("Zone 1", 4,  8,  [255, 183, 3,   95]),
    ("Zone 2", 8,  12, [67,  170, 139, 95]),
    ("Zone 3", 12, 16, [42,  157, 143, 95]),
    ("Zone 4", 16, 20, [69,  123, 157, 95]),
    ("Zone 5", 20, 24, [106, 76,  147, 95]),
    ("Zone 6", 24, 28, [168, 218, 220, 95]),
    ("Zone 7", 28, 32, [230, 57,  70,  95]),
    ("Zone 8", 32, 36, [244, 162, 97,  95]),
]
MONTH_ORDER = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December",
]
SEASONAL_MONTH_ORDER = [
    "August", "September", "October", "November", "December",
    "January", "February", "March", "April", "May", "June", "July",
]
PHASE_COLORS = {
    "Leaves":  "#2a9d8f",
    "Flowers": "#facc15",
    "Fruits":  "#dc2626",
}
PLOT_BACKGROUND = "#fffaf2"
PAPER_BACKGROUND = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(106, 76, 147, 0.12)"
STATUS_ORDER = ["Yes", "No", "Unclear", "Unknown"]
NO_SELECTION = "No Selection"
ANALYSIS_START = pd.Timestamp("2020-08-01")
ANALYSIS_END = pd.Timestamp("2021-07-31")
ANALYSIS_ZONE_BINS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
ANALYSIS_MIN_LATITUDE = 8.0
ANALYSIS_MAX_LATITUDE = 37.0
ANALYSIS_MIN_LONGITUDE = 68.0
ANALYSIS_MAX_LONGITUDE = 98.0
ZONE_ANALYSIS_SEASONS = [
    "2020-2021", "2021-2022", "2022-2023",
    "2023-2024", "2024-2025", "2025-2026",
]
ZONE_COLOR_HEX = {
    "Zone 1": "#ffb703", "Zone 2": "#43aa8b", "Zone 3": "#2a9d8f",
    "Zone 4": "#457b9d", "Zone 5": "#6a4c93", "Zone 6": "#a8dadc",
    "Zone 7": "#e63946", "Zone 8": "#f4a261",
}

# ── Notebook bins/labels (match doc-5 exactly) ────────────────────────────────
NB_BINS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
NB_LABELS = ["Zone 1", "Zone 2", "Zone 3", "Zone 4",
             "Zone 5", "Zone 6", "Zone 7", "Zone 8"]

cache_data = getattr(st, "cache_data", None)
if cache_data is None:
    cache_data = st.cache


# ── st.image compatibility (old Streamlit uses use_column_width) ─────────────
def st_image(buf):
    try:
        st.image(buf, use_container_width=True)
    except TypeError:
        buf.seek(0)
        st.image(buf, use_column_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  data loading
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_data(df: pd.DataFrame):
    df = harmonize_kobo_columns(df.copy())

    df["_Location_latitude"] = pd.to_numeric(
        df["_Location_latitude"],  errors="coerce")
    df["_Location_longitude"] = pd.to_numeric(
        df["_Location_longitude"], errors="coerce")

    df["Location_Zone"] = pd.cut(
        df["_Location_latitude"],
        bins=ZONE_BINS,
        labels=ZONE_LABELS,
        include_lowest=True,
    )

    date_candidates = ["Time", "today", "start", "_submission_time"]
    df["Observation_Date"] = pd.NaT
    for column in date_candidates:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
            parsed = parsed.dt.tz_localize(None)
            df["Observation_Date"] = df["Observation_Date"].fillna(parsed)

    df["Observation_Date"] = pd.to_datetime(
        df["Observation_Date"], errors="coerce")
    df["Year"] = df["Observation_Date"].dt.year
    df["Month"] = df["Observation_Date"].dt.month
    df["Month_Name"] = pd.Categorical(
        df["Observation_Date"].dt.month_name(),
        categories=MONTH_ORDER, ordered=True,
    )
    df["Season_Year"] = df["Year"].where(df["Month"] <= 7, df["Year"] + 1)
    df["Season_Year"] = df["Season_Year"].astype("Int64")
    df["Season_Label"] = df["Season_Year"].apply(
        lambda v: f"{int(v) - 1}-{int(v)}" if pd.notna(v) else pd.NA
    )
    df["Season_Month_Name"] = pd.Categorical(
        df["Observation_Date"].dt.month_name(),
        categories=SEASONAL_MONTH_ORDER, ordered=True,
    )

    obs_type = df["Select the Observation type"].fillna(
        "").str.strip().str.lower()
    leaf_status = normalize_status(df["Details of the Tree/Leaves"])
    flower_status = normalize_status(df["Details of the Tree/Flowers"])
    fruit_status = normalize_status(df["Details of the Tree/Fruits"])

    leaf_status = leaf_status.where(~obs_type.eq("leaves sprouting"),  "Yes")
    flower_status = flower_status.where(~obs_type.eq("mango flowering"), "Yes")
    fruit_status = fruit_status.where(~obs_type.eq("mango fruiting"),   "Yes")

    df["Leaves_Status"] = pd.Categorical(
        leaf_status,   categories=STATUS_ORDER, ordered=True)
    df["Flowers_Status"] = pd.Categorical(
        flower_status, categories=STATUS_ORDER, ordered=True)
    df["Fruits_Status"] = pd.Categorical(
        fruit_status,  categories=STATUS_ORDER, ordered=True)

    location_text = (
        df["Location_of_Observation"]
        .fillna("")
        .where(df["Location_of_Observation"].fillna("").str.strip().ne(""),
               df["Location"].fillna(""))
        .replace("", "Unknown location")
    )
    df["Display_Location"] = location_text

    df = df.dropna(subset=["_Location_latitude", "_Location_longitude",
                           "Observation_Date", "Location_Zone"]).copy()
    df["Month"] = df["Month"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df["Season_Year"] = df["Season_Year"].astype(int)

    phase_status_map = {
        "Leaves":  df["Leaves_Status"],
        "Flowers": df["Flowers_Status"],
        "Fruits":  df["Fruits_Status"],
    }
    phase_frames = []
    for phase_name, status_series in phase_status_map.items():
        phase_df = df.copy()
        phase_df["Phenophase"] = phase_name
        phase_df["Observation_Status"] = pd.Categorical(
            status_series, categories=STATUS_ORDER, ordered=True)
        phase_frames.append(phase_df)

    observations = pd.concat(phase_frames, ignore_index=True)
    observations["Phenophase_Color"] = observations["Phenophase"].map(
        PHASE_COLORS)
    observations["Zone_Color"] = observations["Location_Zone"].map(
        ZONE_COLOR_HEX)
    observations["tooltip"] = observations.apply(
        lambda row: (
            f"<b>{row['Phenophase']}</b><br/>"
            f"Status: {row['Observation_Status']}<br/>"
            f"{row['Display_Location']}<br/>"
            f"Zone: {row['Location_Zone']}<br/>"
            f"Date: {row['Observation_Date'].date()}"
        ), axis=1,
    )
    return df, observations


def harmonize_kobo_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = {
        "Select the Observation type": [
            "Select_the_Observation_type",
        ],
        "Details of the Tree/Leaves": [
            "Details_of_the_Tree/Leave_Sprouting",
            "Leaves",
        ],
        "Details of the Tree/Flowers": [
            "Details_of_the_Tree/Flowering",
            "Flowers",
        ],
        "Details of the Tree/Fruits": [
            "Details_of_the_Tree/Fruiting",
            "Fruits",
        ],
        "Location_of_Observation": [
            "Location_of_Observation",
            "Location_of_Observation_001",
        ],
        "Location": [
            "Location",
            "Mark_the_location_of_your_Observation",
            "Mark_your_Location_of_Observation",
        ],
        "Time": [
            "Time",
            "Enter_the_date_and_time_of_observation",
        ],
        "Name of Reporter": [
            "Name_of_the_Reporter",
        ],
    }

    rename_map = {}
    for target, candidates in rename_candidates.items():
        if target in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    if "_Location_latitude" not in df.columns or "_Location_longitude" not in df.columns:
        lat_series, lon_series = extract_geolocation_columns(df)
        if "_Location_latitude" not in df.columns:
            df["_Location_latitude"] = lat_series
        if "_Location_longitude" not in df.columns:
            df["_Location_longitude"] = lon_series

    required_defaults = {
        "Select the Observation type": pd.NA,
        "Details of the Tree/Leaves": pd.NA,
        "Details of the Tree/Flowers": pd.NA,
        "Details of the Tree/Fruits": pd.NA,
        "Location_of_Observation": pd.NA,
        "Location": pd.NA,
        "Time": pd.NA,
        "start": pd.NA,
        "today": pd.NA,
        "_submission_time": pd.NA,
    }
    for column_name, default_value in required_defaults.items():
        if column_name not in df.columns:
            df[column_name] = default_value

    return df


def extract_geolocation_columns(df: pd.DataFrame):
    def coerce_list(value):
        if isinstance(value, list) and len(value) >= 2:
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list) and len(parsed) >= 2:
                        return parsed
                except json.JSONDecodeError:
                    pass
            parts = stripped.replace(",", " ").split()
            if len(parts) >= 2:
                return parts
        return [pd.NA, pd.NA]

    geolocation_sources = [
        "_geolocation",
        "Location",
        "Mark_the_location_of_your_Observation",
        "Mark_your_Location_of_Observation",
    ]
    for source in geolocation_sources:
        if source in df.columns:
            parsed = df[source].apply(coerce_list)
            lat_series = pd.to_numeric(parsed.str[0], errors="coerce")
            lon_series = pd.to_numeric(parsed.str[1], errors="coerce")
            if lat_series.notna().any() and lon_series.notna().any():
                return lat_series, lon_series

    lat_candidates = [
        "_Location_latitude",
        "_Mark_your_Location_of_Observation_latitude",
        "_Mark_the_location_of_your_Observation_latitude",
        "Location_latitude",
    ]
    lon_candidates = [
        "_Location_longitude",
        "_Mark_your_Location_of_Observation_longitude",
        "_Mark_the_location_of_your_Observation_longitude",
        "Location_longitude",
    ]

    lat_series = pd.Series(pd.NA, index=df.index, dtype="object")
    lon_series = pd.Series(pd.NA, index=df.index, dtype="object")

    for candidate in lat_candidates:
        if candidate in df.columns:
            lat_series = pd.to_numeric(df[candidate], errors="coerce")
            break
    for candidate in lon_candidates:
        if candidate in df.columns:
            lon_series = pd.to_numeric(df[candidate], errors="coerce")
            break

    return lat_series, lon_series


def _kobo_request_json(url: str, api_token: str) -> dict:
    request = Request(
        url,
        headers={"Authorization": f"Token {api_token}"},
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


@cache_data(show_spinner=False)
def fetch_kobo_api_dataframe(base_url: str, asset_uid: str, api_token: str) -> pd.DataFrame:
    next_url = f"{base_url}/api/v2/assets/{asset_uid}/data/?format=json&limit=2000"
    records = []

    while next_url:
        payload = _kobo_request_json(next_url, api_token)
        page_results = payload.get("results", [])
        if not isinstance(page_results, list):
            raise ValueError("Kobo data API returned an unexpected payload.")
        records.extend(page_results)
        next_url = payload.get("next")

    if not records:
        raise ValueError("Kobo data API returned no submissions.")

    df = pd.json_normalize(records, sep="/")
    return df


@cache_data(show_spinner=False)
def load_live_data(base_url: str, asset_uid: str, api_token: str):
    df = fetch_kobo_api_dataframe(base_url, asset_uid, api_token)
    return preprocess_data(df)


def load_primary_dataset():
    if not KOBO_API_TOKEN:
        raise RuntimeError("KOBO_API_TOKEN is not configured.")

    try:
        base_df, observations = load_live_data(
            KOBO_BASE_URL, KOBO_ASSET_UID, KOBO_API_TOKEN
        )
        return base_df, observations, "Live Kobo API"
    except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"Kobo API fetch failed: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Kobo API fetch failed: {type(exc).__name__}: {exc}") from exc


def normalize_status(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip().str.lower()
    return cleaned.map({
        "yes": "Yes", "no": "No", "unclear": "Unclear",
        "not clear": "Unclear", "img not found": "Unknown", "": "Unknown",
    }).fillna("Unknown")


# ─────────────────────────────────────────────────────────────────────────────
#  inject_styles  (original — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 221, 168, 0.35), transparent 28%),
                radial-gradient(circle at top right, rgba(168, 218, 220, 0.22), transparent 24%),
                linear-gradient(180deg, #fff8ee 0%, #fffdf7 45%, #f8f3ea 100%);
            color: #2f2419;
            font-family: "DM Sans", sans-serif;
        }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 {
            font-family: "Cormorant Garamond", serif !important;
            letter-spacing: 0.01em; color: #312316;
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(255, 214, 153, 0.35), transparent 30%),
                linear-gradient(180deg, #fff5e3 0%, #f4ecde 52%, #efe4d1 100%);
            border-right: 1px solid rgba(125, 91, 49, 0.12);
            box-shadow: inset -1px 0 0 rgba(255,255,255,0.4);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
        [data-testid="stSidebar"] h2 { font-size: 1.35rem; margin-bottom: 0.35rem; }
        [data-testid="stSidebar"] .stRadio > div,
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            background: transparent !important; border: none !important;
            border-radius: 0 !important; padding: 0 !important; box-shadow: none !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [data-baseweb="radio"] > div {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(249,239,225,0.92) 100%);
            border: 1px solid rgba(170,123,73,0.14);
            border-radius: 16px; padding: 0.7rem 0.9rem;
            margin-bottom: 0.35rem; transition: all 0.2s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover,
        [data-testid="stSidebar"] [data-baseweb="radio"] > div:hover {
            border-color: rgba(42,157,143,0.35);
            background: linear-gradient(135deg, #fffdf9 0%, #f6ead9 100%);
        }
        [data-testid="stSidebar"] [role="radiogroup"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"] span {
            font-weight: 600; color: #2b2118 !important;
            background: transparent !important; margin: 0 !important;
            padding: 0 !important; line-height: 1.2 !important;
            -webkit-text-fill-color: #2b2118 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
        [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] > div,
        [data-testid="stSidebar"] [data-baseweb="radio"] input:checked ~ div {
            background: linear-gradient(135deg, #f2b84b 0%, #d66a2a 100%) !important;
            border: 2px solid #fff2cf !important;
            box-shadow: 0 0 0 2px rgba(214,106,42,0.35), 0 12px 24px rgba(89,63,35,0.18) !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] span {
            color: #1d1208 !important; -webkit-text-fill-color: #1d1208 !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:not([data-checked="true"]) p,
        [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="false"] p,
        [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="false"] span {
            color: #2b2118 !important; -webkit-text-fill-color: #2b2118 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] input,
        [data-testid="stSidebar"] [data-baseweb="radio"] input { accent-color: #2a9d8f !important; }
        [data-testid="stSidebar"] [role="radiogroup"] input + div,
        [data-testid="stSidebar"] [role="radiogroup"] input + div::before,
        [data-testid="stSidebar"] [role="radiogroup"] input + div::after,
        [data-testid="stSidebar"] [data-baseweb="radio"] svg {
            border-color: #2a9d8f !important; background-color: transparent !important;
            box-shadow: inset 0 0 0 1px #2a9d8f !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label svg,
        [data-testid="stSidebar"] [data-baseweb="radio"] label svg {
            fill: #2a9d8f !important; stroke: #2a9d8f !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label * {
            background: transparent !important; box-shadow: none !important;
        }
        [data-testid="stSidebar"] .stMarkdown p { color: #6f5235; }
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stRadio > div > label,
        [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
            color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
        }
        [data-testid="stMetric"] {
            background: rgba(255,251,244,0.88);
            border: 1px solid rgba(167,116,67,0.16);
            border-radius: 18px; padding: 0.9rem 1rem;
            box-shadow: 0 10px 30px rgba(89,63,35,0.06);
        }
        label, .stSelectbox label, .stMultiSelect label, .stMarkdown, .stCaption {
            color: #3a2a1c !important;
        }
        [data-baseweb="select"] > div, [data-baseweb="tag"] {
            background: #fffaf2 !important; color: #3a2a1c !important;
        }
        [data-baseweb="select"] span, [data-baseweb="select"] div,
        [data-baseweb="tag"] span, input { color: #3a2a1c !important; }
        [data-testid="stDataFrame"], .stAlert, .stCodeBlock {
            border-radius: 18px; overflow: hidden;
        }
        .hero {
            padding: 1.6rem 1.7rem; border-radius: 26px;
            background: linear-gradient(135deg, rgba(99,73,45,0.98) 0%, rgba(58,108,93,0.96) 100%);
            color: #fff7ea; box-shadow: 0 18px 48px rgba(53,39,24,0.18); margin-bottom: 1rem;
        }
        .hero-title {
            font-family: "Cormorant Garamond", serif;
            font-size: 2.35rem; line-height: 1; margin-bottom: 0.5rem;
        }
        .hero-copy {
            max-width: 760px; font-size: 1rem; line-height: 1.55;
            color: rgba(255,247,234,0.88);
        }
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
        .chip {
            padding: 0.35rem 0.8rem; border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.14); font-size: 0.82rem;
        }
        .section-card {
            background: rgba(255,251,245,0.82);
            border: 1px solid rgba(167,116,67,0.14);
            border-radius: 24px; padding: 1rem 1rem 0.6rem 1rem;
            box-shadow: 0 14px 40px rgba(89,63,35,0.05); margin-bottom: 1rem;
        }
        .section-label {
            font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.16em; color: #976b41; margin-bottom: 0.15rem;
        }
        .zone-popup {
            background: linear-gradient(180deg, rgba(255,251,245,0.98), rgba(252,246,237,0.96));
            border: 1px solid rgba(167,116,67,0.18);
            border-radius: 24px; padding: 1rem 1rem 0.3rem 1rem;
            box-shadow: 0 18px 44px rgba(89,63,35,0.1); margin-top: 1rem;
        }
        .zone-popup-header {
            display: flex; align-items: center;
            justify-content: space-between; gap: 1rem; margin-bottom: 0.35rem;
        }
        .zone-popup-title {
            font-family: "Cormorant Garamond", serif;
            font-size: 2rem; color: #2f2419; line-height: 1;
        }
        .zone-button-label {
            font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.14em; color: #4d6b5f; margin-bottom: 0.35rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #dff6ea 0%, #9fd8bb 100%);
            color: #16392d !important; border: 1px solid rgba(33,94,72,0.22);
            border-radius: 14px; font-weight: 700;
            box-shadow: 0 10px 20px rgba(42,157,143,0.12);
        }
        .stButton > button:hover {
            border-color: rgba(33,94,72,0.4); color: #0f2d23 !important;
            background: linear-gradient(135deg, #c9efd9 0%, #7fc9a4 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_open(label: str, title: str):
    st.markdown(
        f'<div class="section-card">'
        f'<div class="section-label">{label}</div>'
        f'<h3 style="margin-top:0;">{title}</h3>',
        unsafe_allow_html=True,
    )


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


def style_figure(fig):
    fig.update_layout(
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "DM Sans, sans-serif", "color": "#3a2a1c"},
        title_font={"family": "Cormorant Garamond, serif",
                    "size": 24, "color": "#2f2419"},
        margin={"l": 18, "r": 18, "t": 58, "b": 18},
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def checkbox_filter(label: str, options: list[str], key_prefix: str) -> list[str]:
    st.markdown(f"#### {label}")
    selected = []
    for option in options:
        key = f"{key_prefix}-{re.sub(r'[^a-zA-Z0-9]+', '-', str(option)).strip('-')}"
        if st.checkbox(str(option), value=True, key=key):
            selected.append(option)
    return selected


def apply_filters(data: pd.DataFrame, view_mode: str):
    filtered = data.copy()

    with st.sidebar:
        st.markdown("## Filters")

        zone_options = [
            zone for zone in ZONE_LABELS if zone in filtered["Location_Zone"].dropna().astype(str).unique()
        ]
        selected_zones = checkbox_filter("Zones", zone_options, "filter-zone")

        season_values = [
            season for season in filtered["Season_Label"].dropna().astype(str).unique().tolist()
        ]
        selected_seasons = checkbox_filter("Seasons", sorted(
            season_values,
            key=lambda value: int(value.split("-")[0]) if "-" in value else 0,
        ), "filter-season")

        selected_phases = checkbox_filter(
            "Phenophases", list(PHASE_COLORS.keys()), "filter-phase")
        selected_statuses = checkbox_filter(
            "Statuses", STATUS_ORDER, "filter-status")

    normal_context_df = filtered.copy()
    filtered = filtered[filtered["Location_Zone"].astype(str).isin(selected_zones)]
    filtered = filtered[filtered["Season_Label"].astype(str).isin(selected_seasons)]
    filtered = filtered[filtered["Phenophase"].isin(selected_phases)]
    filtered = filtered[filtered["Observation_Status"].astype(str).isin(selected_statuses)]

    normal_filtered = filtered.copy()
    seasonal_filtered = filtered.copy()
    return {
        "view_mode":         view_mode,
        "selected_zones":    selected_zones,
        "selected_seasons":  selected_seasons,
        "selected_phases":   selected_phases,
        "selected_statuses": selected_statuses,
        "normal_context_df": normal_context_df,
        "normal_filtered":   normal_filtered,
        "seasonal_filtered": seasonal_filtered,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEBOOK LOGIC  (doc-5) — build melted_df for any zone + season
#  Uses the exact same loop as the original Colab notebook.
# ─────────────────────────────────────────────────────────────────────────────

@cache_data
def build_melted_df_for_zone(path: Path, zone_name: str, season_label: str) -> pd.DataFrame:
    """
    Replicates doc-5 notebook loop exactly.
    Loads data.csv, applies the same India filter + zone cut,
    iterates month-by-month for the chosen season window,
    and returns a melted DataFrame with 'Yes Count' per Detail Type.
    """
    # ── Load & clean (identical to doc-5) ─────────────────────────────────────
    df = pd.read_csv(path)
    df["_Location_latitude"] = pd.to_numeric(
        df["_Location_latitude"],  errors="coerce")
    df["_Location_longitude"] = pd.to_numeric(
        df["_Location_longitude"], errors="coerce")

    # same bins/labels as notebook
    df["Location_Zone"] = pd.cut(
        df["_Location_latitude"],
        bins=NB_BINS, labels=NB_LABELS,
        right=True, include_lowest=True,
    )
    df.dropna(subset=["_Location_latitude",
              "_Location_longitude"], inplace=True)
    df = df[
        (df["_Location_latitude"] >= ANALYSIS_MIN_LATITUDE) &
        (df["_Location_latitude"] <= ANALYSIS_MAX_LATITUDE) &
        (df["_Location_longitude"] >= ANALYSIS_MIN_LONGITUDE) &
        (df["_Location_longitude"] <= ANALYSIS_MAX_LONGITUDE)
    ].copy()

    # parse start — try timezone-aware first, fall back to naive
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    if df["start"].dt.tz is not None:
        df["start"] = df["start"].dt.tz_localize(None)

    # ── Season window ──────────────────────────────────────────────────────────
    sy, ey = map(int, season_label.split("-"))
    start_date = date(sy, 8, 1)
    end_date = date(ey, 7, 31)

    # ── Month loop (identical to doc-5) ───────────────────────────────────────
    columns_to_analyze = [
        "Details of the Tree/Leaves",
        "Details of the Tree/Flowers",
        "Details of the Tree/Fruits",
    ]
    results_list = []
    current_month_year = start_date

    while current_month_year <= end_date:
        month_num = current_month_year.month
        year_num = current_month_year.year
        month_name = calendar.month_name[month_num]

        monthly_data = df[
            (df["start"].dt.month == month_num) &
            (df["start"].dt.year == year_num)
        ]
        monthly_zone_data = monthly_data[monthly_data["Location_Zone"] == zone_name]

        counts_dict = {
            "Month":         month_name,
            "Year":          year_num,
            "Location_Zone": zone_name,
            "Total Entries": monthly_zone_data.shape[0],
        }
        for col in columns_to_analyze:
            yes_count = (
                monthly_zone_data[col].value_counts().get("Yes", 0)
                if col in monthly_zone_data.columns else 0
            )
            counts_dict[f"{col} (Yes)"] = yes_count
        results_list.append(counts_dict)

        current_month_year = (
            date(year_num + 1, 1, 1) if month_num == 12
            else date(year_num, month_num + 1, 1)
        )

    # ── Build & melt (identical to doc-5) ─────────────────────────────────────
    monthly_zone_counts_df = pd.DataFrame(results_list)
    monthly_zone_counts_df["Date"] = monthly_zone_counts_df.apply(
        lambda row: pd.to_datetime(f"{row['Year']}-{row['Month']}-01"), axis=1
    )
    monthly_zone_counts_df = monthly_zone_counts_df.sort_values(
        by=["Date", "Location_Zone"]
    ).reset_index(drop=True)

    melted_df = monthly_zone_counts_df.melt(
        id_vars=["Date", "Month", "Year", "Location_Zone", "Total Entries"],
        value_vars=[
            "Details of the Tree/Leaves (Yes)",
            "Details of the Tree/Flowers (Yes)",
            "Details of the Tree/Fruits (Yes)",
        ],
        var_name="Detail Type",
        value_name="Yes Count",
    )
    melted_df["Detail Type"] = (
        melted_df["Detail Type"]
        .str.replace("Details of the Tree/", "", regex=False)
        .str.replace(" (Yes)", "",             regex=False)
    )
    return melted_df


@cache_data
def build_melted_df_for_zone_df(base_df: pd.DataFrame, zone_name: str, season_label: str) -> pd.DataFrame:
    csv_buffer = io.StringIO()
    base_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    temp_df = pd.read_csv(csv_buffer)
    return preprocess_zone_melt(temp_df, zone_name, season_label)


def preprocess_zone_melt(df: pd.DataFrame, zone_name: str, season_label: str) -> pd.DataFrame:
    df["_Location_latitude"] = pd.to_numeric(
        df["_Location_latitude"], errors="coerce"
    )
    df["_Location_longitude"] = pd.to_numeric(
        df["_Location_longitude"], errors="coerce"
    )

    df["Location_Zone"] = pd.cut(
        df["_Location_latitude"],
        bins=NB_BINS, labels=NB_LABELS,
        right=True, include_lowest=True,
    )
    df.dropna(subset=["_Location_latitude",
              "_Location_longitude"], inplace=True)
    df = df[
        (df["_Location_latitude"] >= ANALYSIS_MIN_LATITUDE) &
        (df["_Location_latitude"] <= ANALYSIS_MAX_LATITUDE) &
        (df["_Location_longitude"] >= ANALYSIS_MIN_LONGITUDE) &
        (df["_Location_longitude"] <= ANALYSIS_MAX_LONGITUDE)
    ].copy()

    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    if df["start"].dt.tz is not None:
        df["start"] = df["start"].dt.tz_localize(None)

    sy, ey = map(int, season_label.split("-"))
    start_date = date(sy, 8, 1)
    end_date = date(ey, 7, 31)

    columns_to_analyze = [
        "Details of the Tree/Leaves",
        "Details of the Tree/Flowers",
        "Details of the Tree/Fruits",
    ]
    results_list = []
    current_month_year = start_date

    while current_month_year <= end_date:
        month_num = current_month_year.month
        year_num = current_month_year.year
        month_name = calendar.month_name[month_num]

        monthly_data = df[
            (df["start"].dt.month == month_num) &
            (df["start"].dt.year == year_num)
        ]
        monthly_zone_data = monthly_data[monthly_data["Location_Zone"] == zone_name]

        counts_dict = {
            "Month": month_name,
            "Year": year_num,
            "Location_Zone": zone_name,
            "Total Entries": monthly_zone_data.shape[0],
        }
        for col in columns_to_analyze:
            yes_count = (
                monthly_zone_data[col].value_counts().get("Yes", 0)
                if col in monthly_zone_data.columns else 0
            )
            counts_dict[f"{col} (Yes)"] = yes_count
        results_list.append(counts_dict)

        current_month_year = (
            date(year_num + 1, 1, 1) if month_num == 12
            else date(year_num, month_num + 1, 1)
        )

    monthly_zone_counts_df = pd.DataFrame(results_list)
    monthly_zone_counts_df["Date"] = monthly_zone_counts_df.apply(
        lambda row: pd.to_datetime(f"{row['Year']}-{row['Month']}-01"), axis=1
    )
    monthly_zone_counts_df = monthly_zone_counts_df.sort_values(
        by=["Date", "Location_Zone"]
    ).reset_index(drop=True)

    melted_df = monthly_zone_counts_df.melt(
        id_vars=["Date", "Month", "Year", "Location_Zone", "Total Entries"],
        value_vars=[
            "Details of the Tree/Leaves (Yes)",
            "Details of the Tree/Flowers (Yes)",
            "Details of the Tree/Fruits (Yes)",
        ],
        var_name="Detail Type",
        value_name="Yes Count",
    )
    melted_df["Detail Type"] = (
        melted_df["Detail Type"]
        .str.replace("Details of the Tree/", "", regex=False)
        .str.replace(" (Yes)", "", regex=False)
    )
    return melted_df


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEBOOK CHARTS  (doc-5 style — blue/green lines, value labels on points)
# ─────────────────────────────────────────────────────────────────────────────

def render_notebook_charts(melted_df: pd.DataFrame, zone_name: str, season_label: str):
    """
    Produces the exact same two charts shown in the screenshot:
      • Blue line  — Monthly 'Yes' Counts for Flowers
      • Green line — Monthly 'Yes' Counts for Fruits
    Value labels annotated on every data point, dashed grid, %b %Y x-axis.
    """
    sy, ey = season_label.split("-")
    date_range = f"Aug {sy} - Jul {ey}"

    for detail_type, color in [("Flowers", "blue"), ("Fruits", "green")]:
        plot_df = melted_df[melted_df["Detail Type"] == detail_type].copy()

        st.markdown(
            f"### Monthly 'Yes' Counts for {detail_type} in "
            f"{zone_name} ({date_range})"
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=plot_df, x="Date", y="Yes Count",
            marker="o", markersize=8, linewidth=2.5,
            color=color, ax=ax,
        )
        ax.set_title(
            f"Monthly 'Yes' Counts for {detail_type} in {zone_name} ({date_range})",
            fontsize=16,
        )
        ax.set_xlabel("Date",                   fontsize=12)
        ax.set_ylabel("Number of 'Yes' Counts", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, linestyle="--", alpha=0.6)

        # value labels on each point (identical to doc-5)
        for _, row in plot_df.iterrows():
            ax.text(
                x=row["Date"],
                y=row["Yes Count"],
                s=f"{int(row['Yes Count'])}",
                color="black", fontsize=9,
                ha="center", va="bottom",
            )

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        st_image(buf)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  excel helper  (original — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def excel_serial_to_timestamp(value):
    if pd.isna(value) or value == "":
        return pd.NaT
    try:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    except (TypeError, ValueError):
        return pd.to_datetime(value, errors="coerce")


@cache_data
def load_data2_workbook(path: Path) -> pd.DataFrame:
    namespace = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def cell_text(cell):
        inline = cell.find("a:is", namespace)
        if inline is not None:
            return "".join(node.text or "" for node in inline.iterfind(".//a:t", namespace))
        value = cell.find("a:v", namespace)
        return value.text if value is not None else ""

    records = []
    with ZipFile(path) as workbook:
        root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = root.find("a:sheetData", namespace)
        if rows is None:
            return pd.DataFrame()
        header_map = {}
        for row_index, row in enumerate(rows.findall("a:row", namespace), start=1):
            row_data = {}
            for cell in row.findall("a:c", namespace):
                reference = cell.attrib.get("r", "")
                column_ref = re.sub(r"\d+", "", reference)
                value = cell_text(cell)
                if row_index == 1:
                    header_map[column_ref] = value
                elif column_ref in header_map:
                    row_data[header_map[column_ref]] = value
            if row_index > 1 and row_data:
                records.append(row_data)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    numeric_columns = [
        "_Location_latitude", "_Location_longitude",
        "_Mark your Location of Observation_latitude",
        "_Mark your Location of Observation_longitude",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "start" in df.columns:
        df["start"] = df["start"].apply(excel_serial_to_timestamp)
    else:
        df["start"] = pd.NaT
    return df


def render_phase_filter(active_phase: str):
    st.markdown("#### Select Display")
    phase_cols = st.columns(3)
    phase_options = ["Flowering", "Fruiting", "Both"]
    for col, option in zip(phase_cols, phase_options):
        with col:
            if st.button(option, key=f"phase-filter-{option.lower()}"):
                st.session_state["selected_phase_view"] = option
    st.caption(f"Showing: {active_phase} observations with `Yes` status only.")


# ─────────────────────────────────────────────────────────────────────────────
#  render_zone_analysis_popup
#  Original UI (metrics + plotly charts) KEPT.
#  Notebook matplotlib charts APPENDED below them.
# ─────────────────────────────────────────────────────────────────────────────

def build_monthly_zone_data2(base_df: pd.DataFrame, zone_name: str, season_label: str) -> pd.DataFrame:
    melted_df = build_melted_df_for_zone_df(base_df, zone_name, season_label)
    flower_df = (
        melted_df[melted_df["Detail Type"] == "Flowers"]
        .rename(columns={"Yes Count": "Flowers Yes"})
        [["Date", "Month", "Year", "Location_Zone", "Total Entries", "Flowers Yes"]]
    )
    fruit_df = (
        melted_df[melted_df["Detail Type"] == "Fruits"]
        .rename(columns={"Yes Count": "Fruits Yes"})
        [["Date", "Fruits Yes"]]
    )
    monthly_zone_counts_df = flower_df.merge(fruit_df, on="Date", how="left")
    monthly_zone_counts_df["Month Label"] = monthly_zone_counts_df["Month"]
    monthly_zone_counts_df["Month"] = monthly_zone_counts_df["Date"]
    return monthly_zone_counts_df[[
        "Date", "Month Label", "Year", "Location_Zone",
        "Total Entries", "Flowers Yes", "Fruits Yes", "Month",
    ]]


def render_dataframe(df: pd.DataFrame):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df)


def render_kobo_data_preview(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    section_open("API Data", "Fetched Kobo Records")
    st.caption(
        f"Raw submissions fetched from Kobo API: {len(raw_df):,} rows. "
        f"Filtered processed observations: {len(processed_df):,} rows."
    )

    with st.expander("Raw Kobo API data", expanded=False):
        st.caption(f"Showing all {len(raw_df):,} raw rows.")
        render_dataframe(raw_df)

    with st.expander("Filtered processed data used by charts", expanded=False):
        st.caption(f"Showing all {len(processed_df):,} processed rows.")
        render_dataframe(processed_df)

    section_close()


def selected_season_window(season_label: str):
    season_start_year, season_end_year = map(int, season_label.split("-"))
    return date(season_start_year, 8, 1), date(season_end_year, 7, 31)


def build_filtered_monthly_counts(
    filtered_observations: pd.DataFrame,
    zone_name: str,
    season_label: str,
) -> pd.DataFrame:
    start_date, end_date = selected_season_window(season_label)
    chart_df = filtered_observations[
        (filtered_observations["Location_Zone"].astype(str) == zone_name) &
        (filtered_observations["Season_Label"].astype(str) == season_label)
    ].copy()

    results = []
    current_month_year = start_date
    selected_phases = [
        phase for phase in PHASE_COLORS
        if phase in chart_df["Phenophase"].dropna().astype(str).unique()
    ]

    while current_month_year <= end_date:
        month_num = current_month_year.month
        year_num = current_month_year.year
        month_name = calendar.month_name[month_num]
        month_start = pd.to_datetime(f"{year_num}-{month_num}-01")
        monthly_data = chart_df[
            (chart_df["Observation_Date"].dt.month == month_num) &
            (chart_df["Observation_Date"].dt.year == year_num)
        ]

        for phase in selected_phases:
            phase_data = monthly_data[monthly_data["Phenophase"] == phase]
            results.append({
                "Date": month_start,
                "Month": month_name,
                "Year": year_num,
                "Location_Zone": zone_name,
                "Phenophase": phase,
                "Count": int(phase_data.shape[0]),
                "Total Entries": int(monthly_data.shape[0]),
            })

        current_month_year = (
            date(year_num + 1, 1, 1) if month_num == 12
            else date(year_num, month_num + 1, 1)
        )

    return pd.DataFrame(results)


def render_filtered_monthly_charts(
    filtered_observations: pd.DataFrame,
    selected_zones: list[str],
    selected_seasons: list[str],
    selected_statuses: list[str],
):
    if filtered_observations.empty:
        st.warning("No records match the selected sidebar filters.")
        return

    chart_df = filtered_observations.copy()
    chart_df["Month_Date"] = chart_df["Observation_Date"].dt.to_period("M").dt.to_timestamp()
    monthly_counts = (
        chart_df
        .groupby(["Month_Date", "Phenophase"], observed=True)
        .size()
        .reset_index(name="Count")
        .rename(columns={"Month_Date": "Date"})
        .sort_values(["Date", "Phenophase"])
    )

    status_label = ", ".join(selected_statuses)
    zone_label = ", ".join(selected_zones) if selected_zones else "None"
    season_label = ", ".join(selected_seasons) if selected_seasons else "None"
    st.markdown("### Monthly filtered counts")
    st.caption(
        f"Counts are calculated from the sidebar-filtered Kobo data. "
        f"Zones: {zone_label}. Seasons: {season_label}. "
        f"Included statuses: {status_label}."
    )

    color_map = {
        "Leaves": "#2a9d8f",
        "Flowers": "#1d4ed8",
        "Fruits": "#16a34a",
    }

    for phase in monthly_counts["Phenophase"].dropna().unique():
        plot_df = monthly_counts[monthly_counts["Phenophase"] == phase].copy()
        st.markdown(f"### Monthly counts for {phase}")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=plot_df,
            x="Date",
            y="Count",
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=color_map.get(phase, "#1d4ed8"),
            ax=ax,
        )
        ax.set_title(
            f"Monthly counts for {phase}",
            fontsize=16,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of filtered records", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45, ha="right")

        for _, row in plot_df.iterrows():
            ax.text(
                x=row["Date"],
                y=row["Count"],
                s=f"{int(row['Count'])}",
                color="black",
                fontsize=9,
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        st_image(buf)
        plt.close(fig)

    with st.expander("Monthly filtered counts table", expanded=False):
        render_dataframe(monthly_counts)


def render_zone_analysis_popup(base_df: pd.DataFrame, zone_name: str, season_label: str):
    # ── original plotly section (data2 workbook) ──────────────────────────────
    monthwise_df = build_monthly_zone_data2(base_df, zone_name, season_label)
    presented_df = monthwise_df[[
        "Date", "Month Label", "Year", "Total Entries",
        "Flowers Yes", "Fruits Yes", "Month",
    ]].copy()

    st.session_state["presented_zone_monthwise_data"] = {
        "zone": zone_name, "season": season_label,
        "data": presented_df.copy(),
    }

    zone_color = ZONE_COLOR_HEX.get(zone_name, "#2a9d8f")
    st.markdown('<div class="zone-popup">', unsafe_allow_html=True)

    title_col, close_col = st.columns([6, 1])
    with title_col:
        st.markdown(
            f'<div class="zone-popup-header">'
            f'<div><div class="section-label">Selected Zone</div>'
            f'<div class="zone-popup-title">{zone_name}</div></div></div>',
            unsafe_allow_html=True,
        )
    with close_col:
        st.write("")
        if st.button("Close", key="zone-popup-close"):
            st.markdown("</div>", unsafe_allow_html=True)
            return

    total_flowers = int(
        presented_df["Flowers Yes"].sum()) if not presented_df.empty else 0
    total_fruits = int(
        presented_df["Fruits Yes"].sum()) if not presented_df.empty else 0
    metric_cols = st.columns(2)
    metric_cols[0].metric("Flowering Yes", f"{total_flowers:,}")
    metric_cols[1].metric("Fruiting Yes",  f"{total_fruits:,}")

    # combined plotly trend
    trend_df = presented_df.melt(
        id_vars=["Month"],
        value_vars=["Flowers Yes", "Fruits Yes"],
        var_name="Series", value_name="Count",
    )
    trend_df = trend_df.groupby(["Month", "Series"], as_index=False)[
        "Count"].sum()
    trend_df["Month"] = pd.to_datetime(trend_df["Month"])
    trend_df = trend_df.sort_values("Month")

    linear_fig = px.line(
        trend_df, x="Month", y="Count", color="Series",
        markers=True,
        title=f"Flowering vs Fruiting Trend in {zone_name} ({season_label})",
    )
    linear_fig.update_traces(line_width=4, marker_size=8)
    linear_fig.update_layout(
        xaxis_title="Month", yaxis_title="Observations", hovermode="x unified"
    )
    linear_fig.update_xaxes(
        tickformat="%b %Y",
        tickangle=-45,
        tickfont={"color": "#000000"},
    )
    style_figure(linear_fig)
    st.plotly_chart(linear_fig, use_container_width=True)

    # individual plotly charts side-by-side
    chart_cols = st.columns(2)

    flowers_fig = px.line(
        presented_df, x="Month", y="Flowers Yes", markers=True,
        title=f"Monthly 'Yes' Counts for Flowers in {zone_name} ({season_label})",
    )
    flowers_fig.update_traces(line_color="#1d4ed8",
                              line_width=4, marker_size=11)
    flowers_fig.update_layout(
        xaxis_title="Date", yaxis_title="Number of 'Yes' Counts")
    flowers_fig.update_xaxes(tickformat="%b %Y", tickangle=-45,
                             tickfont={"color": "#000000"},
                             showgrid=True, gridcolor="rgba(58,42,28,0.12)")
    flowers_fig.update_yaxes(showgrid=True, gridcolor="rgba(58,42,28,0.12)")
    style_figure(flowers_fig)
    chart_cols[0].plotly_chart(flowers_fig, use_container_width=True)

    fruits_fig = px.line(
        presented_df, x="Month", y="Fruits Yes", markers=True,
        title=f"Monthly 'Yes' Counts for Fruits in {zone_name} ({season_label})",
    )
    fruits_fig.update_traces(line_color="#16a34a",
                             line_width=4, marker_size=11)
    fruits_fig.update_layout(
        xaxis_title="Date", yaxis_title="Number of 'Yes' Counts")
    fruits_fig.update_xaxes(tickformat="%b %Y", tickangle=-45,
                            tickfont={"color": "#000000"},
                            showgrid=True, gridcolor="rgba(58,42,28,0.12)")
    fruits_fig.update_yaxes(showgrid=True, gridcolor="rgba(58,42,28,0.12)")
    style_figure(fruits_fig)
    chart_cols[1].plotly_chart(fruits_fig, use_container_width=True)

    st.caption(
        f"Focused season: {season_label}.  Map highlight color: {zone_color}."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Map helpers  (original — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_zone_lines() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": f"{b} degree latitude", "path": [[68.0, b], [97.0, b]]}
        for b in ZONE_BOUNDARIES
    ])


def build_zone_polygons() -> pd.DataFrame:
    return pd.DataFrame([
        {"zone": zn, "polygon": [[68.0, mn], [97.0, mn], [
            97.0, mx], [68.0, mx]], "fill_color": c}
        for zn, mn, mx, c in ZONE_EXTENTS
    ])


def build_zone_labels() -> pd.DataFrame:
    return pd.DataFrame([
        {"zone": zn, "coordinates": [82.5, (mn+mx)/2], "label_size": 30}
        for zn, mn, mx, _ in ZONE_EXTENTS
    ])


def render_deck_chart(deck: pdk.Deck, height: int = 500):
    try:
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering deck chart: {e}")


def render_map(filtered: pd.DataFrame):
    try:
        # Ensure required columns exist
        required_columns = [
            "_Location_latitude", "_Location_longitude",
            "Phenophase", "Phenophase_Color", "tooltip", "Location_Zone",
        ]
        for col in required_columns:
            if col not in filtered.columns:
                raise ValueError(f"Missing required column: {col}")

        # Drop rows with missing latitude or longitude
        scatter_data = filtered.dropna(
            subset=["_Location_latitude", "_Location_longitude"]).copy()

        # Map colors for Phenophase
        scatter_data["color"] = scatter_data["Phenophase"].map({
            "Leaves":  [42,  157, 143, 190],
            "Flowers": [250, 204, 21,  190],
            "Fruits":  [220, 38,  38,  190],
        })

        # Create the pydeck.Deck object
        deck = pdk.Deck(
            map_provider="carto",
            map_style="light",
            initial_view_state=pdk.ViewState(
                latitude=MAP_CENTER["lat"], longitude=MAP_CENTER["lon"],
                zoom=4.2, pitch=0,
            ),
            tooltip={"html": "{tooltip}"},
            layers=[
                pdk.Layer("PolygonLayer", data=build_zone_polygons(),
                          get_polygon="polygon", get_fill_color="fill_color",
                          get_line_color=[255, 255, 255, 0], stroked=False, filled=True,
                          pickable=True, auto_highlight=True),
                pdk.Layer("PathLayer", data=build_zone_lines(),
                          get_path="path", get_color=[37, 99, 235, 180],
                          width_scale=1, width_min_pixels=2, pickable=False),
                pdk.Layer("ScatterplotLayer", data=scatter_data,
                          get_position="[_Location_longitude, _Location_latitude]",
                          get_fill_color="color", get_radius=18000,
                          radius_min_pixels=4, radius_max_pixels=12, pickable=True),
            ],
        )

        # Render the map
        render_deck_chart(deck, height=560)

    except Exception as e:
        # Log the error and data for debugging
        st.error(f"Error rendering map: {e}")
        st.write("Filtered Data:", filtered.head())
        st.write("Scatter Data:", scatter_data.head()
                 if 'scatter_data' in locals() else "N/A")


def filter_map_observations_for_phase(filtered: pd.DataFrame, phase_view: str) -> pd.DataFrame:
    yes_only = filtered[filtered["Observation_Status"] == "Yes"].copy()
    if phase_view == "Flowering":
        return yes_only[yes_only["Phenophase"] == "Flowers"].copy()
    if phase_view == "Fruiting":
        return yes_only[yes_only["Phenophase"] == "Fruits"].copy()
    return yes_only[yes_only["Phenophase"].isin(["Flowers", "Fruits"])].copy()


# ─────────────────────────────────────────────────────────────────────────────
#  main  (original — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_styles()
    view_mode = "Normal Visual"

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Mango Flowering and Fruiting Seasonality Across India</div>
            <div class="hero-copy">
                Explore phenology observations across India through latitude-based zones, monthly seasonal
                shifts, and spatial clustering. Use the sidebar to isolate flowering or fruiting activity,
                then read the map and charts together for zone-level seasonality patterns.
            </div>
            <div class="chip-row">
                <div class="chip">Latitude band zoning</div>
                <div class="chip">Interactive India map</div>
                <div class="chip">Monthly seasonality</div>
                <div class="chip">Flowering vs fruiting</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        raw_api_df = fetch_kobo_api_dataframe(
            KOBO_BASE_URL, KOBO_ASSET_UID, KOBO_API_TOKEN
        )
        base_df, data, data_source_label = load_primary_dataset()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Kobo API fetch failed: {type(exc).__name__}: {exc}")
        return
    except Exception as exc:
        st.error(f"Kobo API fetch failed: {type(exc).__name__}: {exc}")
        return

    st.caption(f"Data source: {data_source_label}")

    if data.empty or base_df.empty:
        st.error("No usable flowering or fruiting records were found in the dataset.")
        return

    filters = apply_filters(data, view_mode)
    normal_context_df = filters["normal_context_df"]
    normal_filtered = filters["normal_filtered"]
    filtered_zones = filters["selected_zones"]
    filtered_seasons = filters["selected_seasons"]
    filtered_phases = filters["selected_phases"]
    filtered_statuses = filters["selected_statuses"]

    section_open("Spatial View", "Interactive Map")

    map_filtered = normal_filtered.copy()

    render_map(map_filtered)
    st.caption(
        f"Sidebar filters applied: Zones = {', '.join(filtered_zones) or 'None'}, "
        f"Seasons = {', '.join(filtered_seasons) or 'None'}, "
        f"Phenophases = {', '.join(filtered_phases) or 'None'}, "
        f"Statuses = {', '.join(filtered_statuses) or 'None'}."
    )

    render_filtered_monthly_charts(
        normal_filtered, filtered_zones, filtered_seasons, filtered_statuses)

    section_close()
    render_kobo_data_preview(raw_api_df, normal_filtered)


if __name__ == "__main__":
    main()
