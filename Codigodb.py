"""
Stroke Prediction Dashboard — QUEST Framework
Team: Group 4
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Prediction · QUEST",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL PALETTE & THEME
# ─────────────────────────────────────────────
PALETTE = {
    "bg":         "#0D1117",
    "surface":    "#161B22",
    "surface2":   "#21262D",
    "border":     "#30363D",
    "accent":     "#238636",
    "accent2":    "#1F6FEB",
    "accent3":    "#DA3633",
    "accent4":    "#E3B341",
    "text":       "#E6EDF3",
    "text_muted": "#8B949E",
    "stroke_yes": "#DA3633",
    "stroke_no":  "#1F6FEB",
    "gradient_a": "#238636",
    "gradient_b": "#1F6FEB",
}

SEQ_COLORS = ["#1F6FEB", "#2EA043", "#E3B341", "#DA3633", "#8957E5", "#F78166"]

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
      background-color: {PALETTE["bg"]};
      color: {PALETTE["text"]};
      font-family: 'Inter', sans-serif;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 4px;
      background: {PALETTE["surface"]};
      padding: 6px 10px;
      border-radius: 12px;
      border: 1px solid {PALETTE["border"]};
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border-radius: 8px;
      color: {PALETTE["text_muted"]};
      font-family: 'Syne', sans-serif;
      font-weight: 600;
      font-size: 13px;
      padding: 8px 16px;
      border: none;
      transition: all 0.2s ease;
  }}
  .stTabs [aria-selected="true"] {{
      background: {PALETTE["accent2"]};
      color: {PALETTE["text"]} !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
      padding-top: 24px;
  }}

  /* ── Cards ── */
  .quest-card {{
      background: {PALETTE["surface"]};
      border: 1px solid {PALETTE["border"]};
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 16px;
  }}
  .kpi-card {{
      background: {PALETTE["surface2"]};
      border-left: 3px solid {PALETTE["accent2"]};
      border-radius: 10px;
      padding: 16px 20px;
      text-align: center;
  }}
  .kpi-value {{
      font-family: 'Space Mono', monospace;
      font-size: 28px;
      font-weight: 700;
      color: {PALETTE["text"]};
      line-height: 1.1;
  }}
  .kpi-label {{
      font-size: 11px;
      color: {PALETTE["text_muted"]};
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 4px;
  }}
  .kpi-delta {{
      font-size: 12px;
      margin-top: 6px;
  }}
  .kpi-delta.up   {{ color: {PALETTE["accent3"]}; }}
  .kpi-delta.down {{ color: {PALETTE["accent"]};  }}

  /* ── Missing card ── */
  .missing-card {{
      background: linear-gradient(135deg, {PALETTE["surface2"]}, {PALETTE["surface"]});
      border: 1px solid {PALETTE["border"]};
      border-radius: 12px;
      padding: 18px 22px;
      position: relative;
      overflow: hidden;
  }}
  .missing-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, {PALETTE["accent4"]}, {PALETTE["accent3"]});
  }}
  .mc-variable {{
      font-family: 'Space Mono', monospace;
      font-size: 14px;
      color: {PALETTE["accent4"]};
      font-weight: 700;
  }}
  .mc-count {{
      font-size: 22px;
      font-weight: 700;
      color: {PALETTE["text"]};
  }}
  .mc-pct {{
      font-size: 13px;
      color: {PALETTE["text_muted"]};
  }}

  /* ── Section headers ── */
  .section-title {{
      font-family: 'Syne', sans-serif;
      font-size: 20px;
      font-weight: 800;
      color: {PALETTE["text"]};
      letter-spacing: -0.3px;
      margin-bottom: 8px;
  }}
  .section-subtitle {{
      font-size: 13px;
      color: {PALETTE["text_muted"]};
      margin-bottom: 20px;
      line-height: 1.5;
  }}

  /* ── Hero header ── */
  .hero-wrap {{
      background: linear-gradient(135deg, {PALETTE["surface"]} 0%, {PALETTE["surface2"]} 100%);
      border: 1px solid {PALETTE["border"]};
      border-radius: 16px;
      padding: 40px 48px;
      margin-bottom: 32px;
      position: relative;
      overflow: hidden;
  }}
  .hero-wrap::after {{
      content: '🧠';
      position: absolute;
      right: 48px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 80px;
      opacity: 0.15;
  }}
  .hero-tag {{
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      color: {PALETTE["accent2"]};
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 10px;
  }}
  .hero-title {{
      font-family: 'Syne', sans-serif;
      font-size: 36px;
      font-weight: 800;
      color: {PALETTE["text"]};
      line-height: 1.1;
      margin-bottom: 12px;
  }}
  .hero-body {{
      font-size: 14px;
      color: {PALETTE["text_muted"]};
      max-width: 650px;
      line-height: 1.6;
  }}

  /* ── Letter badge ── */
  .letter-badge {{
      display: inline-block;
      background: linear-gradient(135deg, {PALETTE["accent2"]}, {PALETTE["gradient_a"]});
      color: white;
      font-family: 'Space Mono', monospace;
      font-size: 18px;
      font-weight: 700;
      width: 42px;
      height: 42px;
      border-radius: 10px;
      text-align: center;
      line-height: 42px;
      margin-right: 12px;
      vertical-align: middle;
  }}

  /* ── Insight box ── */
  .insight-box {{
      background: rgba(31, 111, 235, 0.08);
      border-left: 3px solid {PALETTE["accent2"]};
      border-radius: 0 8px 8px 0;
      padding: 12px 16px;
      font-size: 13px;
      color: {PALETTE["text_muted"]};
      margin-top: 12px;
      line-height: 1.5;
  }}
  .insight-box strong {{ color: {PALETTE["text"]}; }}

  /* ── Slider label ── */
  .slider-label {{
      font-family: 'Space Mono', monospace;
      font-size: 12px;
      color: {PALETTE["accent2"]};
      margin-bottom: 4px;
  }}

  /* ── Table ── */
  .dataframe {{ width: 100%; }}
  div[data-testid="stDataFrame"] {{
      border-radius: 10px;
      overflow: hidden;
  }}

  /* ── Divider ── */
  hr {{ border-color: {PALETTE["border"]}; margin: 24px 0; }}

  /* ── Plotly container ── */
  .js-plotly-plot .plotly {{ background: transparent !important; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: {PALETTE["surface"]} !important;
      border-right: 1px solid {PALETTE["border"]};
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=PALETTE["surface"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=12),
    title_font=dict(family="Syne, sans-serif", size=16, color=PALETTE["text"]),
    xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"], color=PALETTE["text_muted"]),
    yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"], color=PALETTE["text_muted"]),
    legend=dict(bgcolor=PALETTE["surface2"], bordercolor=PALETTE["border"], borderwidth=1, font_color=PALETTE["text"]),
    margin=dict(t=50, b=40, l=40, r=20),
    colorway=SEQ_COLORS,
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    import kagglehub, os
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    csv_path = os.path.join(path, "healthcare-dataset-stroke-data.csv")
    df = pd.read_csv(csv_path)
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    df["original_missing_bmi"] = df["bmi"].isna()
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed = imputer.fit_transform(df[features])
    df["bmi"] = imputed[:, features.index("bmi")]
    df["BMI_missing"] = False
    return df


# ─────────────────────────────────────────────
#  CRAMÉR'S V
# ─────────────────────────────────────────────
def cramers_v(x, y):
    cm = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm)[0]
    n = cm.sum().sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return np.nan
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# ─────────────────────────────────────────────
#  HELPER: KPI CARD HTML
# ─────────────────────────────────────────────
def kpi_html(value, label, delta=None, delta_type="up", border_color=None):
    color = border_color or PALETTE["accent2"]
    delta_html = ""
    if delta:
        cls = delta_type
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    return f"""
    <div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


# ─────────────────────────────────────────────
#  HELPER: MISSING CARD HTML
# ─────────────────────────────────────────────
def missing_card_html(var, count, pct):
    bar_width = min(pct * 10, 100)   # scale for visual
    return f"""
    <div class="missing-card">
        <div class="mc-variable">{var}</div>
        <div class="mc-count">{count:,}</div>
        <div class="mc-pct">{pct:.2f}% missing</div>
        <div style="margin-top:10px; background:{PALETTE['border']}; border-radius:4px; height:4px;">
            <div style="width:{bar_width}%; background:linear-gradient(90deg,{PALETTE['accent4']},{PALETTE['accent3']}); height:4px; border-radius:4px;"></div>
        </div>
    </div>"""


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    try:
        df_raw = load_data()
        df = preprocess(df_raw)
        data_loaded = True
    except Exception as e:
        data_loaded = False
        load_error = str(e)

if not data_loaded:
    st.error(f"⚠️ Could not load dataset automatically. Error: {load_error}")
    st.info("Please upload the CSV file manually below.")
    uploaded = st.file_uploader("Upload `healthcare-dataset-stroke-data.csv`", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df = preprocess(df_raw)
        data_loaded = True
    else:
        st.stop()


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_preview, tab_q, tab_u, tab_e, tab_s, tab_t = st.tabs([
    "🧠 Overview",
    "Q · Question",
    "U · Understand",
    "E · Explore",
    "S · Study",
    "T · Tell",
])


# ══════════════════════════════════════════════
#  TAB 0 — OVERVIEW / PREVIEW
# ══════════════════════════════════════════════
with tab_preview:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">Healthcare Analytics · Group 4</div>
        <div class="hero-title">Stroke Prediction<br>Analysis Dashboard</div>
        <div class="hero-body">
            According to the World Health Organization (WHO), stroke is the <strong style="color:#E6EDF3">2nd leading cause
            of death globally</strong>, responsible for approximately <strong style="color:#DA3633">11% of total deaths</strong>.
            This dataset is used to predict whether a patient is likely to get stroke based on input parameters like
            gender, age, various diseases, and smoking status.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──
    total   = len(df)
    strokes = df["stroke"].sum()
    stroke_rate = df["stroke"].mean()
    avg_age = df["age"].mean()
    avg_gluc = df["avg_glucose_level"].mean()
    avg_bmi  = df["bmi"].mean()
    hyp_pct  = df["hypertension"].mean()
    hd_pct   = df["heart_disease"].mean()

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    cards = [
        (c1, f"{total:,}",          "Total Patients",      None,          "up",   PALETTE["accent2"]),
        (c2, f"{strokes:,}",        "Stroke Cases",        "4.87%",       "up",   PALETTE["accent3"]),
        (c3, f"{stroke_rate:.1%}",  "Stroke Rate",         "Imbalanced",  "up",   PALETTE["accent3"]),
        (c4, f"{avg_age:.1f} yr",   "Mean Age",            None,          "down", PALETTE["accent4"]),
        (c5, f"{avg_gluc:.0f}",     "Avg Glucose (mg/dL)", None,          "down", PALETTE["gradient_a"]),
        (c6, f"{avg_bmi:.1f}",      "Avg BMI",             None,          "down", PALETTE["accent2"]),
        (c7, f"{hd_pct:.1%}",       "Heart Disease",       None,          "up",   "#8957E5"),
    ]
    for col, val, lbl, delta, dtype, color in cards:
        with col:
            st.markdown(kpi_html(val, lbl, delta, dtype, color), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    # Donut: stroke distribution
    with col_left:
        st.markdown('<div class="section-title">Stroke Distribution</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["No Stroke", "Stroke"],
            values=df["stroke"].value_counts().values,
            hole=0.65,
            marker_colors=[PALETTE["accent2"], PALETTE["accent3"]],
            textfont_color=PALETTE["text"],
            textinfo="percent+label",
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=True,
            annotations=[dict(text=f"{stroke_rate:.1%}<br><span style='font-size:10px'>Stroke</span>",
                              showarrow=False, font_size=18, font_color=PALETTE["text"])],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Variables overview table
    with col_right:
        st.markdown('<div class="section-title">Dataset Variables</div>', unsafe_allow_html=True)
        meta = pd.DataFrame({
            "Variable":   ["id","gender","age","hypertension","heart_disease",
                           "ever_married","work_type","Residence_type",
                           "avg_glucose_level","bmi","smoking_status","stroke"],
            "Type":       ["ID","Categ.","Num.","Binary","Binary","Categ.","Categ.",
                           "Categ.","Num.","Num.","Categ.","Binary (Target)"],
            "Role":       ["—","Feature","Feature","Feature","Feature","Feature",
                           "Feature","Feature","Feature","Feature","Feature","Target"],
        })
        st.dataframe(meta, use_container_width=True, hide_index=True, height=340)

    st.markdown("""
    <div class="insight-box">
        <strong>Dataset at a glance:</strong> 5,110 patient records with 11 features.
        The dataset is heavily imbalanced (≈95% no-stroke). Only <code>bmi</code> contains missing values (≈3.9%).
        Analysis follows the <strong>Q-U-E-S-T</strong> analytical framework.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 1 — Q: QUESTION
# ══════════════════════════════════════════════
with tab_q:
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <span class="letter-badge">Q</span>
        <span class="section-title" style="margin-bottom:0">Question — Define the Analytical Mission</span>
    </div>
    <div class="section-subtitle">
        Four research questions that guide every step of this exploratory analysis.
    </div>
    """, unsafe_allow_html=True)

    questions = [
        ("Q1", PALETTE["accent2"],  "Demographics & Clinical Characteristics",
         "What patient demographics and clinical characteristics are associated with stroke occurrence?"),
        ("Q2", PALETTE["gradient_a"], "Interaction Effects",
         "Are there interaction effects between patient demographics and clinical variables beyond what each factor suggests independently?"),
        ("Q3", PALETTE["accent4"],  "Missing Values & Class Imbalance",
         "How are missing values and class imbalance distributed, and could they introduce bias in our analysis?"),
        ("Q4", PALETTE["accent3"],  "Class Imbalance & Modeling",
         "Is stroke occurrence evenly distributed in the dataset, or is there significant class imbalance that could affect modeling and interpretation?"),
    ]

    for tag, color, title, body in questions:
        st.markdown(f"""
        <div class="quest-card" style="border-left: 3px solid {color}">
            <span style="font-family:'Space Mono',monospace; font-size:11px; color:{color};
                         text-transform:uppercase; letter-spacing:2px;">{tag}</span>
            <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700;
                        color:{PALETTE['text']}; margin: 6px 0 8px;">{title}</div>
            <div style="font-size:14px; color:{PALETTE['text_muted']}; line-height:1.6;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="quest-card">
        <div class="section-title">Target Audience & Constraints</div>
        <div style="font-size:13px; color:{};  line-height:1.7;">
            <strong style="color:{}"  >Audience:</strong> Clinical analytics team and hospital quality improvement officers.<br>
            <strong style="color:{}"  >Constraints:</strong> Patient privacy requires aggregated reporting — no individual-level identifiers in outputs.
        </div>
    </div>
    """.format(PALETTE["text_muted"], PALETTE["text"], PALETTE["text"]), unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — U: UNDERSTAND
# ══════════════════════════════════════════════
with tab_u:
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <span class="letter-badge">U</span>
        <span class="section-title" style="margin-bottom:0">Understand — Acquire, Inspect & Audit Data</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Missing values section ──
    st.markdown('<div class="section-title" style="margin-top:8px">⚠️ Missing Value Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Identifying and characterizing null values across all features before imputation.</div>', unsafe_allow_html=True)

    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).query("missing_count > 0")

    if missing_df.empty:
        st.info("No missing values found in the dataset.")
    else:
        # Missing cards
        cols_m = st.columns(min(len(missing_df), 4))
        for i, (var, row) in enumerate(missing_df.iterrows()):
            with cols_m[i % len(cols_m)]:
                st.markdown(missing_card_html(var, int(row["missing_count"]), row["missing_pct"]),
                            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── MAR analysis ──
    st.markdown('<div class="section-title">Is BMI Missing at Random?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Comparing key clinical means between patients with and without missing BMI.</div>', unsafe_allow_html=True)

    df_raw2 = df_raw.copy()
    df_raw2["bmi"] = pd.to_numeric(df_raw2["bmi"], errors="coerce")
    df_raw2["BMI_missing"] = df_raw2["bmi"].isna()

    grouped = df_raw2.groupby("BMI_missing")[["age","hypertension","heart_disease","stroke"]].mean().round(4)
    diff_row = ((grouped.loc[True] - grouped.loc[False]) / grouped.loc[False] * 100).round(2)

    # Bar chart comparison
    cols_comp = ["age", "hypertension", "heart_disease", "stroke"]
    vals_false = [grouped.loc[False, c] for c in cols_comp]
    vals_true  = [grouped.loc[True,  c] for c in cols_comp]

    fig_mar = go.Figure()
    fig_mar.add_trace(go.Bar(name="BMI Present",  x=cols_comp, y=vals_false,
                             marker_color=PALETTE["accent2"], opacity=0.85))
    fig_mar.add_trace(go.Bar(name="BMI Missing",  x=cols_comp, y=vals_true,
                             marker_color=PALETTE["accent4"], opacity=0.85))
    fig_mar.update_layout(**PLOTLY_LAYOUT, barmode="group",
                          title="Mean Values: BMI Present vs. BMI Missing")
    apply_theme(fig_mar)
    st.plotly_chart(fig_mar, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Insight:</strong> The missing BMI group is on average <strong>younger</strong> and shows
        <strong>lower rates</strong> of hypertension, heart disease, and stroke. This pattern suggests
        the data is <strong>Missing Not At Random (MNAR)</strong> — younger, healthier patients may not
        have had BMI measured. A <strong>KNN Imputer (k=5, distance-weighted)</strong> was used to
        fill these values using clinical context.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KNN Imputation result ──
    st.markdown('<div class="section-title">KNN Imputation Results</div>', unsafe_allow_html=True)
    bmi_imputed_mean = df[df_raw2["BMI_missing"] == True]["bmi"].mean()
    bmi_original_mean = df_raw2["bmi"].mean()

    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.markdown(kpi_html(f"{int(missing_df.loc['bmi','missing_count'])}",
                             "BMI Null Count (Before)", border_color=PALETTE["accent3"]), unsafe_allow_html=True)
    with ci2:
        st.markdown(kpi_html("0", "BMI Null Count (After)",
                             "✓ Fully imputed", "down", PALETTE["gradient_a"]), unsafe_allow_html=True)
    with ci3:
        st.markdown(kpi_html(f"{bmi_imputed_mean:.2f}",
                             "Mean BMI of Imputed Rows", border_color=PALETTE["accent4"]), unsafe_allow_html=True)

    # BMI distribution before/after
    fig_bmi = go.Figure()
    fig_bmi.add_trace(go.Histogram(
        x=df["bmi"], name="After KNN Imputation",
        nbinsx=60, marker_color=PALETTE["accent2"], opacity=0.7,
        histnorm="percent"))
    fig_bmi.add_trace(go.Histogram(
        x=df_raw2["bmi"].dropna(), name="Original (Non-missing)",
        nbinsx=60, marker_color=PALETTE["gradient_a"], opacity=0.7,
        histnorm="percent"))
    fig_bmi.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                          title="BMI Distribution — Before vs. After Imputation",
                          xaxis_title="BMI", yaxis_title="Percent")
    st.plotly_chart(fig_bmi, use_container_width=True)

    # ── Class imbalance ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Class Imbalance</div>', unsafe_allow_html=True)
    stroke_counts = df["stroke"].value_counts()
    ratio = stroke_counts[0] / stroke_counts[1]

    cj1, cj2, cj3 = st.columns(3)
    with cj1:
        st.markdown(kpi_html(f"{stroke_counts[0]:,}", "No Stroke (Class 0)",
                             f"{stroke_counts[0]/len(df):.1%}", "down", PALETTE["accent2"]), unsafe_allow_html=True)
    with cj2:
        st.markdown(kpi_html(f"{stroke_counts[1]:,}", "Stroke (Class 1)",
                             f"{stroke_counts[1]/len(df):.1%}", "up", PALETTE["accent3"]), unsafe_allow_html=True)
    with cj3:
        st.markdown(kpi_html(f"{ratio:.0f}:1", "Imbalance Ratio",
                             "Requires resampling for ML", "up", PALETTE["accent4"]), unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 — E: EXPLORE
# ══════════════════════════════════════════════
with tab_e:
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <span class="letter-badge">E</span>
        <span class="section-title" style="margin-bottom:0">Explore — Univariate Profiling</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Age slider ── (interactive stroke probability)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎚️ Age & Stroke Risk — Interactive Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Use the slider to define an age window and observe how stroke probability changes across the lifespan.</div>', unsafe_allow_html=True)

    age_range = st.slider(
        "Select Age Range",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(int(df["age"].min()), int(df["age"].max())),
        step=1,
        format="%d yr",
    )

    # Rolling stroke probability by age
    df_age = df.groupby("age")["stroke"].agg(["mean", "sum", "count"]).reset_index()
    df_age.columns = ["age", "stroke_prob", "stroke_cases", "total"]
    df_age_filt = df_age[(df_age["age"] >= age_range[0]) & (df_age["age"] <= age_range[1])]

    # Smoothed using rolling average
    df_age_filt = df_age_filt.copy()
    df_age_filt["smooth_prob"] = df_age_filt["stroke_prob"].rolling(window=5, center=True, min_periods=1).mean()

    # Selected window KPIs
    df_win = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
    w_total   = len(df_win)
    w_strokes = df_win["stroke"].sum()
    w_rate    = df_win["stroke"].mean()

    wa1, wa2, wa3 = st.columns(3)
    with wa1:
        st.markdown(kpi_html(f"{w_total:,}", f"Patients aged {age_range[0]}–{age_range[1]}", border_color=PALETTE["accent2"]), unsafe_allow_html=True)
    with wa2:
        st.markdown(kpi_html(f"{w_strokes:,}", "Stroke Cases in Window", border_color=PALETTE["accent3"]), unsafe_allow_html=True)
    with wa3:
        st.markdown(kpi_html(f"{w_rate:.1%}", "Stroke Rate in Window", border_color=PALETTE["accent4"]), unsafe_allow_html=True)

    fig_age = go.Figure()
    fig_age.add_trace(go.Bar(
        x=df_age_filt["age"], y=df_age_filt["stroke_prob"],
        name="Raw Probability", marker_color=PALETTE["accent3"],
        opacity=0.35, yaxis="y",
    ))
    fig_age.add_trace(go.Scatter(
        x=df_age_filt["age"], y=df_age_filt["smooth_prob"],
        name="Smoothed (5-yr rolling)", mode="lines",
        line=dict(color=PALETTE["accent4"], width=3), yaxis="y",
    ))
    fig_age.add_trace(go.Bar(
        x=df_age_filt["age"], y=df_age_filt["total"],
        name="Patient Count", marker_color=PALETTE["accent2"],
        opacity=0.2, yaxis="y2",
    ))
    fig_age.update_layout(
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=12),
        title=dict(text="Stroke Probability by Age",
                   font=dict(family="Syne, sans-serif", size=16, color=PALETTE["text"])),
        xaxis=dict(title="Age (years)", gridcolor=PALETTE["border"],
                   zerolinecolor=PALETTE["border"], color=PALETTE["text_muted"]),
        yaxis=dict(title="Stroke Probability", tickformat=".0%",
                   gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
                   color=PALETTE["text_muted"]),
        yaxis2=dict(title="Patient Count", overlaying="y", side="right",
                    showgrid=False, color=PALETTE["text_muted"]),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.15, bgcolor=PALETTE["surface2"],
                    bordercolor=PALETTE["border"], borderwidth=1,
                    font_color=PALETTE["text"]),
        margin=dict(t=50, b=40, l=40, r=20),
        colorway=SEQ_COLORS,
    )
    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Key finding:</strong> Stroke probability increases sharply after age 55.
        Patients over 70 show a stroke rate approaching <strong>15–20%</strong>, while those under 40
        exhibit near-zero incidence. Age is the strongest univariate predictor in this dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Numerical distributions ──
    st.markdown('<div class="section-title">Numerical Variables — Distribution by Stroke</div>', unsafe_allow_html=True)

    num_var = st.selectbox("Select variable", ["age", "avg_glucose_level", "bmi"], index=0)

    fig_kde = go.Figure()
    for stroke_val, color, label in [(0, PALETTE["accent2"], "No Stroke"), (1, PALETTE["accent3"], "Stroke")]:
        sub = df[df["stroke"] == stroke_val][num_var].dropna()
        # Kernel density approximation via histogram normalized
        fig_kde.add_trace(go.Violin(
            x=[label] * len(sub), y=sub,
            name=label, fillcolor=color, opacity=0.7,
            line_color=color, meanline_visible=True,
            box_visible=True,
        ))
    fig_kde.update_layout(**PLOTLY_LAYOUT,
                          title=f"Distribution of {num_var.replace('_', ' ').title()} by Stroke",
                          yaxis_title=num_var.replace("_", " ").title(),
                          violingroupgap=0.3, violingap=0.3)
    st.plotly_chart(fig_kde, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Categorical distributions ──
    st.markdown('<div class="section-title">Categorical Variables — Stroke Rate by Category</div>', unsafe_allow_html=True)

    cat_var = st.selectbox("Select categorical variable",
                           ["gender", "hypertension", "heart_disease", "ever_married",
                            "work_type", "smoking_status"], index=0)

    cat_stroke = df.groupby(cat_var)["stroke"].agg(["mean", "sum", "count"]).reset_index()
    cat_stroke.columns = [cat_var, "stroke_rate", "stroke_cases", "total"]
    cat_stroke = cat_stroke.sort_values("stroke_rate", ascending=False)

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        x=cat_stroke[cat_var].astype(str), y=cat_stroke["stroke_rate"],
        marker=dict(color=cat_stroke["stroke_rate"],
                    colorscale=[[0, PALETTE["accent2"]], [1, PALETTE["accent3"]]],
                    showscale=False),
        text=[f"{r:.1%}" for r in cat_stroke["stroke_rate"]],
        textposition="outside", textfont_color=PALETTE["text"],
    ))
    fig_cat.update_layout(**PLOTLY_LAYOUT,
                          title=f"Stroke Rate by {cat_var.replace('_', ' ').title()}",
                          yaxis_title="Stroke Rate", yaxis_tickformat=".1%")
    st.plotly_chart(fig_cat, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — S: STUDY
# ══════════════════════════════════════════════
with tab_s:
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <span class="letter-badge">S</span>
        <span class="section-title" style="margin-bottom:0">Study — Relationships & Patterns</span>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    # ── Pearson correlation heatmap ──
    with col_s1:
        st.markdown('<div class="section-title">Pearson Correlation</div>', unsafe_allow_html=True)
        num_bin_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
        corr = df[num_bin_cols].corr()

        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, PALETTE["accent2"]], [0.5, PALETTE["surface2"]], [1, PALETTE["accent3"]]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont_size=11,
            colorbar=dict(tickfont_color=PALETTE["text"]),
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, title="Pearson Correlation Heatmap",
                               xaxis=dict(tickangle=-35, color=PALETTE["text_muted"]),
                               height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Cramér's V heatmap ──
    with col_s2:
        st.markdown('<div class="section-title">Cramér\'s V Association</div>', unsafe_allow_html=True)
        cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "stroke"]
        cv_mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
        for c1 in cat_cols:
            for c2 in cat_cols:
                cv_mat.loc[c1, c2] = 1.0 if c1 == c2 else cramers_v(df[c1].astype(str), df[c2].astype(str))

        fig_cv = go.Figure(go.Heatmap(
            z=cv_mat.values.astype(float), x=cv_mat.columns, y=cv_mat.columns,
            colorscale=[[0, PALETTE["surface2"]], [0.5, PALETTE["accent2"]], [1, PALETTE["gradient_a"]]],
            zmin=0, zmax=1,
            text=np.round(cv_mat.values.astype(float), 2),
            texttemplate="%{text}",
            textfont_size=11,
            colorbar=dict(tickfont_color=PALETTE["text"]),
        ))
        fig_cv.update_layout(**PLOTLY_LAYOUT, title="Cramér's V Heatmap (Categorical)",
                             xaxis=dict(tickangle=-35, color=PALETTE["text_muted"]),
                             height=420)
        st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Pair scatter ──
    st.markdown('<div class="section-title">Variable Relationships — Scatter Matrix</div>', unsafe_allow_html=True)
    pair_cols = st.multiselect(
        "Select variables for scatter matrix",
        ["age", "avg_glucose_level", "bmi"],
        default=["age", "avg_glucose_level", "bmi"],
    )
    if len(pair_cols) >= 2:
        sample_df = df[pair_cols + ["stroke"]].sample(min(1500, len(df)), random_state=42)
        fig_pair = px.scatter_matrix(
            sample_df, dimensions=pair_cols, color="stroke",
            color_discrete_map={0: PALETTE["accent2"], 1: PALETTE["accent3"]},
            labels={c: c.replace("_", " ").title() for c in pair_cols},
            opacity=0.5,
        )
        fig_pair.update_traces(diagonal_visible=False, marker_size=3)
        fig_pair.update_layout(**PLOTLY_LAYOUT, title="Pairwise Scatter (sample n=1,500)", height=500)
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("Select at least 2 variables.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Interaction: Age × Glucose × Stroke ──
    st.markdown('<div class="section-title">Interaction: Age × Glucose Level × Stroke</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Adjust the glucose threshold to observe how the joint risk profile shifts.</div>', unsafe_allow_html=True)

    gluc_thresh = st.slider("Glucose threshold (mg/dL)", 50, 300, 125, step=5)

    df["high_glucose"] = df["avg_glucose_level"] >= gluc_thresh
    fig_int = px.scatter(
        df.sample(min(2000, len(df)), random_state=0),
        x="age", y="avg_glucose_level", color="stroke",
        color_discrete_map={0: PALETTE["accent2"], 1: PALETTE["accent3"]},
        opacity=0.55, size_max=8,
        labels={"age": "Age", "avg_glucose_level": "Avg Glucose Level", "stroke": "Stroke"},
    )
    fig_int.add_hline(y=gluc_thresh,
                      line=dict(color=PALETTE["accent4"], width=2, dash="dash"),
                      annotation_text=f"Threshold: {gluc_thresh} mg/dL",
                      annotation_font_color=PALETTE["accent4"])
    fig_int.update_layout(**PLOTLY_LAYOUT, title="Age vs. Glucose — Stroke Incidence (sample n=2,000)")
    st.plotly_chart(fig_int, use_container_width=True)

    # Quadrant summary
    q_labels = ["Low age, Low glucose", "Low age, High glucose",
                "High age, Low glucose", "High age, High glucose"]
    age_med = df["age"].median()
    q_rates = []
    for age_cond, gluc_cond in [(df["age"] < age_med, df["avg_glucose_level"] < gluc_thresh),
                                 (df["age"] < age_med, df["avg_glucose_level"] >= gluc_thresh),
                                 (df["age"] >= age_med, df["avg_glucose_level"] < gluc_thresh),
                                 (df["age"] >= age_med, df["avg_glucose_level"] >= gluc_thresh)]:
        sub = df[age_cond & gluc_cond]
        q_rates.append(sub["stroke"].mean() if len(sub) > 0 else 0)

    fig_quad = go.Figure(go.Bar(
        x=q_labels, y=q_rates,
        marker=dict(color=q_rates,
                    colorscale=[[0, PALETTE["accent2"]], [1, PALETTE["accent3"]]],
                    showscale=False),
        text=[f"{r:.1%}" for r in q_rates], textposition="outside",
        textfont_color=PALETTE["text"],
    ))
    fig_quad.update_layout(**PLOTLY_LAYOUT, title="Stroke Rate by Age–Glucose Quadrant",
                           yaxis_title="Stroke Rate", yaxis_tickformat=".1%")
    st.plotly_chart(fig_quad, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 — T: TELL
# ══════════════════════════════════════════════
with tab_t:
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <span class="letter-badge">T</span>
        <span class="section-title" style="margin-bottom:0">Tell — Synthesize & Communicate Findings</span>
    </div>
    <div class="section-subtitle">A structured synthesis of the four analytical questions answered through this EDA.</div>
    """, unsafe_allow_html=True)

    findings = [
        ("Q1", PALETTE["accent2"], "Demographics & Clinical Risk Factors",
         "Age is the strongest predictor of stroke — risk increases sharply after 55. "
         "Hypertension and heart disease are the most discriminating binary clinical features. "
         "Higher average glucose levels are consistently associated with stroke events."),
        ("Q2", PALETTE["gradient_a"], "Interaction Effects",
         "The combination of high age + elevated glucose creates the highest-risk profile, "
         "with stroke rates exceeding 15% in the oldest, high-glucose quartile. "
         "Ever-married status correlates with age and thus proxies age-related risk. "
         "Residence type and gender show minimal independent contribution."),
        ("Q3", PALETTE["accent4"], "Missing Values",
         "Only BMI is missing (~3.9%, 201 rows). The pattern is non-random (MNAR): "
         "missing rows tend to be younger patients with lower comorbidities. "
         "KNN imputation (k=5, distance-weighted) preserves the distributional shape "
         "and is preferred over mean imputation for this clinically-structured missingness."),
        ("Q4", PALETTE["accent3"], "Class Imbalance",
         "The dataset is severely imbalanced: ~95% no-stroke vs. ~5% stroke (ratio ≈19:1). "
         "Any downstream classification model will require oversampling (SMOTE), "
         "undersampling, or cost-sensitive learning. Accuracy alone is a misleading metric — "
         "AUROC, F1-score, and precision-recall curves should be prioritized."),
    ]

    for tag, color, title, body in findings:
        col_badge, col_body = st.columns([0.07, 0.93])
        with col_badge:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color},{PALETTE['bg']});
                        border-radius:10px; padding:12px 8px; text-align:center;
                        font-family:'Space Mono',monospace; font-weight:700; font-size:13px;
                        color:{PALETTE['text']}; margin-top:4px;">{tag}</div>
            """, unsafe_allow_html=True)
        with col_body:
            st.markdown(f"""
            <div class="quest-card" style="border-left:3px solid {color}; margin-top:4px;">
                <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                            color:{PALETTE['text']}; margin-bottom:8px;">{title}</div>
                <div style="font-size:13px; color:{PALETTE['text_muted']}; line-height:1.7;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Summary risk chart ──
    st.markdown('<div class="section-title">Consolidated Risk Factor Profile</div>', unsafe_allow_html=True)

    risk_factors = {
        "Age > 65": df[df["age"] > 65]["stroke"].mean(),
        "Hypertension": df[df["hypertension"] == 1]["stroke"].mean(),
        "Heart Disease": df[df["heart_disease"] == 1]["stroke"].mean(),
        "High Glucose\n(>140)": df[df["avg_glucose_level"] > 140]["stroke"].mean(),
        "BMI Obese\n(>30)": df[df["bmi"] > 30]["stroke"].mean(),
        "Ever Married": df[df["ever_married"] == "Yes"]["stroke"].mean(),
        "Formerly Smoked": df[df["smoking_status"] == "formerly smoked"]["stroke"].mean(),
        "Urban Resident": df[df["Residence_type"] == "Urban"]["stroke"].mean(),
    }
    baseline = df["stroke"].mean()
    risk_df = pd.DataFrame({"Factor": list(risk_factors.keys()),
                            "Stroke Rate": list(risk_factors.values())})
    risk_df["Lift"] = risk_df["Stroke Rate"] / baseline
    risk_df = risk_df.sort_values("Stroke Rate", ascending=True)

    fig_risk = go.Figure()
    fig_risk.add_vline(x=baseline, line=dict(color=PALETTE["accent4"], width=2, dash="dot"),
                       annotation_text=f"Baseline {baseline:.1%}",
                       annotation_font_color=PALETTE["accent4"])
    fig_risk.add_trace(go.Bar(
        y=risk_df["Factor"], x=risk_df["Stroke Rate"],
        orientation="h",
        marker=dict(color=risk_df["Stroke Rate"],
                    colorscale=[[0, PALETTE["accent2"]], [1, PALETTE["accent3"]]],
                    showscale=False),
        text=[f"{r:.1%}  ({l:.1f}x)" for r, l in zip(risk_df["Stroke Rate"], risk_df["Lift"])],
        textposition="outside", textfont_color=PALETTE["text"],
    ))
    fig_risk.update_layout(**PLOTLY_LAYOUT,
                           title="Stroke Rate by Risk Factor vs. Population Baseline",
                           xaxis_title="Stroke Rate", xaxis_tickformat=".1%",
                           height=420, margin=dict(l=160, r=80, t=50, b=40))
    st.plotly_chart(fig_risk, use_container_width=True)

    # ── Next steps ──
    st.markdown("""
    <div class="quest-card" style="border-left: 3px solid #8957E5; margin-top: 8px;">
        <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                    color:#E6EDF3; margin-bottom:10px;">📌 Recommended Next Steps</div>
        <div style="font-size:13px; color:#8B949E; line-height:1.9;">
            <strong style="color:#E6EDF3">1. Address class imbalance</strong> — Apply SMOTE or cost-sensitive classifiers before modeling.<br>
            <strong style="color:#E6EDF3">2. Feature engineering</strong> — Create interaction terms (age × hypertension, age × glucose).<br>
            <strong style="color:#E6EDF3">3. Model selection</strong> — Gradient Boosting (XGBoost/LightGBM) handles imbalance well; tune with AUROC.<br>
            <strong style="color:#E6EDF3">4. Fairness audit</strong> — Evaluate model performance across gender and residence subgroups.<br>
            <strong style="color:#E6EDF3">5. Clinical validation</strong> — Present probability scores to clinical team for threshold calibration.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <div style="text-align:center; margin-top:40px; padding:24px;
                border-top:1px solid {PALETTE['border']};
                font-family:'Space Mono',monospace; font-size:11px; color:{PALETTE['text_muted']};">
        Stroke Prediction Dashboard &nbsp;·&nbsp; QUEST Framework &nbsp;·&nbsp; Group 4 &nbsp;·&nbsp;
        Built with Streamlit & Plotly
    </div>
    """, unsafe_allow_html=True)
