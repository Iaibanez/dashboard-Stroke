"""
Stroke Prediction Dashboard — QUEST Framework
Team: Group 4
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
#  PALETTE — PROFESSIONAL LIGHT MODE
# ─────────────────────────────────────────────
BG         = "#F4F6FA"
SURFACE    = "#FFFFFF"
SURFACE2   = "#EDF0F7"
BORDER     = "#CDD5E0"
ACCENT     = "#1A7F5A"
ACCENT2    = "#1558C0"
ACCENT3    = "#C0392B"
ACCENT4    = "#D97706"
TEXT       = "#1A2332"
TEXT_MUTED = "#566478"
STROKE_YES = "#C0392B"
STROKE_NO  = "#1558C0"

SEQ_COLORS = ["#1558C0","#1A7F5A","#D97706","#C0392B","#7C3AED","#EA580C"]

# ─────────────────────────────────────────────
#  CSS — LIGHT MODE
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

  html, body, .stApp {
      background-color: #F4F6FA !important;
      color: #1A2332;
      font-family: 'Inter', sans-serif;
  }
  .block-container { background-color: #F4F6FA !important; padding-top: 2rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      gap: 4px;
      background: #FFFFFF;
      padding: 6px 10px;
      border-radius: 12px;
      border: 1px solid #CDD5E0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent !important;
      border-radius: 8px;
      color: #566478 !important;
      font-family: 'Syne', sans-serif;
      font-weight: 600;
      font-size: 13px;
      padding: 8px 16px;
      border: none !important;
      transition: all 0.2s ease;
  }
  .stTabs [aria-selected="true"] {
      background: #1558C0 !important;
      color: #FFFFFF !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

  /* Cards */
  .quest-card {
      background: #FFFFFF;
      border: 1px solid #CDD5E0;
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 16px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  }
  .kpi-card {
      background: #FFFFFF !important;
      border: 1px solid #CDD5E0 !important;
      border-left: 4px solid #1558C0 !important;
      border-radius: 10px !important;
      padding: 16px 20px;
      text-align: center;
      box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .kpi-value {
      font-family: 'Space Mono', monospace;
      font-size: 26px;
      font-weight: 700;
      color: #1A2332;
      line-height: 1.1;
  }
  .kpi-label {
      font-size: 11px;
      color: #566478;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 4px;
  }
  .kpi-delta { font-size: 12px; margin-top: 6px; font-weight: 600; }
  .kpi-delta.up   { color: #C0392B; }
  .kpi-delta.down { color: #1A7F5A; }

  /* Missing card */
  .missing-card {
      background: #FFFFFF;
      border: 1px solid #CDD5E0;
      border-radius: 12px;
      padding: 18px 22px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .missing-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, #D97706, #C0392B);
  }
  .mc-variable { font-family: 'Space Mono', monospace; font-size: 14px; color: #D97706; font-weight: 700; }
  .mc-count    { font-size: 22px; font-weight: 700; color: #1A2332; }
  .mc-pct      { font-size: 13px; color: #566478; }

  /* Headers */
  .section-title {
      font-family: 'Syne', sans-serif;
      font-size: 20px;
      font-weight: 800;
      color: #1A2332;
      letter-spacing: -0.3px;
      margin-bottom: 8px;
  }
  .section-subtitle {
      font-size: 13px;
      color: #566478;
      margin-bottom: 20px;
      line-height: 1.5;
  }

  /* Hero */
  .hero-wrap {
      background: linear-gradient(135deg,#E8EFF9 0%,#F0F5FF 50%,#E8F5EE 100%);
      border: 1px solid #CDD5E0;
      border-radius: 16px;
      padding: 40px 48px;
      margin-bottom: 32px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 2px 12px rgba(21,88,192,0.07);
  }
  .hero-wrap::after {
      content: '🧠';
      position: absolute;
      right: 48px; top: 50%;
      transform: translateY(-50%);
      font-size: 80px;
      opacity: 0.10;
  }
  .hero-tag   { font-family: 'Space Mono', monospace; font-size: 11px; color: #1558C0; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
  .hero-title { font-family: 'Syne', sans-serif; font-size: 36px; font-weight: 800; color: #1A2332; line-height: 1.1; margin-bottom: 12px; }
  .hero-body  { font-size: 14px; color: #566478; max-width: 650px; line-height: 1.6; }

  /* Letter badge */
  .letter-badge {
      display: inline-block;
      background: linear-gradient(135deg, #1558C0, #1A7F5A);
      color: white;
      font-family: 'Space Mono', monospace;
      font-size: 18px;
      font-weight: 700;
      width: 42px; height: 42px;
      border-radius: 10px;
      text-align: center;
      line-height: 42px;
      margin-right: 12px;
      vertical-align: middle;
      box-shadow: 0 2px 8px rgba(21,88,192,0.2);
  }

  /* Insight box */
  .insight-box {
      background: rgba(21,88,192,0.05);
      border-left: 3px solid #1558C0;
      border-radius: 0 8px 8px 0;
      padding: 12px 16px;
      font-size: 13px;
      color: #566478;
      margin-top: 12px;
      line-height: 1.5;
  }
  .insight-box strong { color: #1A2332; }

  /* Misc */
  hr { border-color: #CDD5E0; margin: 24px 0; }
  div[data-testid="stDataFrame"] {
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid #CDD5E0;
  }
  [data-testid="stSidebar"] {
      background: #FFFFFF !important;
      border-right: 1px solid #CDD5E0;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  THEME HELPER
# ─────────────────────────────────────────────
def apply_theme(fig, title="", height=None, extra=None):
    layout = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=16, color=TEXT)),
        legend=dict(bgcolor=SURFACE2, bordercolor=BORDER,
                    borderwidth=1, font_color=TEXT),
        margin=dict(t=50, b=40, l=40, r=20),
        colorway=SEQ_COLORS,
    )
    if height:
        layout["height"] = height
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER,
                     color=TEXT_MUTED, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER,
                     color=TEXT_MUTED, linecolor=BORDER)
    return fig


# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    import kagglehub, os
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    return pd.read_csv(os.path.join(path, "healthcare-dataset-stroke-data.csv"))

@st.cache_data
def preprocess(df):
    df = df.copy()
    df["original_missing_bmi"] = df["bmi"].isna()
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    feats = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    imp = KNNImputer(n_neighbors=5, weights="distance")
    df["bmi"] = imp.fit_transform(df[feats])[:, feats.index("bmi")]
    return df

def cramers_v(x, y):
    cm = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm)[0]
    n = cm.sum().sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rc = r - ((r-1)**2)/(n-1)
    kc = k - ((k-1)**2)/(n-1)
    return np.nan if min(kc-1, rc-1) == 0 else np.sqrt(phi2c / min(kc-1, rc-1))

def kpi_html(value, label, delta=None, delta_type="up", border_color=None):
    color = border_color or ACCENT2
    d = f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ""
    return f"""<div class="kpi-card" style="border-left-color:{color} !important;">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>{d}</div>"""

def missing_card_html(var, count, pct):
    w = min(pct * 10, 100)
    return f"""<div class="missing-card">
        <div class="mc-variable">{var}</div>
        <div class="mc-count">{count:,}</div>
        <div class="mc-pct">{pct:.2f}% missing</div>
        <div style="margin-top:10px;background:{BORDER};border-radius:4px;height:4px;">
        <div style="width:{w}%;background:linear-gradient(90deg,{ACCENT4},{ACCENT3});height:4px;border-radius:4px;"></div>
        </div></div>"""


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    try:
        df_raw = load_data()
        df     = preprocess(df_raw)
        data_loaded = True
    except Exception as e:
        data_loaded = False
        load_error  = str(e)

if not data_loaded:
    st.error(f"⚠️ Could not load dataset automatically.\n{load_error}")
    uploaded = st.file_uploader("Upload healthcare-dataset-stroke-data.csv", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df     = preprocess(df_raw)
    else:
        st.stop()


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_prev, tab_q, tab_u, tab_e, tab_s, tab_t = st.tabs([
    "🧠 Overview", "Q · Question", "U · Understand",
    "E · Explore",  "S · Study",   "T · Tell",
])


# ══════════════════════════════════════════════  OVERVIEW
with tab_prev:
    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-tag">Healthcare Analytics · Group 4</div>
      <div class="hero-tag">Gina Paola Huguet Berdugo, Iván Andres Ibañez Castillo, Leiry Laura Mares Cure, Jorge Ernesto Salazar Arrieta </div>
      <div class="hero-title">Stroke Prediction<br>Analysis Dashboard</div>
      <div class="hero-body">
        According to the World Health Organization (WHO), stroke is the
        <strong style="color:{TEXT}">2nd leading cause of death globally</strong>,
        responsible for approximately <strong style="color:{ACCENT3}">11% of total deaths</strong>.
        This dataset predicts whether a patient is likely to get a stroke based on parameters like
        gender, age, various diseases, and smoking status.
      </div>
    </div>""", unsafe_allow_html=True)

    total   = len(df)
    strokes = df["stroke"].sum()
    sr      = df["stroke"].mean()

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    for col, val, lbl, delta, dt, clr in [
        (c1, f"{total:,}",                       "Total Patients",     None,               "up",   ACCENT2),
        (c2, f"{strokes:,}",                     "Stroke Cases",       "4.87%",            "up",   ACCENT3),
        (c3, f"{sr:.1%}",                        "Stroke Rate",        "Imbalanced",       "up",   ACCENT3),
        (c4, f"{df['age'].mean():.1f} yr",        "Mean Age",           None,               "down", ACCENT4),
        (c5, f"{df['avg_glucose_level'].mean():.0f}", "Avg Glucose",    None,               "down", ACCENT),
        (c6, f"{df['bmi'].mean():.1f}",           "Avg BMI",            None,               "down", ACCENT2),
        (c7, f"{df['heart_disease'].mean():.1%}", "Heart Disease",      None,               "up",   "#7C3AED"),
    ]:
        with col:
            st.markdown(kpi_html(val, lbl, delta, dt, clr), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    cl, cr = st.columns([1.3, 1])

    with cl:
        st.markdown('<div class="section-title">Stroke Distribution</div>', unsafe_allow_html=True)
        fig_d = go.Figure(go.Pie(
            labels=["No Stroke","Stroke"],
            values=df["stroke"].value_counts().values,
            hole=0.65,
            marker_colors=[ACCENT2, ACCENT3],
            textfont_color=TEXT,
            textinfo="percent+label",
        ))
        apply_theme(fig_d, extra=dict(showlegend=True,
            annotations=[dict(text=f"{sr:.1%}<br><span style='font-size:10px'>Stroke</span>",
                              showarrow=False, font_size=18, font_color=TEXT)]))
        st.plotly_chart(fig_d, use_container_width=True)

    with cr:
        st.markdown('<div class="section-title">Dataset Variables</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Variable": ["id","gender","age","hypertension","heart_disease","ever_married",
                         "work_type","Residence_type","avg_glucose_level","bmi","smoking_status","stroke"],
            "Type":     ["int","object","float","int","int","object","object","object","float","float","object","int"],
            "Role":     ["ID","Feature","Feature","Feature","Feature","Feature",
                         "Feature","Feature","Feature","Feature","Feature","Target"],
        }), use_container_width=True, hide_index=True)
        
    st.markdown("""
        <div class="insight-box">
            <strong>Dataset at a glance:</strong> 5,110 patient records with 11 features.
            The dataset is heavily imbalanced (≈95% no-stroke). Only <code>bmi</code> contains missing values (≈3.9%).
            Analysis follows the <strong>Q-U-E-S-T</strong> analytical framework.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════  Q
with tab_q:
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:6px;">
      <span class="letter-badge">Q</span>
      <span class="section-title" style="margin-bottom:0">Question — Define the Analytical Mission</span>
    </div>
    <div class="section-subtitle">Four research questions guiding this exploratory analysis.</div>
    """, unsafe_allow_html=True)

    for tag, clr, ttl, body in [
        ("Q1", ACCENT2,  "Demographics & Clinical Characteristics",
         "What patient demographics and clinical characteristics are associated with stroke occurrence?"),
        ("Q2", ACCENT,   "Interaction Effects",
         "Are there interaction effects between demographics and clinical variables beyond what each factor suggests independently?"),
        ("Q3", ACCENT4,  "Missing Values & Class Imbalance",
         "How are missing values and class imbalance distributed, and could they introduce bias in our analysis?"),
        ("Q4", ACCENT3,  "Class Imbalance & Modeling",
         "Is stroke occurrence evenly distributed, or is there significant class imbalance that could affect modeling?"),
    ]:
        st.markdown(f"""<div class="quest-card" style="border-left:3px solid {clr}">
            <span style="font-family:'Space Mono',monospace;font-size:11px;color:{clr};text-transform:uppercase;letter-spacing:2px;">{tag}</span>
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:{TEXT};margin:6px 0 8px;">{ttl}</div>
            <div style="font-size:14px;color:{TEXT_MUTED};line-height:1.6;">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""<div class="quest-card">
      <div class="section-title">Target Audience & Constraints</div>
      <div style="font-size:13px;color:{TEXT_MUTED};line-height:1.7;">
        <strong style="color:{TEXT}">Audience:</strong> Clinical analytics team and hospital quality improvement officers.<br>
        <strong style="color:{TEXT}">Constraints:</strong> Patient privacy requires aggregated reporting — no individual-level identifiers in outputs.
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════  U
with tab_u:
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:6px;">
      <span class="letter-badge">U</span>
      <span class="section-title" style="margin-bottom:0">Understand — Acquire, Inspect & Audit Data</span>
    </div>""", unsafe_allow_html=True)

    # Missing audit
    st.markdown('<div class="section-title" style="margin-top:8px">⚠️ Missing Value Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Identifying null values across all features before imputation.</div>', unsafe_allow_html=True)
    # CITA DEL ARTICULO
    st.markdown("""
        <div class="insight-box">
          <strong>Reference for Missing-Data Handling:</strong><br>
          <span style="font-style: italic;">
            A Comprehensive Review of Handling Missing Data: Exploring Special Missing Mechanisms
          </span><br>
          <span style="opacity: 0.9;">
            Youran Zhou, Sunil Aryal, Mohamed Reda Bouadjeneka
          </span>
        </div>
        """, unsafe_allow_html=True)

# ── Imagen + descripción ──
st.image(
    "RUTA_DE_LA_IMAGEN",          # ← reemplaza con la ruta o URL de tu imagen
    use_container_width=True,
)
st.markdown("""
    <div class="section-subtitle" style="margin-top: 8px;">
        En el articulo se llega a la conclusion anterior, donde logran estandarizar los metodos de eliminacion e imputacion que existen en la literatura para cada escenario de datos faltantes.
    </div>
    """, unsafe_allow_html=True)

    miss     = df_raw.isnull().sum()
    miss_pct = (miss / len(df_raw) * 100).round(2)
    miss_df  = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct}).query("missing_count > 0")

    if miss_df.empty:
        st.info("No missing values found.")
    else:
        cols_m = st.columns(min(len(miss_df), 4))
        for i, (var, row) in enumerate(miss_df.iterrows()):
            with cols_m[i % len(cols_m)]:
                st.markdown(missing_card_html(var, int(row["missing_count"]), row["missing_pct"]),
                            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── MAR: % difference chart ──
    # Uses the same logic as EDA: groupby BMI_missing, compute % diff vs BMI-present group
    st.markdown('<div class="section-title">Is BMI Missing at Random?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">'
        'Percentage difference between the BMI-missing group and the BMI-present group for each clinical variable. '
        'A negative value means the missing-BMI group has a <strong>lower</strong> rate than those with BMI recorded.'
        '</div>', unsafe_allow_html=True)

    # Build MAR table exactly as in original EDA
    df_raw2 = df_raw.copy()
    df_raw2["bmi"] = pd.to_numeric(df_raw2["bmi"], errors="coerce")
    df_raw2["BMI_missing"] = df_raw2["bmi"].isna()

    grp     = df_raw2.groupby("BMI_missing")[["age","hypertension","heart_disease","stroke"]].mean()
    base    = grp.loc[False]   # BMI present = reference (denominator)
    comp    = grp.loc[True]    # BMI missing = comparison
    pct_diff = ((comp - base) / base * 100).round(1)

    vars_mar  = pct_diff.index.tolist()
    vals_diff = pct_diff.values.tolist()
    # Red = missing group is lower (negative % diff), Blue = missing group is higher
    bar_clrs = [ACCENT3 if v < 0 else ACCENT2 for v in vals_diff]

    # Reference table
    ref_tbl = pd.DataFrame({
        "Variable":           vars_mar,
        "BMI Present (mean)": grp.loc[False].round(4).values,
        "BMI Missing (mean)": grp.loc[True].round(4).values,
        "% Difference":       [f"{v:+.1f}%" for v in vals_diff],
    })

    col_mar, col_tbl = st.columns([1.6, 1])

    with col_mar:
        fig_mar = go.Figure()
        fig_mar.add_trace(go.Bar(
            x=vars_mar,
            y=vals_diff,
            marker_color=bar_clrs,
            text=[f"{v:+.1f}%" for v in vals_diff],
            textposition="outside",
            textfont=dict(color=TEXT, size=12, family="Space Mono"),
        ))
        fig_mar.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1.5, dash="dash"))
        apply_theme(fig_mar, title="% Difference: BMI-Missing vs. BMI-Present Group")
        fig_mar.update_yaxes(title_text="% Difference vs. BMI-Present group", ticksuffix="%")
        fig_mar.update_xaxes(title_text="Clinical Variable")
        st.plotly_chart(fig_mar, use_container_width=True)

    with col_tbl:
        st.markdown('<div class="section-title" style="font-size:15px;margin-top:28px">Reference Means</div>',
                    unsafe_allow_html=True)
        st.dataframe(ref_tbl, use_container_width=True, hide_index=True)

    st.markdown("""<div class="insight-box">
        <strong>Insight:</strong> The missing-BMI group is on average <strong>younger</strong> and shows
        <strong>lower rates</strong> of hypertension, heart disease, and stroke — suggesting data is
        <strong>Missing Not At Random (MNAR)</strong>. A <strong>KNN Imputer (k=5, distance-weighted)</strong>
        was used to fill these values preserving clinical context.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # KNN imputation results
    st.markdown('<div class="section-title">KNN Imputation Results</div>', unsafe_allow_html=True)
    bmi_imp_mean = df[df_raw2["BMI_missing"] == True]["bmi"].mean()

    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.markdown(kpi_html(f"{int(miss_df.loc['bmi','missing_count'])}",
                             "BMI Null Count (Before)", border_color=ACCENT3), unsafe_allow_html=True)
    with ci2:
        st.markdown(kpi_html("0","BMI Null Count (After)","✓ Fully imputed","down",
                             ACCENT), unsafe_allow_html=True)
    with ci3:
        st.markdown(kpi_html(f"{bmi_imp_mean:.2f}","Mean BMI of Imputed Rows",
                             border_color=ACCENT4), unsafe_allow_html=True)

    fig_bmi = go.Figure()
    fig_bmi.add_trace(go.Histogram(x=df["bmi"], name="After KNN Imputation",
        nbinsx=60, marker_color=ACCENT2, opacity=0.7, histnorm="percent"))
    fig_bmi.add_trace(go.Histogram(x=df_raw2["bmi"].dropna(), name="Original (Non-missing)",
        nbinsx=60, marker_color=ACCENT, opacity=0.7, histnorm="percent"))
    apply_theme(fig_bmi, title="BMI Distribution — Before vs. After Imputation",
                extra=dict(barmode="overlay"))
    fig_bmi.update_xaxes(title_text="BMI")
    fig_bmi.update_yaxes(title_text="Percent (%)")
    st.plotly_chart(fig_bmi, use_container_width=True)

    # Class imbalance
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Class Imbalance</div>', unsafe_allow_html=True)
    sc    = df["stroke"].value_counts()
    ratio = sc[0] / sc[1]
    cj1, cj2, cj3 = st.columns(3)
    with cj1:
        st.markdown(kpi_html(f"{sc[0]:,}","No Stroke (Class 0)",f"{sc[0]/len(df):.1%}",
                             "down", ACCENT2), unsafe_allow_html=True)
    with cj2:
        st.markdown(kpi_html(f"{sc[1]:,}","Stroke (Class 1)",f"{sc[1]/len(df):.1%}",
                             "up", ACCENT3), unsafe_allow_html=True)
    with cj3:
        st.markdown(kpi_html(f"{ratio:.0f}:1","Imbalance Ratio","Requires resampling for ML",
                             "up", ACCENT4), unsafe_allow_html=True)


# ══════════════════════════════════════════════  E  — Univariate Profiling
with tab_e:
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:6px;">
      <span class="letter-badge">E</span>
      <span class="section-title" style="margin-bottom:0">Explore — Univariate Profiling</span>
    </div>
    <div class="section-subtitle">Independent distribution of each variable — shape, central tendency, spread, and frequency. No cross-variable relationships here.</div>
    """, unsafe_allow_html=True)

    # ── Descriptive Statistics Table ──
    st.markdown('<div class="section-title">📋 Descriptive Statistics — Numerical Variables</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Summary statistics (mean, median, std, min, max, quartiles) for all continuous features in the dataset.</div>',
                unsafe_allow_html=True)

    num_desc_cols = ["age", "avg_glucose_level", "bmi"]
    desc = df[num_desc_cols].describe().T.round(2)
    desc.index = ["Age (years)", "Avg Glucose (mg/dL)", "BMI"]
    desc.columns = ["Count", "Mean", "Std Dev", "Min", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max"]
    st.dataframe(desc, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Numerical Histograms ──
    st.markdown('<div class="section-title">📊 Numerical Variables — Individual Distributions</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Histograms with density overlay for each continuous feature — examined independently, without grouping by stroke outcome.</div>',
                unsafe_allow_html=True)

    num_uni_var = st.selectbox("Select variable to explore", ["age", "avg_glucose_level", "bmi"], index=0)

    col_hist, col_box = st.columns([1.6, 1])

    with col_hist:
        series = df[num_uni_var].dropna()
        mean_v = series.mean()
        med_v  = series.median()
        std_v  = series.std()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=series, nbinsx=50,
            marker_color=ACCENT2, opacity=0.75,
            name="Frequency",
            histnorm="percent",
        ))
        fig_hist.add_vline(x=mean_v, line=dict(color=ACCENT3, width=2, dash="dash"),
                           annotation_text=f"Mean: {mean_v:.1f}",
                           annotation_font_color=ACCENT3,
                           annotation_position="top right")
        fig_hist.add_vline(x=med_v, line=dict(color=ACCENT4, width=2, dash="dot"),
                           annotation_text=f"Median: {med_v:.1f}",
                           annotation_font_color=ACCENT4,
                           annotation_position="top left")
        apply_theme(fig_hist,
                    title=f"Distribution of {num_uni_var.replace('_',' ').title()} (full dataset)",
                    height=380)
        fig_hist.update_xaxes(title_text=num_uni_var.replace("_"," ").title())
        fig_hist.update_yaxes(title_text="Percent (%)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=series, name=num_uni_var.replace("_"," ").title(),
            marker_color=ACCENT2, line_color=ACCENT2,
            boxmean="sd", fillcolor=ACCENT2,
            opacity=0.65,
        ))
        apply_theme(fig_box,
                    title=f"Box Plot — {num_uni_var.replace('_',' ').title()}",
                    height=380)
        fig_box.update_yaxes(title_text=num_uni_var.replace("_"," ").title())
        st.plotly_chart(fig_box, use_container_width=True)

    # KPIs for selected variable
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl, clr in [
        (k1, f"{mean_v:.2f}",  "Mean",    ACCENT2),
        (k2, f"{med_v:.2f}",   "Median",  ACCENT4),
        (k3, f"{std_v:.2f}",   "Std Dev", ACCENT),
        (k4, f"{series.skew():.2f}", "Skewness", ACCENT3),
    ]:
        with col:
            st.markdown(kpi_html(val, lbl, border_color=clr), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── All 3 numerical distributions side by side ──
    st.markdown('<div class="section-title">📈 All Numerical Variables — Side-by-Side Overview</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Quick visual comparison of the shape and spread of each continuous feature across the full dataset.</div>',
                unsafe_allow_html=True)

    ov_cols = st.columns(3)
    for i, (vc, vl) in enumerate([("age","Age (years)"), ("avg_glucose_level","Avg Glucose (mg/dL)"), ("bmi","BMI")]):
        with ov_cols[i]:
            s = df[vc].dropna()
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Histogram(
                x=s, nbinsx=40,
                marker_color=SEQ_COLORS[i], opacity=0.8,
                histnorm="percent", name=vl,
            ))
            fig_ov.add_vline(x=s.mean(), line=dict(color=TEXT_MUTED, width=1.5, dash="dash"))
            apply_theme(fig_ov, title=vl, height=300)
            fig_ov.update_xaxes(title_text=vl)
            fig_ov.update_yaxes(title_text="%")
            st.plotly_chart(fig_ov, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Categorical Frequency Charts ──
    st.markdown('<div class="section-title">🗂️ Categorical Variables — Frequency Distribution</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Count and proportion of each category within each categorical feature, examined independently.</div>',
                unsafe_allow_html=True)

    cat_uni_vars = [
        ("gender",          "Gender"),
        ("ever_married",    "Ever Married"),
        ("work_type",       "Work Type"),
        ("Residence_type",  "Residence Type"),
        ("smoking_status",  "Smoking Status"),
        ("hypertension",    "Hypertension (0/1)"),
        ("heart_disease",   "Heart Disease (0/1)"),
        ("stroke",          "Stroke — Target (0/1)"),
    ]

    for i in range(0, len(cat_uni_vars), 2):
        row_cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(cat_uni_vars):
                break
            vk, vl = cat_uni_vars[idx]
            with row_cols[j]:
                vc_counts = df[vk].astype(str).value_counts().reset_index()
                vc_counts.columns = [vk, "count"]
                vc_counts["pct"] = vc_counts["count"] / vc_counts["count"].sum()
                vc_counts = vc_counts.sort_values("count", ascending=False)

                fig_vc = go.Figure()
                fig_vc.add_trace(go.Bar(
                    x=vc_counts[vk],
                    y=vc_counts["count"],
                    marker_color=SEQ_COLORS[:len(vc_counts)],
                    text=[f"{p:.1%}" for p in vc_counts["pct"]],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=11),
                    hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
                ))
                apply_theme(fig_vc, title=f"Frequency: {vl}", height=300)
                fig_vc.update_yaxes(title_text="Count")
                fig_vc.update_xaxes(title_text=vl)
                st.plotly_chart(fig_vc, use_container_width=True)

    st.markdown("""<div class="insight-box">
        <strong>Univariate observations:</strong> Age is right-skewed with a concentration of patients
        between 40–70 years. Glucose is bimodal, suggesting two sub-populations. BMI approximates a
        normal distribution centred ~28. The dataset is dominated by female patients, married individuals,
        and private-sector workers. Stroke (target) is severely imbalanced at ≈5% positive class.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════  S  — Relationships & Patterns
with tab_s:
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:6px;">
      <span class="letter-badge">S</span>
      <span class="section-title" style="margin-bottom:0">Study — Multivariate Analysis & Relationships</span>
    </div>
    <div class="section-subtitle">Cross-variable patterns: how each feature relates to stroke outcome and to other variables.</div>
    """, unsafe_allow_html=True)

    # ── Numerical Variables by Stroke (moved from E) ──
    st.markdown('<div class="section-title">📦 Numerical Variables — Distribution by Stroke Outcome</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Violin + box plots comparing the distribution of each continuous feature between stroke and non-stroke patients.</div>',
                unsafe_allow_html=True)

    num_var = st.selectbox("Select variable", ["age", "avg_glucose_level", "bmi"], index=0)
    fig_kde = go.Figure()
    for sv, clr, lbl in [(0, ACCENT2, "No Stroke"), (1, ACCENT3, "Stroke")]:
        sub = df[df["stroke"] == sv][num_var].dropna()
        fig_kde.add_trace(go.Violin(x=[lbl]*len(sub), y=sub, name=lbl,
            fillcolor=clr, opacity=0.7, line_color=clr,
            meanline_visible=True, box_visible=True))
    apply_theme(fig_kde,
                title=f"Distribution of {num_var.replace('_',' ').title()} by Stroke Outcome",
                extra=dict(violingroupgap=0.3, violingap=0.3))
    fig_kde.update_yaxes(title_text=num_var.replace("_"," ").title())
    st.plotly_chart(fig_kde, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Categorical Stroke Rate (moved from E) ──
    st.markdown('<div class="section-title">📊 Categorical Variables — Stroke Rate by Category</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Stroke rate for every category of each categorical feature, sorted from highest to lowest risk. This is a bivariate view: category × stroke outcome.</div>',
                unsafe_allow_html=True)

    cat_vars_s = [
        ("gender",         "Gender"),
        ("hypertension",   "Hypertension"),
        ("heart_disease",  "Heart Disease"),
        ("ever_married",   "Ever Married"),
        ("work_type",      "Work Type"),
        ("smoking_status", "Smoking Status"),
    ]

    for i in range(0, len(cat_vars_s), 2):
        row_cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(cat_vars_s):
                break
            vk, vl = cat_vars_s[idx]
            with row_cols[j]:
                cs = (df.groupby(vk)["stroke"]
                        .agg(["mean","sum","count"])
                        .reset_index())
                cs.columns = [vk, "stroke_rate", "stroke_cases", "total"]
                cs = cs.sort_values("stroke_rate", ascending=False)

                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(
                    x=cs[vk].astype(str),
                    y=cs["stroke_rate"],
                    marker=dict(
                        color=cs["stroke_rate"].tolist(),
                        colorscale=[[0, ACCENT2], [1, ACCENT3]],
                        showscale=False,
                    ),
                    text=[f"{r:.1%}" for r in cs["stroke_rate"]],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=11),
                    customdata=cs[["stroke_cases","total"]].values,
                    hovertemplate=(
                        "<b>%{x}</b><br>Stroke Rate: %{y:.1%}<br>"
                        "Cases: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>"
                    ),
                ))
                apply_theme(fig_c, title=f"Stroke Rate by {vl}", height=320)
                fig_c.update_yaxes(title_text="Stroke Rate", tickformat=".1%")
                fig_c.update_xaxes(title_text=vl)
                st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("""<div class="insight-box">
        <strong>Key patterns:</strong> Hypertension and heart disease show the highest relative stroke
        rates. For smoking status, formerly-smoked patients show elevated risk, likely reflecting age
        confounding. Work type shows high rates for self-employed patients, also likely age-related.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Age slider (moved from Explore) ──
    st.markdown('<div class="section-title">🎚️ Age & Stroke Risk — Interactive Explorer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Use the slider to define an age window and observe how stroke probability changes.</div>',
                unsafe_allow_html=True)

    age_range = st.slider(
        "Select Age Range",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(int(df["age"].min()), int(df["age"].max())),
        step=1, format="%d yr"
    )

    df_age = df.groupby("age")["stroke"].agg(["mean","sum","count"]).reset_index()
    df_age.columns = ["age","stroke_prob","stroke_cases","total"]
    daf = df_age[(df_age["age"] >= age_range[0]) & (df_age["age"] <= age_range[1])].copy()
    daf["smooth"] = daf["stroke_prob"].rolling(5, center=True, min_periods=1).mean()

    dw = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
    wa1, wa2, wa3 = st.columns(3)
    with wa1:
        st.markdown(kpi_html(f"{len(dw):,}", f"Patients {age_range[0]}–{age_range[1]} yr",
                             border_color=ACCENT2), unsafe_allow_html=True)
    with wa2:
        st.markdown(kpi_html(f"{dw['stroke'].sum():,}", "Stroke Cases in Window",
                             border_color=ACCENT3), unsafe_allow_html=True)
    with wa3:
        st.markdown(kpi_html(f"{dw['stroke'].mean():.1%}", "Stroke Rate in Window",
                             border_color=ACCENT4), unsafe_allow_html=True)

    fig_age = go.Figure()
    fig_age.add_trace(go.Bar(x=daf["age"], y=daf["stroke_prob"], name="Raw Probability",
        marker_color=ACCENT3, opacity=0.35, yaxis="y"))
    fig_age.add_trace(go.Scatter(x=daf["age"], y=daf["smooth"], name="Smoothed (5-yr)",
        mode="lines", line=dict(color=ACCENT4, width=3), yaxis="y"))
    fig_age.add_trace(go.Bar(x=daf["age"], y=daf["total"], name="Patient Count",
        marker_color=ACCENT2, opacity=0.15, yaxis="y2"))
    fig_age.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title=dict(text="Stroke Probability by Age",
                   font=dict(family="Syne, sans-serif", size=16, color=TEXT)),
        xaxis=dict(title="Age (years)", gridcolor=BORDER,
                   zerolinecolor=BORDER, color=TEXT_MUTED, linecolor=BORDER),
        yaxis=dict(title="Stroke Probability", tickformat=".0%",
                   gridcolor=BORDER, zerolinecolor=BORDER,
                   color=TEXT_MUTED, linecolor=BORDER),
        yaxis2=dict(title="Patient Count", overlaying="y", side="right",
                    showgrid=False, color=TEXT_MUTED),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.18, bgcolor=SURFACE2,
                    bordercolor=BORDER, borderwidth=1, font_color=TEXT),
        margin=dict(t=50, b=70, l=40, r=70),
        colorway=SEQ_COLORS,
    )
    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("""<div class="insight-box">
        <strong>Key finding:</strong> Stroke probability increases sharply after age 55.
        Patients over 70 show rates approaching <strong>15–20%</strong>, while those under 40
        exhibit near-zero incidence. Age is the strongest univariate predictor.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pearson & Cramér's V
    cs1, cs2 = st.columns(2)
    with cs1:
        st.markdown('<div class="section-title">Pearson Correlation</div>', unsafe_allow_html=True)
        num_cols = ["age","hypertension","heart_disease","avg_glucose_level","bmi","stroke"]
        corr = df[num_cols].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,ACCENT2],[0.5,"#FFFFFF"],[1,ACCENT3]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values,2), texttemplate="%{text}", textfont_size=11,
            colorbar=dict(tickfont_color=TEXT_MUTED),
        ))
        apply_theme(fig_h, title="Pearson Correlation Heatmap", height=420)
        fig_h.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_h, use_container_width=True)

    with cs2:
        st.markdown("<div class=\"section-title\">Cramér's V Association</div>", unsafe_allow_html=True)
        cat_c = ["gender","ever_married","work_type","Residence_type","smoking_status","stroke"]
        cv_mat = pd.DataFrame(index=cat_c, columns=cat_c, dtype=float)
        for c1 in cat_c:
            for c2 in cat_c:
                cv_mat.loc[c1,c2] = 1.0 if c1==c2 else cramers_v(df[c1].astype(str), df[c2].astype(str))
        fig_cv = go.Figure(go.Heatmap(
            z=cv_mat.values.astype(float), x=cv_mat.columns, y=cv_mat.columns,
            colorscale=[[0,"#FFFFFF"],[0.5,ACCENT2],[1,ACCENT]],
            zmin=0, zmax=1,
            text=np.round(cv_mat.values.astype(float),2), texttemplate="%{text}", textfont_size=11,
            colorbar=dict(tickfont_color=TEXT_MUTED),
        ))
        apply_theme(fig_cv, title="Cramér's V Heatmap (Categorical)", height=420)
        fig_cv.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Scatter matrix
    st.markdown('<div class="section-title">Variable Relationships — Scatter Matrix</div>',
                unsafe_allow_html=True)
    pair_cols = st.multiselect("Select variables",
        ["age","avg_glucose_level","bmi"], default=["age","avg_glucose_level","bmi"])
    if len(pair_cols) >= 2:
        sdf = df[pair_cols+["stroke"]].sample(min(1500,len(df)), random_state=42)
        fig_p = px.scatter_matrix(sdf, dimensions=pair_cols, color="stroke",
            color_discrete_map={0:ACCENT2, 1:ACCENT3},
            labels={c:c.replace("_"," ").title() for c in pair_cols}, opacity=0.5)
        fig_p.update_traces(diagonal_visible=False, marker_size=3)
        apply_theme(fig_p, title="Pairwise Scatter (sample n=1,500)", height=500)
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Select at least 2 variables.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Glucose slider interaction
    st.markdown('<div class="section-title">Interaction: Age × Glucose Level × Stroke</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Adjust the glucose threshold to observe how the joint risk profile shifts.</div>',
                unsafe_allow_html=True)

    gluc_thresh = st.slider("Glucose threshold (mg/dL)", 50, 300, 125, step=5)
    fig_int = px.scatter(df.sample(min(2000,len(df)), random_state=0),
        x="age", y="avg_glucose_level", color="stroke",
        color_discrete_map={0:ACCENT2, 1:ACCENT3},
        opacity=0.55, labels={"age":"Age","avg_glucose_level":"Avg Glucose","stroke":"Stroke"})
    fig_int.add_hline(y=gluc_thresh,
        line=dict(color=ACCENT4, width=2, dash="dash"),
        annotation_text=f"Threshold: {gluc_thresh} mg/dL",
        annotation_font_color=ACCENT4)
    apply_theme(fig_int, title="Age vs. Glucose — Stroke Incidence (sample n=2,000)")
    st.plotly_chart(fig_int, use_container_width=True)

    age_med = df["age"].median()
    q_labels = ["Low age\nLow glucose","Low age\nHigh glucose",
                "High age\nLow glucose","High age\nHigh glucose"]
    q_rates = [
        df[(df["age"]<age_med)  & (df["avg_glucose_level"]<gluc_thresh)]["stroke"].mean(),
        df[(df["age"]<age_med)  & (df["avg_glucose_level"]>=gluc_thresh)]["stroke"].mean(),
        df[(df["age"]>=age_med) & (df["avg_glucose_level"]<gluc_thresh)]["stroke"].mean(),
        df[(df["age"]>=age_med) & (df["avg_glucose_level"]>=gluc_thresh)]["stroke"].mean(),
    ]
    fig_quad = go.Figure(go.Bar(
        x=q_labels, y=q_rates,
        marker=dict(color=q_rates,
                    colorscale=[[0,ACCENT2],[1,ACCENT3]],
                    showscale=False),
        text=[f"{r:.1%}" for r in q_rates], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    apply_theme(fig_quad, title="Stroke Rate by Age–Glucose Quadrant")
    fig_quad.update_yaxes(title_text="Stroke Rate", tickformat=".1%")
    st.plotly_chart(fig_quad, use_container_width=True)


# ══════════════════════════════════════════════  T
with tab_t:
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:6px;">
      <span class="letter-badge">T</span>
      <span class="section-title" style="margin-bottom:0">Tell — Synthesize & Communicate Findings</span>
    </div>
    <div class="section-subtitle">Structured synthesis of the four analytical questions.</div>
    """, unsafe_allow_html=True)

    for tag, clr, ttl, body in [
        ("Q1", ACCENT2,  "Demographics & Clinical Risk Factors",
         "Age is the strongest predictor — risk increases sharply after 55. Hypertension and heart disease are the most discriminating binary features. Higher average glucose is consistently associated with stroke events."),
        ("Q2", ACCENT,   "Interaction Effects",
         "High age + elevated glucose creates the highest-risk profile, with stroke rates exceeding 15% in the oldest, high-glucose quartile. Ever-married status proxies age-related risk. Residence type and gender show minimal independent contribution."),
        ("Q3", ACCENT4,  "Missing Values",
         "Only BMI is missing (~3.9%, 201 rows). Pattern is non-random (MNAR): missing rows tend to be younger patients with lower comorbidities. KNN imputation (k=5) preserves distributional shape."),
        ("Q4", ACCENT3,  "Class Imbalance",
         "Dataset is severely imbalanced: ~95% no-stroke vs. ~5% stroke (ratio ≈19:1). Downstream models require SMOTE, undersampling, or cost-sensitive learning. AUROC, F1, and precision-recall curves should be prioritized over accuracy."),
    ]:
        cb, cy = st.columns([0.07, 0.93])
        with cb:
            st.markdown(f"""<div style="background:linear-gradient(135deg,{clr},{SURFACE2});
                border-radius:10px;padding:12px 8px;text-align:center;
                font-family:'Space Mono',monospace;font-weight:700;font-size:13px;
                color:{TEXT};margin-top:4px;border:1px solid {BORDER};">{tag}</div>""",
                unsafe_allow_html=True)
        with cy:
            st.markdown(f"""<div class="quest-card" style="border-left:3px solid {clr};margin-top:4px;">
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                            color:{TEXT};margin-bottom:8px;">{ttl}</div>
                <div style="font-size:13px;color:{TEXT_MUTED};line-height:1.7;">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Consolidated Risk Factor Profile</div>', unsafe_allow_html=True)

    baseline = df["stroke"].mean()
    rfs = {
        "Age > 65":            df[df["age"]>65]["stroke"].mean(),
        "Hypertension":        df[df["hypertension"]==1]["stroke"].mean(),
        "Heart Disease":       df[df["heart_disease"]==1]["stroke"].mean(),
        "High Glucose (>140)": df[df["avg_glucose_level"]>140]["stroke"].mean(),
        "BMI Obese (>30)":     df[df["bmi"]>30]["stroke"].mean(),
        "Ever Married":        df[df["ever_married"]=="Yes"]["stroke"].mean(),
        "Formerly Smoked":     df[df["smoking_status"]=="formerly smoked"]["stroke"].mean(),
        "Urban Resident":      df[df["Residence_type"]=="Urban"]["stroke"].mean(),
    }
    rdf = pd.DataFrame({"Factor":list(rfs.keys()),"Stroke Rate":list(rfs.values())})
    rdf["Lift"] = rdf["Stroke Rate"] / baseline
    rdf = rdf.sort_values("Stroke Rate", ascending=True)

    fig_r = go.Figure()
    fig_r.add_vline(x=baseline, line=dict(color=ACCENT4, width=2, dash="dot"),
                    annotation_text=f"Baseline {baseline:.1%}",
                    annotation_font_color=ACCENT4)
    fig_r.add_trace(go.Bar(
        y=rdf["Factor"], x=rdf["Stroke Rate"], orientation="h",
        marker=dict(color=rdf["Stroke Rate"].tolist(),
                    colorscale=[[0,ACCENT2],[1,ACCENT3]],
                    showscale=False),
        text=[f"{r:.1%}  ({l:.1f}x)" for r,l in zip(rdf["Stroke Rate"],rdf["Lift"])],
        textposition="outside", textfont=dict(color=TEXT),
    ))
    apply_theme(fig_r, title="Stroke Rate by Risk Factor vs. Population Baseline",
                height=420, extra=dict(margin=dict(l=160,r=110,t=50,b=40)))
    fig_r.update_xaxes(title_text="Stroke Rate", tickformat=".1%")
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown(f"""<div class="quest-card" style="border-left:3px solid #7C3AED;margin-top:8px;">
      <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                  color:{TEXT};margin-bottom:10px;">📌 Recommended Next Steps</div>
      <div style="font-size:13px;color:{TEXT_MUTED};line-height:1.9;">
        <strong style="color:{TEXT}">1. Address class imbalance</strong> — Apply SMOTE or cost-sensitive classifiers before modeling.<br>
        <strong style="color:{TEXT}">2. Feature engineering</strong> — Create interaction terms (age × hypertension, age × glucose).<br>
        <strong style="color:{TEXT}">3. Model selection</strong> — Gradient Boosting handles imbalance well; tune with AUROC.<br>
        <strong style="color:{TEXT}">4. Fairness audit</strong> — Evaluate model performance across gender and residence subgroups.<br>
        <strong style="color:{TEXT}">5. Clinical validation</strong> — Present probability scores to clinical team for threshold calibration.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;margin-top:40px;padding:24px;
        border-top:1px solid {BORDER};
        font-family:'Space Mono',monospace;font-size:11px;color:{TEXT_MUTED};">
        Stroke Prediction Dashboard &nbsp;·&nbsp; QUEST Framework &nbsp;·&nbsp; Group 4 &nbsp;·&nbsp;
        Built with Streamlit &amp; Plotly
    </div>""", unsafe_allow_html=True)
