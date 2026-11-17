import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import time
from pathlib import Path

import dl_data_generator as dlg
from dl_inference import load_surrogate

# --- 1. PAGE CONFIGURATION AND VISUAL STYLES ---
st.set_page_config(
    page_title="AGD-Sentinel | Geotechnics",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Injection for UI styling
st.markdown("""
    <style>
    /* 1. Global dark background and base text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* HEADER CORRECTION */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    
    /* 2. Force white text */
    h1, h2, h3, h4, h5, h6, p, li, span, label {
        color: #FAFAFA !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* 3. IMPROVED NUMERIC INPUTS */
    input[type="number"] {
        color: #FFFFFF !important;
        background-color: #262730 !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1px solid #464B5C !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }
    input {
        caret-color: #FF4B4B !important;
    }
    input::-webkit-outer-spin-button,
    input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    /* 4. NATIVE TOOLTIPS CORRECTION */
    div[data-testid="stTooltipContent"] {
        background-color: #262730 !important; 
        color: #FFFFFF !important; 
        border: 1px solid #464B5C !important;
    }

    /* 5. Metric Cards Style */
    [data-testid="stMetricLabel"] {
        color: #A3A8B8 !important; 
        font-size: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; 
        font-size: 1.8rem !important;
    }

    /* 6. CUSTOM LED INDICATOR FOR ML METRICS */
    .led-container {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        position: relative;
        margin-bottom: 10px;
    }
    .led-label {
        color: #A3A8B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .led-value {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }
    .led-dot {
        width: 15px;
        height: 15px;
        border-radius: 50%;
        display: inline-block;
        cursor: help;
        position: relative;
    }
    .led-dot .tooltip-box {
        visibility: hidden;
        width: 200px;
        background-color: #FFFFFF;
        color: #000000 !important;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        font-size: 0.8rem;
        font-weight: normal;
        border: 2px solid #ccc;
    }
    .led-dot .tooltip-box strong {
        color: #000000 !important;
    }
    .led-dot:hover .tooltip-box {
        visibility: visible;
        opacity: 1;
    }

    /* 7. Sliders and Widgets */
    .stSlider label {
        color: #FAFAFA !important;
        font-weight: 500 !important;
    }
    
    /* 8. Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730 !important;
    }
    
    /* 9. Machine Learning Button */
    div.stButton > button:first-child {
        background-color: #FF4B4B; 
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 0.75em 1.5em;
        border-radius: 10px;
        border: none;
        width: 100%;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF2B2B;
        box-shadow: 0px 6px 20px rgba(255, 75, 75, 0.6);
        transform: translateY(-2px);
        border: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AUXILIARY FUNCTIONS ---

def get_metric_status(metric_name, value):
    """Returns color and status message for the ML evaluation LED indicator."""
    value = round(value, 3) 
    if metric_name == "R2":
        if value >= 0.90:
            return "#00CC96", "EXCELLENT", "The model perfectly understands the terrain's history."
        elif value >= 0.75:
            return "#FFA500", "ACCEPTABLE", "Reasonable fit, but data is scattered."
        else:
            return "#FF4B4B", "POOR", "Input data contradicts the historical trend."
    elif metric_name == "RMSE":
        if value <= 0.05:
            return "#00CC96", "EXCELLENT", "Minimal error margin (< 0.05)."
        elif value <= 0.10:
            return "#FFA500", "ACCEPTABLE", "Moderate error margin."
        else:
            return "#FF4B4B", "HIGH", "High uncertainty in prediction."
    elif metric_name == "MAE":
        if value <= 0.04:
            return "#00CC96", "EXCELLENT", "Insignificant average deviation."
        elif value <= 0.08:
            return "#FFA500", "ACCEPTABLE", "Normal average deviation."
        else:
            return "#FF4B4B", "HIGH", "The model fails to connect the data points well."
    return "#FFFFFF", "N/A", "No data"

def render_led_metric(label, value, metric_type):
    """Generates the HTML string for the custom LED metric indicator."""
    color, status, desc = get_metric_status(metric_type, value)
    led_style = f"background-color: {color}; box-shadow: 0 0 10px {color};"
    
    html = f"""
    <div class="led-container">
        <div class="led-label">{label}</div>
        <div class="led-value">
            {value:.3f}
            <div class="led-dot" style="{led_style}">
                <div class="tooltip-box">
                    <strong>DIAGNOSIS: {status}</strong><br>
                    <hr style="margin: 5px 0; border-color: #ccc;">
                    {desc}
                </div>
            </div>
        </div>
    </div>
    """
    return html

def calcular_parametros_estimados(spt_n, vs_velocity, resistivity):
    """Simulates field data correlation to geotechnical parameters (AUTOMATIC MODE)."""
    phi = 18 + (spt_n * 0.4) 
    phi = min(max(phi, 15), 35) 
    
    cohesion = (vs_velocity * 0.08) - 2
    cohesion = min(max(cohesion, 5), 40)
    
    gamma = 16 + (spt_n * 0.1)
    gamma = min(max(gamma, 16), 21)
    
    if resistivity < 100:
        factor = 0.85 + (resistivity / 1000)
        cohesion *= factor
        phi *= factor
    
    return round(phi, 2), round(cohesion, 2), round(gamma, 2)

def factor_seguridad_talud_infinito(c, phi, gamma, alpha_deg, hw_ratio, k_sismo):
    """Calculates the pseudo-static Factor of Safety (FS) using the Infinite Slope Limit Equilibrium Model."""
    alpha = np.radians(alpha_deg)
    phi_rad = np.radians(phi)
    
    gamma_w = 9.81  # Unit weight of water (kN/m³)
    z = 5.0  # Assumed depth of the failure surface (m)
    hw = z * hw_ratio  # Height of the groundwater table above the failure surface
    
    # Resisting forces (Shear strength based on Mohr-Coulomb criterion)
    resistencia = c + (gamma * z - gamma_w * hw) * (np.cos(alpha)**2) * np.tan(phi_rad)
    
    # Driving forces (Gravity and pseudo-static seismic loads)
    actuante = (gamma * z * np.sin(alpha) * np.cos(alpha)) + (k_sismo * gamma * z * np.cos(alpha))
    
    # Prevent division by zero for completely flat terrains
    if actuante <= 0.1: return 10.0 
    
    return resistencia / actuante

# --- 3. USER INTERFACE (SIDEBAR WITH ROBUST SYNCHRONIZED INPUTS) ---

def dual_input(label, min_val, max_val, default, key_base, step=1.0, help_text=None):
    """Creates a slider and numeric input that are bidirectionally synchronized via Streamlit Session State to optimize memory handling."""
    slider_key = f"{key_base}_slider"
    input_key = f"{key_base}_input"
    
    # Initialize Session State variables to prevent recursive callback errors
    if key_base not in st.session_state:
        st.session_state[key_base] = default
        if slider_key not in st.session_state: st.session_state[slider_key] = default
        if input_key not in st.session_state: st.session_state[input_key] = default

    def update_from_slider():
        val = st.session_state[slider_key]
        st.session_state[key_base] = val
        st.session_state[input_key] = val

    def update_from_input():
        val = st.session_state[input_key]
        st.session_state[key_base] = val
        st.session_state[slider_key] = val

    st.markdown(f"<label style='font-size: 1rem; font-weight: 500;'>{label}</label>", unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)
        
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.slider(
            label="hidden",
            min_value=float(min_val), max_value=float(max_val), step=float(step),
            key=slider_key,
            on_change=update_from_slider,
            label_visibility="collapsed"
        )
    with col2:
        st.number_input(
            label="hidden",
            min_value=float(min_val), max_value=float(max_val), step=float(step),
            key=input_key,
            on_change=update_from_input,
            label_visibility="collapsed"
        )
    return st.session_state[key_base]

with st.sidebar:
    st.title("🎛️ Field Panel")
    st.markdown("---")
    
    # ---------------- SECTION 1: SENSORS ----------------
    st.markdown("### 1. 📡 Sensor Data", help="Field measured parameters (Input for Estimation).")
    
    input_spt = dual_input(
        "SPT (N-blows)", 2, 50, 15, "spt", step=1.0,
        help_text="< 10: Loose soil | > 30: Dense soil"
    )
    input_vs = dual_input(
        "Shear Wave Velocity Vs (m/s)", 100, 500, 218, "vs", step=1.0,
        help_text="Soil stiffness. Critical < 200 m/s"
    )
    input_resistivity = dual_input(
        "Resistivity (Ohm-m)", 10, 500, 200, "res", step=10.0,
        help_text="Low values = High saturation"
    )
    
    # Automatic geotechnical parameter estimation running in the background
    est_phi, est_c, est_gamma = calcular_parametros_estimados(input_spt, input_vs, input_resistivity)

    # ---------------- SECTION 2: BOUNDARY CONDITIONS ----------------
    st.markdown("---")
    st.markdown("### 2. ⛈️ Boundary Conditions")
    
    input_alpha = dual_input(
        "Slope Angle (°)", 5.0, 90.0, 25.0, "alpha", step=1.0
    )
    input_rain_pct = dual_input(
        "Saturation (%)", 0.0, 100.0, 30.0, "rain", step=5.0
    )
    input_seismic = st.number_input(
        "Seismic (k Coeff.)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.00, 
        step=0.01,
        format="%.2f",
        help="Horizontal seismic acceleration."
    )

    # ---------------- SECTION 3: LABORATORY DATA ----------------
    st.markdown("---")
    st.markdown("### 3. 🧪 Laboratory Data")
    
    # Main toggle switch for manual physical properties override
    manual_mode = st.toggle("Use Laboratory Data?", value=False, 
                            help="Activate this if you have exact values for Cohesion, Friction, and Unit Weight. The system will ignore sensors and use these data.")

    if manual_mode:
        st.success("Manual Mode ACTIVATED: Enter values below.")
        final_c = st.number_input("Cohesion (c) [kPa]", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f")
        final_phi = st.number_input("Friction Angle (ϕ) [°]", min_value=0.0, max_value=45.0, value=25.0, step=0.1, format="%.1f")
        final_gamma = st.number_input("Unit Weight (γ) [kN/m³]", min_value=10.0, max_value=25.0, value=18.0, step=0.1, format="%.1f")
        
        source_label = "LABORATORY (REAL)"
        source_color = "#00CC96" 
    else:
        st.caption("Manual Mode DEACTIVATED: Using sensor estimation.")
        # Assign estimated parameters to the final variables for physical modeling
        final_c = est_c
        final_phi = est_phi
        final_gamma = est_gamma
        
        source_label = "ESTIMATION (CORRELATION)"
        source_color = "#FFA500" 

# --- 3b. NAVIGATION: v1.0 monitor (unchanged) and deep-surrogate views ---

tab_v1, tab_dl, tab_mc = st.tabs(["Monitor v1.0", "Surrogate LSTM", "Validación vs MC"])

with tab_v1:
    # --- 4. CORE COMPUTATIONAL ENGINE ---

    input_hw_ratio = input_rain_pct / 100.0

    # Compute the current Factor of Safety using deterministic Limit Equilibrium calculations
    fs_actual = factor_seguridad_talud_infinito(final_c, final_phi, final_gamma, input_alpha, input_hw_ratio, input_seismic)

    # Monte Carlo Simulation: Introducing Gaussian noise to physical parameters to assess failure probability
    n_sim = 1000
    np.random.seed(42)
    fs_sims = []
    for _ in range(n_sim):
        c_s = max(final_c + np.random.normal(0, 2), 0.1)
        p_s = max(final_phi + np.random.normal(0, 1.5), 5)
        fs_sims.append(factor_seguridad_talud_infinito(c_s, p_s, final_gamma, input_alpha, input_hw_ratio, input_seismic))
    prob_falla = np.mean(np.array(fs_sims) < 1.1) * 100

    # --- 5. REAL-TIME DASHBOARD VISUALIZATION ---

    st.title("Geotechnical Degradation Predictive System (AGD)")
    st.markdown(f"Real-time monitoring sector **Cortinas, Toledo**. Scale 1:5000.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Current State")
        if fs_actual < 1.1:
            color, icon, txt = "#FF4B4B", "🚨", "CRITICAL"
        elif fs_actual < 1.5:
            color, icon, txt = "#FFA500", "⚠️", "WARNING"
        else:
            color, icon, txt = "#00CC96", "✅", "STABLE"
        
        st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
        <h1 style="margin:0; font-size: 3.5rem;">{fs_actual:.2f}</h1>
        <p style="margin:0; font-size: 1.2rem; font-weight: bold;">{icon} {txt}</p>
    </div>
    """, unsafe_allow_html=True)
    
        # Render physical parameters emphasizing the active data source
        st.markdown(f"### Soil Parameters")
        st.markdown(f"<small style='color: {source_color}; font-weight: bold;'>SOURCE: {source_label}</small>", unsafe_allow_html=True)
    
        c1, c2 = st.columns(2)
        c1.metric("Cohesion (c)", f"{final_c} kPa")
        c1.metric("Unit Weight (γ)", f"{final_gamma} kN/m³")
        c2.metric("Friction Angle (ϕ)", f"{final_phi} °")
        c2.metric("Failure Prob.", f"{prob_falla:.1f}%")

    with col2:
        st.subheader("Probability Distribution")
        fig = px.histogram(x=fs_sims, nbins=40, color_discrete_sequence=['#636EFA'])
        fig.add_vline(x=1.1, line_dash="dash", line_color="red")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                          xaxis_title="Factor of Safety", yaxis_title="Frequency")
        st.plotly_chart(fig, width="stretch")

    # --- 6. MACHINE LEARNING PREDICTIVE ENGINE ---

    st.markdown("---")
    st.subheader("🤖 Future Degradation and Recovery Projection (AI)")

    btn_col, graph_col = st.columns([1, 3])

    with btn_col:
        st.write("")
        if st.button("EXECUTE AGD\nPREDICTIVE MODEL"):
        
            # Execute Polynomial Regression Pipeline
            # Historical temporal data (Features)
            years_hist = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1,1)
            # Historical Factor of Safety data (Target)
            fs_hist = np.array([1.25, 1.20, 1.10, 1.00, 0.90, 1.05, 1.15])
        
            # Append current physical state to training matrix
            X_train = np.vstack([years_hist, [[2026]]])
            y_train = np.append(fs_hist, fs_actual)
        
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X_train)
            model = LinearRegression().fit(X_poly, y_train)
            y_pred = model.predict(X_poly)
        
            # Compute standard statistical evaluation metrics
            r2 = r2_score(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            mae = mean_absolute_error(y_train, y_pred)
        
            st.markdown("##### Fit Quality")
            st.markdown(render_led_metric("R² (Accuracy)", r2, "R2"), unsafe_allow_html=True)
            st.markdown(render_led_metric("RMSE (Error)", rmse, "RMSE"), unsafe_allow_html=True)
            st.markdown(render_led_metric("MAE (Absolute)", mae, "MAE"), unsafe_allow_html=True)

            with graph_col:
                years_fut = np.arange(2017, 2031).reshape(-1,1)
                fs_fut = model.predict(poly.transform(years_fut))
            
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=years_hist.flatten(), y=fs_hist, mode='markers', name='Historical'))
                fig_ml.add_trace(go.Scatter(x=[2026], y=[fs_actual], mode='markers', marker_symbol='star', marker_size=15, name='Current'))
                fig_ml.add_trace(go.Scatter(x=years_fut.flatten(), y=fs_fut, mode='lines', name='AI Trend'))
                fig_ml.add_hline(y=1.1, line_color="red", line_dash="dash")
                fig_ml.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                                    xaxis_title="Year", yaxis_title="Factor of Safety", height=450)
                st.plotly_chart(fig_ml, width="stretch")
            
                if fs_actual < 1.1:
                    # Case 1: Immediate and persistent critical instability detected
                    st.error(f"⚠️ PREDICTION: Instability CURRENTLY DETECTED (2026) and persistent in the future.")
                else:
                    # Case 2: Project intersection with the critical failure threshold (FS < 1.1)
                    future_risk = years_fut.flatten()[np.where((years_fut.flatten() > 2026) & (fs_fut < 1.1))]
                    if len(future_risk) > 0:
                        st.error(f"⚠️ PREDICTION: Possible instability detected starting from the year **{future_risk[0]}**.")
                    else:
                        st.success("✅ PREDICTION: Stable trend estimated for the next 5 years.")
        else:
            with graph_col:
                st.info("👈 Execute the model to view the projection.")

# --- 7. DEEP SURROGATE (LSTM) AND VALIDATION AGAINST CLASSIC MONTE CARLO ---

DL_COLORS = {"lstm": "#56B4E9", "band": "rgba(230,159,0,0.30)", "oracle": "#FAFAFA", "mc": "rgba(0,158,115,0.25)",
             "mc_line": "#009E73", "camp": "#D55E00", "crit": "#FF4B4B"}
DL_RAIN_OPTIONS = ["Cortinas historical (2021 and 2024 events)", "Seasonal climatology", "No forcing (v1.0 linear law)"]


@st.cache_resource
def dl_load_assets():
    """Loads NumPy weights, frozen calibration and benchmark metrics once per session."""
    surrogate = load_surrogate()
    ref = json.load(open(dlg.REF_FILE)) if dlg.REF_FILE.exists() else None
    metrics_file = Path(__file__).resolve().parent / "dl_metrics.json"
    metrics = json.load(open(metrics_file)) if metrics_file.exists() else {}
    return surrogate, ref, metrics


def dl_rainfall(option, factor):
    if option == DL_RAIN_OPTIONS[0]:
        return dlg.climatology_rain(dlg.HIST_EVENTS, scale=factor)
    if option == DL_RAIN_OPTIONS[1]:
        return dlg.climatology_rain((), scale=factor)
    return np.ones(dlg.T)


def dl_year_label(tc, cross):
    return f"{dlg.YEAR0 + tc:.2f}" if cross else f"> {dlg.YEAR0 + dlg.TH:.0f}"


def dl_kpi_card(label, value, caption):
    return f"""
    <div class="led-container">
        <div class="led-label">{label}</div>
        <div class="led-value">{value}</div>
        <div style="color:#A3A8B8; font-size:0.8rem; margin-top:4px;">{caption}</div>
    </div>
    """


def dl_figure_layout(fig, height=480):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis_title="Year", yaxis_title="Factor of Safety FS(t)", height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified")
    fig.add_hline(y=dlg.FS_CRIT, line_color=DL_COLORS["crit"], line_dash="dash",
                  annotation_text="FS = 1.1", annotation_position="bottom right")
    fig.add_hline(y=1.0, line_color="#8C8C8C", line_dash="dot")
    return fig


def dl_sample_parameters(N, c0, phi0, gamma, seed=42):
    """master_run scheme: bivariate normal (c'0, phi'0) with rho=-0.62, gamma ~ N, z ~ U(8,16), beta ~ U(26,34)."""
    rng = np.random.default_rng(seed)
    cov = [[dlg.SC**2, dlg.RHO*dlg.SC*dlg.SP], [dlg.RHO*dlg.SC*dlg.SP, dlg.SP**2]]
    s = rng.multivariate_normal([c0, phi0], cov, N)
    return (np.clip(s[:, 0], 10, 30), np.clip(s[:, 1], 20, 40), np.clip(rng.normal(gamma, 0.5, N), 18.5, 21.0),
            rng.uniform(8, 16, N), rng.uniform(26, 34, N))


surrogate, dl_ref, dl_metrics = dl_load_assets()
dl_const = dl_ref["constants"] if dl_ref else dlg.load_constants()

with tab_dl:
    st.subheader("Deep Surrogate: LSTM projection of FS(t) with AGD memory")
    st.markdown(
        "Sequence-to-sequence LSTM trained on 20 000 synthetic trajectories of the calibrated physical oracle "
        "(v1.0 infinite-slope engine + rainfall-driven pore pressure, damage clock and post-failure recovery). "
        "Uncertainty bands come from **MC Dropout** (model uncertainty). Controls are bounded to the training ranges.")
    if surrogate is None:
        st.warning("Surrogate weights not found in `models/`. Showing the physical oracle only "
                   "(run `python dl_train_evaluate.py` to train and export the LSTM).")

    ca, cb, cc, cd = st.columns(4)
    with ca:
        st.markdown("**Initial strength**")
        dl_c0 = st.slider("c'0 [kPa]", 12.0, 25.0, dlg.C0, 0.1, key="dl_c0")
        dl_phi0 = st.slider("φ'0 [°]", 24.0, 37.0, dlg.PHI0, 0.1, key="dl_phi0")
        dl_gamma = st.slider("γ [kN/m³]", 18.5, 21.0, dlg.GAMMA, 0.01, key="dl_gamma")
    with cb:
        st.markdown("**Geometry and degradation rates**")
        dl_beta = st.slider("β [°]", 26.0, 34.0, dlg.BETA, 0.5, key="dl_beta")
        dl_z = st.slider("z [m]", 8.0, 16.0, dlg.Z, 0.5, key="dl_z")
        dl_alpha = st.slider("α [kPa/yr]", 0.80, 2.40, dlg.ALPHA, 0.01, key="dl_alpha")
        dl_delta = st.slider("δ [°/yr]", 1.1875, 3.5625, dlg.DELTA, 0.0125, key="dl_delta", format="%.3f")
    with cc:
        st.markdown("**Hydro-climatic and seismic forcing**")
        dl_rain = st.selectbox("Rainfall scenario", DL_RAIN_OPTIONS, key="dl_rain")
        dl_rain_factor = st.slider("Annual rainfall factor", 0.6, 1.6, 1.0, 0.05, key="dl_rain_factor",
                                   disabled=dl_rain == DL_RAIN_OPTIONS[2], help="Multiplies the seasonal climatology (1.0 = 3000 mm/yr).")
        dl_seismic = st.toggle("Seismic scenario k = 0.294 g", value=False, key="dl_kh_0294",
                               help="Pseudo-static scenario; not the nominal v1.0 condition (static, k = 0).")
        dl_kh = 0.294 if dl_seismic else st.slider("Seismic coefficient kh", 0.0, 0.30, 0.0, 0.01, key="dl_kh")
    with cd:
        st.markdown("**Regime and uncertainty**")
        dl_rec = st.toggle("Post-failure recovery (AGD)", value=True, key="dl_rec")
        dl_tf = st.slider("Failure date t_f [year]", 2019.5, 2024.0, 2021.0, 1 / 12, key="dl_tf", format="%.2f", disabled=not dl_rec)
        dl_zeta = st.slider("Post-failure depth ratio ζ = z_e/z", 0.30, 0.55, round(dl_const["zeta"], 3), 0.005,
                            key="dl_zeta", disabled=not dl_rec)
        dl_kappa = st.slider("Recovery-rate factor κ_r", 0.5, 1.5, 1.0, 0.05, key="dl_kappa", disabled=not dl_rec)
        dl_K = st.slider("MC Dropout passes K", 10, 100, 50, 10, key="dl_K")

    dl_p = dl_rainfall(dl_rain, dl_rain_factor)
    dl_pars = (dl_c0, dl_phi0, dl_gamma, dl_z, dl_beta, dl_alpha, dl_delta, dl_kh)
    dl_kf = int(round((dl_tf - dlg.YEAR0) / dlg.DT)) if dl_rec else -1
    if dl_kf >= 0:
        fs_pre = dlg.run_oracle(*dl_pars, dl_p, None, dl_zeta, dl_kappa, dl_const)["fs"][0, dl_kf]
        if fs_pre >= 1.0:
            st.info(f"FS before t_f = {fs_pre:.2f} ≥ 1: no failure occurs at t_f, so the trajectory stays in the "
                    "continuous-deterioration regime (same rule as the training data).")
            dl_kf = -1
    dl_or = dlg.run_oracle(*dl_pars, dl_p, dl_kf if dl_kf >= 0 else None, dl_zeta, dl_kappa, dl_const)
    fs_or = dl_or["fs"][0]
    years = dlg.YEAR0 + dlg.t
    tc_or, cr_or = dlg.crossing_time(fs_or[None])

    if surrogate is not None:
        X_dl = dlg.build_features(dl_p[None], dl_or["rec"], dl_or["z_e"], dl_c0, dl_phi0, dl_gamma, dl_beta,
                                  dl_alpha, dl_delta, dl_kh, dl_kappa)
        t0 = time.perf_counter()
        fs_det = surrogate.predict(X_dl)[0]
        t_det = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        mc = surrogate.mc_dropout(X_dl, K=dl_K, seed=0)
        t_mcd = (time.perf_counter() - t0) * 1e3
        mu, lo, hi = mc["mean"][0], mc["lo2s"][0], mc["hi2s"][0]
        (tc_mu, tc_lo, tc_hi), (cr_mu, cr_lo, cr_hi) = dlg.crossing_time(np.vstack([mu, lo, hi]))
        rel_err = 100 * np.mean(np.abs(mu - fs_or) / fs_or)
        t_mc_ref = dl_metrics.get("benchmark", {}).get("mc_classic_N1000", {}).get("median_ms")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(dl_kpi_card("FS final (2028.9)", f"{mu[-1]:.3f}", f"± {2 * mc['std'][0, -1]:.3f} (2σ) · oracle {fs_or[-1]:.3f}"), unsafe_allow_html=True)
        band = f"band {dl_year_label(tc_lo, cr_lo)} – {dl_year_label(tc_hi, cr_hi)}"
        k2.markdown(dl_kpi_card("Crossing FS = 1.1", dl_year_label(tc_mu, cr_mu), f"{band} · oracle {dl_year_label(tc_or[0], cr_or[0])}"), unsafe_allow_html=True)
        k3.markdown(dl_kpi_card("Error vs physics", f"{rel_err:.2f} %", f"mean relative |FS_LSTM − FS_oracle|, max {np.abs(mu - fs_or).max():.3f}"), unsafe_allow_html=True)
        k4.markdown(dl_kpi_card("Inference", f"{t_mcd:.1f} ms", f"MC Dropout K={dl_K} · single curve {t_det:.1f} ms"), unsafe_allow_html=True)
        speed = f"{t_mc_ref / t_mcd:.1f}×" if t_mc_ref else "n/a"
        k5.markdown(dl_kpi_card("Speedup vs MC N=1000", speed, f"classic MC {t_mc_ref:.1f} ms (benchmark)" if t_mc_ref else "run dl_train_evaluate.py"), unsafe_allow_html=True)
    else:
        mu, lo, hi = fs_or, None, None
        tc_mu, cr_mu = tc_or[0], cr_or[0]
        st.metric("Crossing FS = 1.1 (oracle)", dl_year_label(tc_mu, cr_mu))

    dl_overlay = st.checkbox("Overlay classic Monte Carlo P5–P95 on the oracle (N = 1000, master_run scheme)", value=False, key="dl_overlay")
    fig_dl = go.Figure()
    if dl_overlay:
        c0s, p0s, gs, zs, bs = dl_sample_parameters(1000, dl_c0, dl_phi0, dl_gamma)
        fs_mc = dlg.run_oracle(c0s, p0s, gs, zs, bs, dl_alpha, dl_delta, dl_kh, dl_p,
                               dl_kf if dl_kf >= 0 else None, dl_zeta, dl_kappa, dl_const)["fs"]
        p5, p95 = np.percentile(fs_mc, [5, 95], axis=0)
        fig_dl.add_trace(go.Scatter(x=years, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_dl.add_trace(go.Scatter(x=years, y=p5, fill="tonexty", fillcolor=DL_COLORS["mc"], line=dict(width=0),
                                    name="Classic MC P5–P95 (oracle)"))
    if lo is not None:
        fig_dl.add_trace(go.Scatter(x=years, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_dl.add_trace(go.Scatter(x=years, y=lo, fill="tonexty", fillcolor=DL_COLORS["band"], line=dict(width=0),
                                    name="LSTM MC Dropout μ ± 2σ"))
    fig_dl.add_trace(go.Scatter(x=years, y=fs_or, mode="lines", line=dict(color=DL_COLORS["oracle"], width=2, dash="dash"),
                                name="Physical oracle"))
    if surrogate is not None:
        fig_dl.add_trace(go.Scatter(x=years, y=mu, mode="lines", line=dict(color=DL_COLORS["lstm"], width=2.5), name="LSTM surrogate (μ)"))
    if dl_ref and dl_rain == DL_RAIN_OPTIONS[0]:
        camp = dl_ref["campaigns"]
        fig_dl.add_trace(go.Scatter(x=[int(y) for y in camp], y=[camp[y]["fs_spencer"] for y in camp], mode="markers",
                                    marker=dict(symbol="square", size=10, color=DL_COLORS["camp"]), name="Campaign FS (Spencer)"))
    if cr_mu:
        fig_dl.add_vline(x=dlg.YEAR0 + tc_mu, line_color=DL_COLORS["crit"], line_width=1,
                         annotation_text=f"FS = 1.1 at {dlg.YEAR0 + tc_mu:.2f}", annotation_position="top right")
    if dl_kf >= 0:
        fig_dl.add_vrect(x0=dlg.YEAR0 + dl_kf * dlg.DT, x1=dlg.YEAR0 + dlg.TH, fillcolor="#00CC96", opacity=0.05, line_width=0,
                         annotation_text="recovery regime", annotation_position="top left")
    st.plotly_chart(dl_figure_layout(fig_dl), width="stretch")
    st.caption("MC Dropout bands describe the surrogate's model uncertainty; parametric uncertainty of the soil is "
               "propagated in the 'Validación vs MC' tab. Physical oracle and surrogate share the same forcing and regime.")
    if dl_rain == DL_RAIN_OPTIONS[0]:
        st.caption("Before the failure the oracle is essentially the v1.0 linear law (identical with forcing off): FS < 1 "
                   "between 2018.9 and 2021.0 is an inherited conservative bias, not a failure prediction. The failure date "
                   "t_f is an input; the 2021 minimum (0.906) is calibrated, the 2024 minimum (1.062 > 1) is not.")

    if dl_metrics.get("models", {}).get("lstm"):
        tst = dl_metrics["models"]["lstm"]["test"]
        st.markdown("##### Surrogate test-set quality (3 000 unseen trajectories)")
        m1, m2, m3 = st.columns(3)
        m1.markdown(render_led_metric("R² (Accuracy)", tst["R2"], "R2"), unsafe_allow_html=True)
        m2.markdown(render_led_metric("RMSE (Error)", tst["RMSE"], "RMSE"), unsafe_allow_html=True)
        m3.markdown(render_led_metric("MAE (Absolute)", tst["MAE"], "MAE"), unsafe_allow_html=True)

with tab_mc:
    st.subheader("Validation against classic Monte Carlo (v1.0 scheme)")
    st.markdown(
        "Parametric uncertainty propagated with the master_run scheme around the scenario of the "
        "'Surrogate LSTM' tab: (c'0, φ'0) bivariate normal (σ = 3.0 kPa / 2.5°, ρ = −0.62), γ ~ N(γ, 0.5), "
        "z ~ U(8, 16) m, β ~ U(26, 34)°; same forcing, rates and regime. The same samples are evaluated by "
        "the physical oracle and by the LSTM in batch (pure NumPy).")
    if surrogate is None:
        st.warning("Surrogate weights not found: validation requires the trained LSTM.")
    else:
        dl_N = st.radio("Number of realisations N", [1000, 10000], horizontal=True, key="dl_mc_n")
        dl_sig = (dl_N,) + tuple(round(float(v), 4) for v in dl_pars) + (dl_rain, dl_rain_factor, dl_kf, dl_zeta, dl_kappa)
        if st.button("RUN MONTE CARLO\nVALIDATION", key="dl_mc_run"):
            with st.spinner("Running the classic Monte Carlo and the surrogate on the same samples..."):
                c0s, p0s, gs, zs, bs = dl_sample_parameters(dl_N, dl_c0, dl_phi0, dl_gamma)
                kf_arg = dl_kf if dl_kf >= 0 else None
                t0 = time.perf_counter()
                res_or = dlg.run_oracle(c0s, p0s, gs, zs, bs, dl_alpha, dl_delta, dl_kh, dl_p, kf_arg, dl_zeta, dl_kappa, dl_const)
                q_or = np.percentile(res_or["fs"], [5, 50, 95], axis=0)
                tc_mc_or, _ = dlg.crossing_time(res_or["fs"])
                t_or = (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter()
                X_mc = dlg.build_features(np.repeat(dl_p[None], dl_N, 0), res_or["rec"], res_or["z_e"], c0s, p0s, gs, bs,
                                          dl_alpha, dl_delta, dl_kh, dl_kappa)
                fs_sur = surrogate.predict(X_mc)
                q_sur = np.percentile(fs_sur, [5, 50, 95], axis=0)
                tc_mc_sur, _ = dlg.crossing_time(fs_sur)
                t_sur = (time.perf_counter() - t0) * 1e3
                st.session_state["dl_mc_result"] = {"sig": dl_sig, "q_or": q_or, "q_sur": q_sur, "t_or": t_or, "t_sur": t_sur,
                                                    "tc_or": tc_mc_or, "tc_sur": tc_mc_sur,
                                                    "rmse_all": float(np.sqrt(np.mean((fs_sur - res_or["fs"]) ** 2)))}
        res = st.session_state.get("dl_mc_result")
        if res is None:
            st.info("Press the button to run the classic Monte Carlo and the surrogate on the same samples.")
        else:
            if res["sig"] != dl_sig:
                st.warning("The scenario changed since the last run: results below belong to the previous scenario.")
            q_or, q_sur = res["q_or"], res["q_sur"]
            v1, v2, v3, v4 = st.columns(4)
            v1.markdown(dl_kpi_card("Classic MC (oracle)", f"{res['t_or']:.0f} ms", f"N = {res['sig'][0]} · full pipeline"), unsafe_allow_html=True)
            v2.markdown(dl_kpi_card("LSTM batch", f"{res['t_sur']:.0f} ms", "features + inference + percentiles"), unsafe_allow_html=True)
            v3.markdown(dl_kpi_card("Speedup S_C", f"{res['t_or'] / res['t_sur']:.2f}×", "t_MC / t_LSTM (same samples)"), unsafe_allow_html=True)
            v4.markdown(dl_kpi_card("RMSE of P50", f"{np.sqrt(np.mean((q_sur[1] - q_or[1]) ** 2)):.4f}", f"all curves RMSE {res['rmse_all']:.4f}"), unsafe_allow_html=True)
            fig_mc = go.Figure()
            for q, name, fill, line in ((q_or, "Classic MC (oracle)", DL_COLORS["mc"], DL_COLORS["mc_line"]),
                                        (q_sur, "LSTM batch MC", DL_COLORS["band"], DL_COLORS["lstm"])):
                fig_mc.add_trace(go.Scatter(x=years, y=q[2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_mc.add_trace(go.Scatter(x=years, y=q[0], fill="tonexty", fillcolor=fill, line=dict(width=0), name=f"{name} P5–P95"))
                fig_mc.add_trace(go.Scatter(x=years, y=q[1], mode="lines", line=dict(color=line, width=2.5), name=f"{name} P50"))
            st.plotly_chart(dl_figure_layout(fig_mc), width="stretch")
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Histogram(x=dlg.YEAR0 + res["tc_or"], name="Classic MC (oracle)", marker_color=DL_COLORS["mc_line"], opacity=0.6, nbinsx=40))
            fig_tc.add_trace(go.Histogram(x=dlg.YEAR0 + res["tc_sur"], name="LSTM batch MC", marker_color=DL_COLORS["lstm"], opacity=0.6, nbinsx=40))
            fig_tc.update_layout(barmode="overlay", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                                 xaxis_title="Year of first crossing FS = 1.1", yaxis_title="Frequency", height=320)
            st.plotly_chart(fig_tc, width="stretch")
            pc_or, pc_sur = np.percentile(res["tc_or"], [5, 50, 95]), np.percentile(res["tc_sur"], [5, 50, 95])
            st.caption(f"Critical window (P5–P50–P95): oracle {dlg.YEAR0 + pc_or[0]:.2f} – {dlg.YEAR0 + pc_or[1]:.2f} – "
                       f"{dlg.YEAR0 + pc_or[2]:.2f} | LSTM {dlg.YEAR0 + pc_sur[0]:.2f} – {dlg.YEAR0 + pc_sur[1]:.2f} – "
                       f"{dlg.YEAR0 + pc_sur[2]:.2f}. Timings are single live runs; the controlled "
                       "benchmark (5 warm-up + 30 repetitions) is reported in dl_metrics.json.")
