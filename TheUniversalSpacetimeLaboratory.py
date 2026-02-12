import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
import plotly.graph_objects as go
import io
import time

# --- 1. STEALTH CONFIGURATION ---
st.set_page_config(
    page_title="Lorentzian Metric Solver", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED STEALTH CSS (SecAI-Nexus Standard) ---
st.markdown("""
<style>
    /* GLOBAL DARK THEME */
    .stApp { background-color: #050505 !important; font-family: 'Courier New', Courier, monospace !important; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stCaption { color: #00ff41 !important; }
    
    /* REMOVE WHITE ELEMENTS */
    header, footer { visibility: hidden; }
    .stDeployButton { display: none; }
    
    /* SYSTEM HEADER */
    .clock-header {
        font-size: 1rem;
        font-weight: bold;
        text-align: right;
        color: #00ff41;
        margin-bottom: -20px;
        text-shadow: 0 0 5px #00ff41;
    }
    
    /* INPUTS & SELECTS */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input, select, .stSelectbox, .stNumberInput {
        background-color: #0a0a0a !important; 
        color: #00ff41 !important; 
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace;
    }
    
    /* DROPDOWN MENUS */
    div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
        background-color: #0a0a0a !important;
        color: #00ff41 !important;
        border: 1px solid #333 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #111 !important;
        color: #fff !important;
        border-left: 2px solid #00ff41;
    }

    /* METRICS BOXES */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a !important;
        border: 1px solid #333;
        border-left: 3px solid #00ff41 !important;
        padding: 5px 10px;
    }
    div[data-testid="stMetricValue"] { color: #00ff41 !important; font-family: 'Courier New', monospace !important; text-shadow: 0 0 5px #00ff41; }
    div[data-testid="stMetricLabel"] { color: #888 !important; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #333; }
    
    /* BUTTONS */
    div.stDownloadButton > button, div.stButton > button { 
        background-color: #000000 !important; 
        color: #00ff41 !important; 
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase; 
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stDownloadButton > button:hover, div.stButton > button:hover { 
        border-color: #00ff41 !important; 
        box-shadow: 0 0 8px #00ff41 !important;
        color: #fff !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] { color: #888 !important; font-family: 'Courier New', monospace !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00ff41 !important; border-bottom-color: #00ff41 !important; }
    
    /* PLOTS */
    canvas { filter: invert(0); } /* Ensure charts don't get inverted if using dark reader extensions */
</style>
""", unsafe_allow_html=True)

# --- 3. THE UNIVERSAL PHYSICS KERNEL ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(metric_type, r0, r_max, params, iters, lr):
        # Prevent singular geometry errors
        if r0 <= 0: r0 = 0.1
        geom = dde.geometry.Interval(r0, r_max)
        
        # --- THE EINSTEIN FIELD EQUATION RESIDUALS ---
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            
            # 1. Wormholes & Topology
            if metric_type == "Morris-Thorne Wormhole":
                return db_dr - (b / r) * params[0] + (params[1] * (params[2] - b/r))
            elif metric_type == "Einstein-Rosen Bridge":
                return db_dr - (b / r)
            elif metric_type == "Ellis Drainhole":
                return db_dr - (b / (r**2 + params[0]**2)) * (params[1] * params[2])

            # 2. Black Hole Dynamics
            elif metric_type == "Kerr Black Hole":
                return db_dr - (2 * params[0] * r / (r**2 + params[2]**2)) * b
            elif metric_type == "Reissner-Nordström (Charged)":
                return db_dr - (b / r) + (params[1]**2 / r**2)
            elif metric_type == "Kerr-Newman (Charge + Rotation)":
                eff_q = np.sqrt(params[1]**2 + params[3]**2)
                return db_dr - (2 * params[0] * r / (r**2 + params[2]**2)) * b + (eff_q**2 / r**2)
            elif metric_type == "Gott Cosmic String":
                 return db_dr - (params[0] * b)

            # 3. Cosmology & Warp
            elif metric_type == "Alcubierre Warp Drive":
                return db_dr + (params[0] * b * (1-b)**params[2]) / (params[1] * params[3] + 1e-6)
            elif "Expansion" in metric_type or "Contraction" in metric_type:
                sign = -1 if "Expansion" in metric_type else 1
                return db_dr - (b / r) + (sign * params[0] * r**params[1] * params[2])
            
            # 4. Exotic Frontiers
            elif metric_type == "Vaidya (Radiating Star)":
                return db_dr - (b / r) * (1 - params[0] * params[1] * params[2])
            elif "Stringy" in metric_type:
                return db_dr - (b / (r - params[0] * params[1] * params[2]))
            elif "Naked" in metric_type:
                return db_dr - (b / (r * params[0]**(params[1]*params[2])))
            elif "Bonnor-Melvin" in metric_type:
                return db_dr - (params[0]**2 * r)

            return db_dr - (b / r) # Default Fallback

        bc_val = r0 if "Warp" not in metric_type else 1.0
        bc = dde.icbc.DirichletBC(geom, lambda x: bc_val, lambda x, on: on and np.isclose(x[0], r0))
        data = dde.data.PDE(geom, pde, bc, num_domain=400, num_boundary=40)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, metric_type, r0, r_max, p_energy):
        r_v = np.linspace(r0, r_max, 600).reshape(-1, 1)
        b = model.net(torch.tensor(r_v, dtype=torch.float32)).detach().numpy()
        
        # 1. Stress-Energy Tensor (T_uv) approximation
        rho = np.gradient(b.flatten(), r_v.flatten()) / (8 * np.pi * r_v.flatten()**2 + 1e-12)
        
        # 2. Geometric Embedding (Spatial Curvature)
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1 if "Warp" not in metric_type else 0.1
            z[i] = z[i-1] + (1.0 / np.sqrt(np.abs(val)) if np.abs(val) > 1e-9 else 10.0) * dr
            
        # 3. Gravitational Potential (Lapse Function g_tt)
        pot = -np.log(np.abs(1 - b.flatten()/r_v.flatten()) + 1e-6)
        
        # 4. Particle Geodesics (Lorentz Factor Gamma)
        p_gamma = p_energy / (np.sqrt(np.abs(1 - b.flatten()/r_v.flatten())) + 1e-6)
        
        return r_v, b, rho, z, pot, p_gamma

# --- 4. THE UNIVERSAL CONTROL DECK ---
st.title("THE UNIVERSAL SPACETIME LABORATORY")
st.markdown("### 🧬 COMPUTATIONAL GENERAL RELATIVITY ENGINE")

st.sidebar.markdown("### 🛠️ SPACETIME CONFIGURATION")
metric_list = [
    "Morris-Thorne Wormhole", 
    "Kerr Black Hole", 
    "Alcubierre Warp Drive", 
    "Reissner-Nordström (Charged)", 
    "Kerr-Newman (Charge + Rotation)", 
    "Schwarzschild-de Sitter (Expansion)", 
    "Schwarzschild-AdS (Contraction)", 
    "GHS Stringy Black Hole", 
    "Vaidya (Radiating Star)", 
    "Einstein-Rosen Bridge", 
    "JNW (Naked Singularity)", 
    "Ellis Drainhole",
    "Bonnor-Melvin (Magnetic Universe)",
    "Gott Cosmic String"
]
metric_type = st.sidebar.selectbox("Select Metric Class", metric_list)
r0 = st.sidebar.number_input(r"Base Scale Radius ($r_0$ or $M$)", 0.1, 1000.0, 5.0, format="%.4f")

# --- DYNAMIC PARAMETER LOGIC ENGINE ---
params = []
if metric_type == "Morris-Thorne Wormhole":
    st.sidebar.markdown("#### 🌀 Topology Factors")
    params = [
        st.sidebar.slider("Throat Curvature (κ)", 0.1, 1.0, 0.5),
        st.sidebar.slider("Redshift Function (Φ)", 0.0, 1.0, 0.0),
        st.sidebar.slider("Exotic Matter Index (ξ)", 0.0, 2.0, 1.0)
    ]
elif "Kerr" in metric_type:
    st.sidebar.markdown("#### 🕳️ Singularity Dynamics")
    params = [
        st.sidebar.number_input("Event Horizon Mass (M)", 1.0, 100.0, 5.0),
        st.sidebar.slider("Electric Charge (Q)", 0.0, 10.0, 0.0 if "Newman" not in metric_type else 1.0),
        st.sidebar.slider("Angular Momentum / Spin (a)", 0.0, 10.0, 1.0),
        st.sidebar.slider("Magnetic Charge (P)", 0.0, 10.0, 0.0)
    ]
elif "Warp" in metric_type:
    st.sidebar.markdown("#### 🚀 Propulsion Metrics")
    params = [
        st.sidebar.slider("Apparent Velocity (v/c)", 0.1, 10.0, 1.0),
        st.sidebar.slider("Bubble Sigma (σ)", 0.1, 5.0, 1.0),
        st.sidebar.slider("Wall Thickness (w)", 1, 10, 2),
        st.sidebar.slider("Metric Modulation", 0.1, 2.0, 1.0)
    ]
elif "Sitter" in metric_type or "AdS" in metric_type:
    st.sidebar.markdown("#### 🌌 Cosmological Constants")
    params = [
        st.sidebar.number_input("Lambda (Λ)", 0.0, 0.01, 0.0001, format="%.6f"),
        st.sidebar.slider("Spatial Curvature (k)", 1, 3, 2),
        st.sidebar.slider("Density Parameter (Ω)", 0.1, 1.0, 1.0)
    ]
elif "Vaidya" in metric_type:
    st.sidebar.markdown("#### ☀️ Stellar Radiation")
    params = [
        st.sidebar.slider("Mass Loss Rate (Ṁ)", 0.0, 1.0, 0.1),
        st.sidebar.slider("Luminosity (L)", 0.1, 10.0, 1.0),
        st.sidebar.slider("Radial Flux (q)", 0.1, 5.0, 1.0)
    ]
elif "Stringy" in metric_type:
    st.sidebar.markdown("#### 🎻 String Theory")
    params = [
        st.sidebar.slider("Dilaton Field (φ)", 0.0, 5.0, 1.0),
        st.sidebar.slider("Coupling Constant (α)", 0.1, 2.0, 0.5),
        st.sidebar.slider("String Tension (T)", 0.1, 5.0, 1.0)
    ]
elif "Cosmic String" in metric_type:
    st.sidebar.markdown("#### 🎐 Conical Defects")
    params = [
        st.sidebar.slider("Mass per unit length (μ)", 0.1, 2.0, 1.0)
    ]
elif "Naked" in metric_type:
    st.sidebar.markdown("#### ⚠️ Singularity Structure")
    params = [
        st.sidebar.slider("Scalar Field (s)", 0.1, 5.0, 1.0),
        st.sidebar.slider("Gamma Factor (γ)", 0.5, 2.0, 1.0),
        st.sidebar.slider("Field Strength", 0.1, 2.0, 1.0)
    ]
elif "Ellis" in metric_type:
    st.sidebar.markdown("#### 💧 Ether Flow")
    params = [
        st.sidebar.slider("Drain Rate (n)", 1.0, 10.0, 2.0),
        st.sidebar.slider("Ether Velocity (v_f)", 0.1, 5.0, 1.0),
        st.sidebar.slider("Pressure (p)", 0.1, 2.0, 1.0)
    ]
elif "Bonnor" in metric_type:
    st.sidebar.markdown("#### 🧲 Electromagnetism")
    params = [st.sidebar.slider("Magnetic Field Strength (B)", 0.1, 10.0, 1.0)]
else:
    params = [1.0] # Fallback for ER Bridge

st.sidebar.markdown("### ☄️ PARTICLE KINEMATICS")
p_energy = st.sidebar.number_input("Infall Energy / Rest Mass (ε)", 0.0001, 100.0, 1.0, format="%.4f")

st.sidebar.markdown("### ⚙️ SOLVER KERNEL")
lr_val = st.sidebar.number_input("Learning Rate (η)", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Training Epochs", options=[1000, 2500, 5000], value=2500)

# --- EXECUTION PHASE ---
# Removed infinite rerun loop for performance optimization
with st.spinner("CALCULATING SPACETIME GEOMETRY..."):
    model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, params, epochs, lr_val)
    r, b, rho, z, pot, p_gamma = SpacetimeSolver.extract_telemetry(model, metric_type, r0, r0 * 10, p_energy)

# --- KPI STRIP ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("KERNEL LOSS", f"{hist.loss_train[-1][0]:.2e}")
m2.metric("METRIC CLASS", metric_type.split()[0])
m3.metric("PEAK DENSITY", f"{np.max(np.abs(rho)):.4f}")
m4.metric("HORIZON DEPTH", f"{np.max(np.abs(pot)):.2f}")

st.markdown("---")

# --- QUAD-QUADRANT VISUALIZATION HUD ---
v_col, d_col = st.columns([2, 1])

with v_col:
    # 3D INTERACTIVE MANIFOLDS (DUAL FULL-MESH)
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z_geom = np.tile(z.flatten(), (60, 1))
    Z_pot = np.tile(pot.flatten(), (60, 1))
    
    st.subheader("Manifold Zenith: Geometric Embedding ($ds^2$)")
    fig1 = go.Figure(data=[
        # Changed colorscale to 'Electric' for cyber feel
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z_geom, colorscale='Electric', showscale=False, name='Upper'),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z_geom, colorscale='Electric', showscale=False, opacity=0.9, name='Lower')
    ])
    fig1.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0), height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Manifold Nadir: Gravitational Potential ($g_{tt}$)")
    fig2 = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z_pot, colorscale='Electric', showscale=False, name='Positive'),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z_pot, colorscale='Electric', showscale=False, opacity=0.9, name='Negative')
    ])
    fig2.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0), height=400)
    st.plotly_chart(fig2, use_container_width=True)

with d_col:
    # TRI-TAB ANALYTICAL SUITE
    tabs = st.tabs(["📊 STRESS-ENERGY", "📈 TENSOR FIELD", "☄️ GEODESICS"])
    
    # Common plotting style helper
    def style_plot(ax, color='#00ff41'):
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.grid(alpha=0.1, color='white')
        ax.spines['bottom'].set_color('#333')
        ax.spines['top'].set_color('#333') 
        ax.spines['right'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.set_xlabel("Radial Distance (r)", color='white')
        return ax

    with tabs[0]:
        st.subheader("Energy Density Profile ($\rho$)")
        fig_r, ax_r = plt.subplots(facecolor='black', figsize=(5,4))
        # Updated to Green
        ax_r.plot(r, rho, color='#00ff41', lw=2)
        style_plot(ax_r)
        st.pyplot(fig_r)
        
    with tabs[1]:
        st.subheader("Shape Function $b(r)$")
        fig_b, ax_b = plt.subplots(facecolor='black', figsize=(5,4))
        # Updated to Green/White dashed
        ax_b.plot(r, b, color='#00ff41', lw=2, linestyle='--')
        style_plot(ax_b)
        st.pyplot(fig_b)

    with tabs[2]:
        st.subheader("Lorentz Factor ($\gamma$)")
        fig_p, ax_p = plt.subplots(facecolor='black', figsize=(5,4))
        # Updated to White for contrast
        ax_p.plot(r, p_gamma, color='white', lw=1.5)
        ax_p.set_yscale('log')
        style_plot(ax_p)
        st.pyplot(fig_p)
        st.caption("Spike indicates event horizon or singularity approach.")

    # DATA EXPORT HUB
    st.markdown("### 💾 DATA HUB")
    st.download_button(
        label="📥 DOWNLOAD TELEMETRY (CSV)", 
        data=pd.DataFrame({
            "radius": r.flatten(),
            "metric_shape": b.flatten(),
            "energy_density": rho.flatten(),
            "potential": pot.flatten(),
            "gamma_factor": p_gamma.flatten()
        }).to_csv(index=False).encode('utf-8'), 
        file_name=f"telemetry_{metric_type.replace(' ','_')}.csv", 
        use_container_width=True
    )
