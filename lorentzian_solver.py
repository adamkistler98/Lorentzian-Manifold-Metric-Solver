import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
import plotly.graph_objects as go
import io
import time

# --- 1. UI CONFIGURATION & NUCLEAR STEALTH CSS ---
st.set_page_config(
    page_title="Lorentzian Metric Solver", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

st.markdown(r"""
<style>
    /* Main Background - True Void */
    .stApp { background-color: #000000 !important; }
    
    /* Headers & Text - Research HUD Cyan */
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 13px; }
    
    /* NUCLEAR STEALTH OVERRIDE: Dropdowns, Inputs, Popovers */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input, select, .stSelectbox, .stNumberInput {
        background-color: #161B22 !important; 
        color: #00FFF5 !important; 
        border: 1px solid #00ADB5 !important;
    }
    div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
        background-color: #161B22 !important;
        color: #00FFF5 !important;
        border: 1px solid #00ADB5 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #1f242d !important;
        color: #00FFF5 !important;
    }

    /* Metrics - Neon Green */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; text-shadow: 0 0 10px rgba(0,255,65,0.4); }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #222; }
    
    /* Stealth Buttons */
    div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; 
        color: #00ADB5 !important; 
        background-color: #161B22 !important; 
        width: 100%; border-radius: 2px; font-weight: bold; text-transform: uppercase;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; }
    .stTabs [data-baseweb="tab"] { color: #888888 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00ADB5 !important; border-bottom-color: #00ADB5 !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE UNIVERSAL PHYSICS KERNEL ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(metric_type, r0, r_max, params, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            if metric_type == "Morris-Thorne Wormhole":
                # params: [kappa, redshift_phi, exoticity_xi]
                return db_dr - (b / r) * params[0] + (params[1] * (params[2] - b/r))
            elif "Kerr" in metric_type:
                # params: [M, Q, a, P]
                eff_q = np.sqrt(params[1]**2 + (params[3]**2 if len(params)>3 else 0))
                return db_dr - (2 * params[0] * r / (r**2 + params[2]**2)) * b + (eff_q**2 / r**2)
            elif "Warp" in metric_type:
                return db_dr + (params[0] * b * (1-b)**params[2]) / (params[1] * params[3] + 1e-6)
            elif "Sitter" in metric_type:
                sign = -1 if "Expansion" in metric_type else 1
                return db_dr - (b / r) + (sign * params[0] * r**params[1] * params[2])
            elif "Vaidya" in metric_type:
                return db_dr - (b / r) * (1 - params[0] * params[1] * params[2])
            return db_dr - (b / r)

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
        rho = np.gradient(b.flatten(), r_v.flatten()) / (8 * np.pi * r_v.flatten()**2 + 1e-12)
        
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1 if "Warp" not in metric_type else 0.1
            z[i] = z[i-1] + (1.0 / np.sqrt(np.abs(val)) if np.abs(val) > 1e-9 else 10.0) * dr
        
        pot = -np.log(np.abs(1 - b.flatten()/r_v.flatten()) + 1e-6)
        p_gamma = p_energy / (np.sqrt(np.abs(1 - b.flatten()/r_v.flatten())) + 1e-6)
        return r_v, b, rho, z, pot, p_gamma

# --- 3. THE UNIVERSAL SIDEBAR ---
st.sidebar.markdown("### 🛠️ MANIFOLD & CONSTANTS")
metric_list = [
    "Morris-Thorne Wormhole", "Kerr Black Hole", "Alcubierre Warp Drive", 
    "Reissner-Nordström (Charged)", "Schwarzschild-de Sitter (Expansion)", 
    "Schwarzschild-AdS (Contraction)", "GHS Stringy Black Hole", 
    "Vaidya (Radiating Star)", "Kerr-Newman (Charge + Rotation)", 
    "Einstein-Rosen Bridge", "JNW (Naked Singularity)", "Ellis Drainhole"
]
metric_type = st.sidebar.selectbox("Metric Class", metric_list)
r0 = st.sidebar.number_input(r"Base Scale ($r_0$ / $M$)", 0.1, 1000.0, 5.0, format="%.4f")

# Dynamic Physics Variable Logic
params = []
if metric_type == "Morris-Thorne Wormhole":
    params = [st.sidebar.slider("Curvature (κ)", 0.1, 1.0, 0.5), st.sidebar.slider("Redshift (Φ)", 0.0, 1.0, 0.0), st.sidebar.slider("Exoticity (ξ)", 0.0, 2.0, 1.0)]
elif "Kerr" in metric_type:
    params = [st.sidebar.number_input("Mass (M)", 1.0, 100.0, 5.0), st.sidebar.slider("Electric (Q)", 0.0, 10.0, 0.0), st.sidebar.slider("Spin (a)", 0.0, 10.0, 1.0), st.sidebar.slider("Magnetic (P)", 0.0, 10.0, 0.0)]
elif "Warp" in metric_type:
    params = [st.sidebar.slider("Velocity (v/c)", 0.1, 10.0, 1.0), st.sidebar.slider("Sigma (σ)", 0.1, 5.0, 1.0), st.sidebar.slider("Thickness (w)", 1, 10, 2), st.sidebar.slider("Modulation", 0.1, 2.0, 1.0)]
elif "Sitter" in metric_type or "AdS" in metric_type:
    params = [st.sidebar.number_input("Lambda (Λ)", 0.0, 0.01, 0.0001, format="%.6f"), st.sidebar.slider("Curvature (k)", 1, 3, 2), st.sidebar.slider("Omega (Ω)", 0.1, 1.0, 1.0)]
else:
    params = [st.sidebar.slider("Coupling Strength", 0.1, 5.0, 1.0)]

st.sidebar.markdown("### ☄️ KINEMATICS")
p_energy = st.sidebar.number_input("Infall Energy (ε)", 0.0001, 100.0, 1.0, format="%.4f")
lr_val = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)
pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# Solver Execution
model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, params, epochs, lr_val)
r, b, rho, z, pot, p_gamma = SpacetimeSolver.extract_telemetry(model, metric_type, r0, r0 * 10, p_energy)

# Metrics Strip
m1, m2, m3 = st.columns(3)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
m2.metric("CLASS", metric_type.split()[0])
m3.metric("PEAK ENERGY", f"{np.max(np.abs(rho)):.4f}")

st.markdown("---")

# QUAD-QUADRANT HUD LAYOUT
v_col, d_col = st.columns([2, 1])

with v_col:
    # DUAL FULL-MANIFOLD SYSTEM
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z_geom = np.tile(z.flatten(), (60, 1))
    Z_pot = np.tile(pot.flatten(), (60, 1))
    
    st.subheader("Manifold Zenith: Proper Geometry")
    fig1 = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z_geom, colorscale='Viridis', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z_geom, colorscale='Viridis', showscale=False, opacity=0.8)
    ])
    fig1.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0), height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Manifold Nadir: Field Potential ($g_{tt}$)")
    fig2 = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z_pot, colorscale='Magma', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z_pot, colorscale='Magma', showscale=False, opacity=0.8)
    ])
    fig2.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0), height=400)
    st.plotly_chart(fig2, use_container_width=True)

with d_col:
    # ANALYTICAL TABS (RIGHT SIDE)
    tabs = st.tabs(["📊 STRESS-ENERGY", "📈 FIELD TENSORS", "☄️ GEODESICS"])
    
    with tabs[0]:
        st.subheader("Matter Distributions")
        
        fig_r, ax_r = plt.subplots(facecolor='black')
        ax_r.set_facecolor('black'); ax_r.plot(r, rho, color='#FF2E63', lw=2)
        ax_r.tick_params(colors='white'); ax_r.grid(alpha=0.1)
        st.pyplot(fig_r)
        
        if "Wormhole" in metric_type:
            
        elif "Kerr" in metric_type:
            
        elif "Charged" in metric_type:
            
        elif "Expansion" in metric_type:
            
        pass

    with tabs[1]:
        st.subheader("Metric Shape Function b(r)")
        fig_b, ax_b = plt.subplots(facecolor='black')
        ax_b.set_facecolor('black'); ax_b.plot(r, b, color='#00ADB5', lw=2)
        ax_b.tick_params(colors='white'); ax_b.grid(alpha=0.1)
        st.pyplot(fig_b)

    with tabs[2]:
        st.subheader("Infall Energy Factor ($\gamma$)")
        fig_p, ax_p = plt.subplots(facecolor='black')
        ax_p.set_facecolor('black'); ax_p.plot(r, p_gamma, color='#00FF41', lw=2)
        ax_p.set_yscale('log'); ax_p.tick_params(colors='white'); ax_p.grid(alpha=0.1)
        st.pyplot(fig_p)

    st.download_button(
        "📥 EXPORT TELEMETRY (CSV)", 
        data=pd.DataFrame({"r":r.flatten(),"b":b.flatten(),"density":rho.flatten()}).to_csv().encode('utf-8'), 
        file_name="telemetry.csv", 
        use_container_width=True
    )

# Lifecycle
if not pause:
    time.sleep(0.01)
    st.rerun()
