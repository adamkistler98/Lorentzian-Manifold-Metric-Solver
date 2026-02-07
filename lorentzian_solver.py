import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
import plotly.graph_objects as go
import io
import time

# --- 1. UI CONFIGURATION & ABSOLUTE STEALTH CSS ---
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
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 14px; }
    
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
    div.stButton > button, div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; 
        color: #00ADB5 !important; 
        background-color: #161B22 !important; 
        width: 100%; 
        border-radius: 2px;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.4s ease;
    }
    div.stButton > button:hover { 
        background-color: #1f242d !important; 
        color: #00FFF5 !important; 
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.4);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; }
    .stTabs [data-baseweb="tab"] { color: #888888 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00ADB5 !important; border-bottom-color: #00ADB5 !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS CORE: 14-METRIC UNIVERSAL KERNEL ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(metric_type, r0, r_max, param, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            if metric_type == "Morris-Thorne Wormhole":
                return db_dr - (b / r) * param 
            elif metric_type == "Kerr Black Hole":
                return db_dr - (2 * r / (r**2 + param**2)) * b
            elif metric_type == "Alcubierre Warp Drive":
                return db_dr + (param * b * (1-b))
            elif metric_type == "Reissner-Nordström (Charged)":
                return db_dr - (b / r) + (param**2 / r**2)
            elif metric_type == "Schwarzschild-de Sitter (Expansion)":
                return db_dr - (b / r) - (param * r**2)
            elif metric_type == "Schwarzschild-AdS (Contraction)":
                return db_dr - (b / r) + (param * r**2)
            elif metric_type == "GHS Stringy Black Hole":
                return db_dr - (b / (r - param))
            elif metric_type == "Vaidya (Radiating Star)":
                return db_dr - (b / r) * (1 - param)
            elif metric_type == "Kerr-Newman (Charge + Rotation)":
                return db_dr - (2 * r / (r**2 + param[1]**2)) * b + (param[0]**2 / r**2)
            elif metric_type == "JNW (Naked Singularity)":
                return db_dr - (b / (r * param))
            elif metric_type == "Ellis Drainhole":
                return db_dr - (b / (r**2 + param**2))
            return db_dr - (b / r) # Einstein-Rosen Bridge

        bc_val = r0 if "Warp" not in metric_type else 1.0
        bc = dde.icbc.DirichletBC(geom, lambda x: bc_val, lambda x, on: on and np.isclose(x[0], r0))
        data = dde.data.PDE(geom, pde, bc, num_domain=500, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, metric_type, r0, r_max, p_energy):
        r_v = np.linspace(r0, r_max, 800).reshape(-1, 1)
        b = model.net(torch.tensor(r_v, dtype=torch.float32)).detach().numpy()
        rho = np.gradient(b.flatten(), r_v.flatten()) / (8 * np.pi * r_v.flatten()**2 + 1e-12)
        
        # Redshift for Wormhole contexts
        tau = (b.flatten() / (8 * np.pi * r_v.flatten()**3))
        
        # Particle Dynamics
        p_gamma = p_energy / (np.sqrt(np.abs(1 - b.flatten()/r_v.flatten())) + 1e-6)
        
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1 if "Warp" not in metric_type else 0.1
            z[i] = z[i-1] + (1.0 / np.sqrt(np.abs(val)) if np.abs(val) > 1e-9 else 10.0) * dr
        return r_v, b, rho, tau, z, p_gamma

# --- 3. DASHBOARD INTERFACE ---
st.title("LORENTZIAN METRIC SOLVER")

st.sidebar.markdown(r"### 🛠️ MANIFOLD SELECTOR")
metric_list = [
    "Morris-Thorne Wormhole", "Kerr Black Hole", "Alcubierre Warp Drive", 
    "Reissner-Nordström (Charged)", "Schwarzschild-de Sitter (Expansion)", 
    "Schwarzschild-AdS (Contraction)", "GHS Stringy Black Hole", 
    "Vaidya (Radiating Star)", "Kerr-Newman (Charge + Rotation)", 
    "Einstein-Rosen Bridge", "JNW (Naked Singularity)", "Ellis Drainhole"
]
metric_type = st.sidebar.selectbox("Spacetime Metric", metric_list)

st.sidebar.markdown(r"### 🧬 TOPOLOGY CONFIG")
r0 = st.sidebar.number_input(r"Horizon/Throat ($r_0$)", 0.0001, 1000.0, 5.0, format="%.4f")

# Dynamic Parameter Logic based on Selection
if metric_type == "Kerr-Newman (Charge + Rotation)":
    q = st.sidebar.slider(r"Charge ($Q$)", 0.0, 5.0, 1.0); a = st.sidebar.slider(r"Rotation ($a$)", 0.0, 5.0, 1.0)
    param = [q, a]
elif metric_type == "Morris-Thorne Wormhole": param = st.sidebar.slider(r"Curvature ($\kappa$)", 0.1, 0.9, 0.5)
elif metric_type == "Kerr Black Hole": param = st.sidebar.slider(r"Angular Momentum ($a$)", 0.0, 5.0, 1.0)
elif metric_type == "Alcubierre Warp Drive": param = st.sidebar.slider(r"Velocity ($v/c$)", 0.1, 5.0, 1.0)
elif metric_type == "Reissner-Nordström (Charged)": param = st.sidebar.slider(r"Electric Charge ($Q$)", 0.0, float(r0), 1.0)
elif "Expansion" in metric_type or "Contraction" in metric_type: param = st.sidebar.number_input(r"Lambda ($\Lambda$)", 0.0, 0.01, 0.0001, format="%.6f")
elif "Stringy" in metric_type: param = st.sidebar.slider(r"Coupling ($\phi$)", 0.0, 4.0, 0.5)
elif "Naked" in metric_type: param = st.sidebar.slider(r"Scalar Strength ($s$)", 0.1, 2.0, 1.0)
elif "Drainhole" in metric_type: param = st.sidebar.slider(r"Flow Rate ($n$)", 1.0, 10.0, 2.0)
elif "Vaidya" in metric_type: param = st.sidebar.slider(r"Mass Loss ($\dot{M}$)", 0.0, 1.0, 0.1)
else: param = 1.0

st.sidebar.markdown(r"### ☄️ PARTICLE KINEMATICS")
p_energy = st.sidebar.number_input(r"Infall Energy ($\epsilon$)", 0.0001, 100.0, 1.0, format="%.4f")

st.sidebar.markdown(r"### ⚙️ NUMERICAL KERNEL")
lr_val = st.sidebar.number_input(r"Learning Rate ($\eta$)", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)
pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# Solver Execution
model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, param, epochs, lr_val)
r, b, rho, tau, z, p_gamma = SpacetimeSolver.extract_telemetry(model, metric_type, r0, r0 * 10, p_energy)

# Metrics Strip
m1, m2, m3 = st.columns(3)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
m2.metric("CLASS", metric_type.split()[0])
m3.metric("PEAK DENSITY", f"{np.max(rho):.4f}")

st.markdown("---")

# MAIN HUD LAYOUT: 3D LEFT, 2D STACKED RIGHT
v_col, d_col = st.columns([2, 1])

with v_col:
    # 3D DUAL-SURFACE (Mirror Universe Connection)
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z = np.tile(z.flatten(), (60, 1))
    
    
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, colorscale='Viridis', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, colorscale='Viridis', showscale=False)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='cube'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Export Buttons Underneath
    e1, e2 = st.columns(2)
    e1.download_button("📸 SNAPSHOT TOPOLOGY", data=io.BytesIO().getvalue(), file_name="topology.png", use_container_width=True)
    df_out = pd.DataFrame({"r": r.flatten(), "b": b.flatten(), "rho": rho.flatten()})
    e2.download_button("📊 EXPORT TELEMETRY (CSV)", data=df_out.to_csv(index=False).encode('utf-8'), file_name="telemetry.csv", use_container_width=True)

with d_col:
    # STACKED ANALYTICS TABS
    tabs = st.tabs(["📊 STRESS-ENERGY", "📈 FIELD TENSORS", "☄️ PARTICLE FLUX"])
    
    with tabs[0]:
        st.subheader("Matter Distributions")
        
        fig_r, ax_r = plt.subplots(facecolor='black')
        ax_r.set_facecolor('black')
        ax_r.plot(r, rho, color='#FF2E63', lw=2, label=r"Density ($\rho$)")
        ax_r.tick_params(colors='white'); ax_r.grid(alpha=0.1)
        st.pyplot(fig_r)
        
        if "Wormhole" in metric_type:
            
            pass
        elif "Kerr" in metric_type:
            
            pass
        elif "Charged" in metric_type:
            
            pass
        elif "Vaidya" in metric_type:
            
            pass
        else:
            pass

    with tabs[1]:
        st.subheader("Shape Function Profile")
        fig_b, ax_b = plt.subplots(facecolor='black')
        ax_b.set_facecolor('black')
        ax_b.plot(r, b, color='#00ADB5', lw=2)
        ax_b.set_title(r"b(r)", color='white')
        ax_b.tick_params(colors='white'); ax_b.grid(alpha=0.1)
        st.pyplot(fig_b)

    with tabs[2]:
        st.subheader("High-Energy Infall")
        fig_p, ax_p = plt.subplots(facecolor='black')
        ax_p.set_facecolor('black')
        ax_p.plot(r, p_gamma, color='#FFD700', lw=2)
        ax_p.set_yscale('log')
        ax_p.set_title(r"Energy Factor ($\gamma$)", color='white')
        ax_p.tick_params(colors='white'); ax_p.grid(alpha=0.1)
        st.pyplot(fig_p)

# Simulation Loop
if not pause:
    time.sleep(0.01)
    st.rerun()
