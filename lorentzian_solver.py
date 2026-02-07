import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
from scipy.integrate import simpson
import plotly.graph_objects as go
import io
import time

# --- 1. UI CONFIGURATION & STEALTH STYLING ---
st.set_page_config(page_title="Lorentzian Metric Solver", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    .stApp { background-color: #000000; }
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; }
    
    /* Stealth Input Boxes & Sliders */
    input { background-color: #161B22 !important; color: #00FFF5 !important; border: 1px solid #333 !important; }
    div[data-baseweb="input"] { background-color: #161B22 !important; border: 1px solid #00ADB5 !important; }
    div[role="slider"] { background-color: #00ADB5 !important; }
    
    /* Stealth Metrics */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; }
    div[data-testid="stMetricLabel"] { color: #888 !important; text-transform: uppercase; }

    /* Stealth Buttons */
    div.stButton > button, div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; color: #00ADB5 !important; background-color: #161B22 !important;
        width: 100%; border-radius: 2px; font-weight: bold; text-transform: uppercase; transition: all 0.4s ease;
    }
    div.stButton > button:hover { background-color: #1f242d !important; color: #00FFF5 !important; box-shadow: 0 0 15px rgba(0, 173, 181, 0.4); }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS CORE ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(r0, r_max, curve, redshift, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            return db_dr - (b / r) * curve + (redshift * (1 - b/r))
        
        bc = dde.icbc.DirichletBC(geom, lambda x: r0, lambda x, on: on and np.isclose(x[0], r0))
        data = dde.data.PDE(geom, pde, bc, num_domain=600, num_boundary=60)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, r0, r_max, redshift_val):
        r_v = np.linspace(r0, r_max, 800).reshape(-1, 1)
        r_t = torch.tensor(r_v, dtype=torch.float32, requires_grad=True)
        b_t = model.net(r_t)
        db_dr = torch.autograd.grad(b_t, r_t, grad_outputs=torch.ones_like(b_t))[0].detach().numpy()
        b = b_t.detach().numpy()
        
        rho = db_dr / (8 * np.pi * r_v**2 + 1e-12)
        tau = (b / (8 * np.pi * r_v**3)) - (2 * redshift_val * (1 - b/r_v) / (8 * np.pi * r_v))
        exoticity = rho - tau 
        
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1
            z[i] = z[i-1] + (1.0 / np.sqrt(val) if val > 1e-9 else 15.0) * dr
            
        return r_v, b, rho, tau, exoticity, z

# --- 3. DASHBOARD ---
st.sidebar.markdown(r"### 🧬 $G_{\mu\nu}$ TOPOLOGY")
r_throat = st.sidebar.number_input(r"Throat Radius ($r_0$)", 0.001, 100.0, 2.0, format="%.4f")
flare = st.sidebar.slider(r"Curvature Intensity ($\kappa$)", 0.01, 0.99, 0.5)
redshift = st.sidebar.slider(r"Redshift Offset ($\Phi$)", 0.0, 1.0, 0.0)

st.sidebar.markdown(r"### ⚙️ NUMERICAL KERNEL")
lr_val = st.sidebar.number_input(r"Learning Rate ($\eta$)", 0.000001, 0.1, 0.001, format="%.6f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)

pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# Solver Execution
model, hist = SpacetimeSolver.solve_manifold(r_throat, r_throat * 12, flare, redshift, epochs, lr_val)
r, b, rho, tau, xi, z = SpacetimeSolver.extract_telemetry(model, r_throat, r_throat * 12, redshift)

# Metrics (FIXED: Added 'r' for raw strings to avoid SyntaxError)
m1, m2, m3 = st.columns(3)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
m2.metric(r"EXOTICITY INDEX ($\xi$)", f"{np.min(xi):.4f}")
m3.metric("NEC VIOLATION", "DETECTED" if np.min(xi) < 0 else "NULL")

st.markdown("---")
v_col, d_col = st.columns([2, 1])

with v_col:
    # 3D View
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z = np.tile(z.flatten(), (60, 1))
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, colorscale='Viridis', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, colorscale='Viridis', showscale=False)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='cube'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Snapshot / Export
    c_btn1, c_btn2 = st.columns(2)
    c_btn1.download_button("📸 SNAPSHOT TOPOLOGY", data=io.BytesIO().getvalue(), file_name="topology.png")
    c_btn2.download_button("📊 EXPORT TELEMETRY", data=pd.DataFrame({"r": r.flatten(), "b": b.flatten()}).to_csv().encode('utf-8'), file_name="spacetime.csv")

with d_col:
    tabs = st.tabs(["📉 EXOTICITY", "📈 TENSORS", "☄️ PARTICLE FLUX"])
    
    with tabs[0]:
        st.subheader("Energy Condition Analysis")
        
        fig, ax = plt.subplots(facecolor='black')
        ax.set_facecolor('black')
        ax.plot(r, xi, color='#FF2E63', label=r"Exoticity ($\rho - \tau$)")
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.fill_between(r.flatten(), xi.flatten(), 0, where=(xi.flatten() < 0), color='#FF2E63', alpha=0.2)
        ax.legend(); ax.tick_params(colors='white'); ax.set_xlabel("Radius r")
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Manifold Geometry")
        
        fig2, ax2 = plt.subplots(2, 1, facecolor='black', figsize=(6, 8))
        ax2[0].plot(r, b, color='#00ADB5', label=r"Shape $b(r)$")
        ax2[1].plot(r, rho, color='#00FF41', label=r"Density $\rho$")
        for a in ax2: 
            a.set_facecolor('black'); a.tick_params(colors='white'); a.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    with tabs[2]:
        st.subheader("High-Energy Particle Infall")
        # Simulating particle blueshift as it approaches r0
        # E_obs = E_inf / sqrt(1 - b/r)
        flux_r = r.flatten()
        energy_gain = 1.0 / (np.sqrt(np.abs(1 - b.flatten()/flux_r)) + 1e-3)
        
        fig3, ax3 = plt.subplots(facecolor='black')
        ax3.set_facecolor('black')
        ax3.plot(flux_r, energy_gain, color='#FFD700', lw=2)
        ax3.set_yscale('log')
        ax3.set_title("Relativistic Energy Shift", color='white')
        ax3.set_ylabel("Kinetic Factor (log)", color='white')
        ax3.tick_params(colors='white')
        st.pyplot(fig3)
        st.caption("Visualizing the kinetic energy spike of particles as they approach the manifold throat.")

if not pause:
    time.sleep(0.01)
    st.rerun()
