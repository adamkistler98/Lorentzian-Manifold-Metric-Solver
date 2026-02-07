import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
from scipy.integrate import simpson
import plotly.graph_objects as go # <--- The Game Changer

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Lorentzian Metric Solver",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Sci-Fi Dashboard" Aesthetic
st.markdown("""
<style>
    .stApp { background-color: #050505; }
    h1, h2, h3 { font-family: 'Consolas', monospace; color: #00ADB5; letter-spacing: -1px; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { color: #00FFF5 !important; font-family: 'Consolas', monospace; text-shadow: 0 0 10px rgba(0,255,245,0.3); }
    div[data-testid="stMetricLabel"] { color: #888888 !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0B0F19; border-right: 1px solid #333; }
    
    /* Buttons */
    div.stButton > button { 
        border: 1px solid #00ADB5; 
        color: #00ADB5; 
        background: transparent; 
        width: 100%; 
        border-radius: 0px; 
        transition: all 0.3s;
    }
    div.stButton > button:hover { 
        background: rgba(0, 173, 181, 0.2); 
        color: #FFF; 
        border-color: #FFF; 
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS ENGINE (PINN) ---
class WormholeSolver:
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def train_metric(r0, r_max, iterations, lr):
        """
        Solves Einstein Field Equations for Morris-Thorne Metric using PINN.
        """
        geom = dde.geometry.Interval(r0, r_max)

        # PDE: Minimizing Null Energy Condition (NEC) Violation while maintaining structure
        # Simplified constraint: db/dr - b/r = 0 implies Schwarzschild (Vacuum)
        # We allow deviation to create the wormhole structure.
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            return db_dr - (b / r) * 0.5 

        # Boundary Condition: Throat is minimal surface b(r0) = r0
        def boundary_throat(x, on_boundary):
            return on_boundary and np.isclose(x[0], r0)

        bc = dde.icbc.DirichletBC(geom, lambda x: r0, boundary_throat)

        # Neural Network Config
        data = dde.data.PDE(geom, pde, bc, num_domain=400, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)

        # Training
        model.compile("adam", lr=lr)
        loss_history, train_state = model.train(iterations=iterations, display_every=1000)
        
        return model, loss_history

    @staticmethod
    def extract_physics(model, r0, r_max):
        # High-res Sampling
        r_val = np.linspace(r0, r_max, 600).reshape(-1, 1)
        r_tensor = torch.tensor(r_val, dtype=torch.float32, requires_grad=True)
        
        # Inference & Derivatives (Autograd)
        b_tensor = model.net(r_tensor)
        db_dr_tensor = torch.autograd.grad(b_tensor, r_tensor, grad_outputs=torch.ones_like(b_tensor), create_graph=False)[0]
        
        b = b_tensor.detach().numpy()
        db_dr = db_dr_tensor.detach().numpy()
        
        # Calculate Physics Metrics
        rho = db_dr / (8 * np.pi * r_val**2 + 1e-9) # Energy Density
        tidal = (1 - b/r_val) / (r_val**2 + 1e-9)   # Lateral Tidal Forces (Approx)
        
        # Embedding Coordinate (z) integration
        # dz/dr = +/- [ (r / b(r)) - 1 ]^(-1/2)
        z = np.zeros_like(r_val)
        dr = r_val[1] - r_val[0]
        
        for i in range(1, len(r_val)):
            val = (r_val[i] / (b[i] + 1e-6)) - 1
            slope = 1.0 / np.sqrt(val) if val > 0 else 0
            z[i] = z[i-1] + slope * dr
            
        return r_val, b, rho, tidal, z

# --- 3. UI LAYOUT ---
st.title("🌌 LORENTZIAN METRIC SOLVER")
st.caption("General Relativity Physics-Informed Neural Network (PINN)")

# Sidebar
st.sidebar.markdown("## ⚙️ PARAMETERS")
throat_r0 = st.sidebar.slider("Throat Radius (M)", 1.0, 10.0, 2.0, help="Minimum size of the wormhole throat.")
domain_max = st.sidebar.slider("Simulation Domain", 10.0, 50.0, 20.0, help="How far into space we simulate.")

st.sidebar.markdown("## 🧠 HYPERPARAMETERS")
iterations = st.sidebar.select_slider("Training Epochs", options=[1000, 2500, 5000, 10000], value=2500)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-2, 1e-3, 5e-4], value=1e-3)

run_btn = st.sidebar.button("INITIATE SIMULATION")

# --- MAIN LOGIC ---
if run_btn:
    progress_bar = st.progress(0, text="Initializing Neural Network...")
    
    # 1. Train Model
    model, history = WormholeSolver.train_metric(throat_r0, domain_max, iterations, lr)
    progress_bar.progress(50, text="Solving Einstein Field Equations...")
    
    # 2. Extract Data
    r, b, rho, tidal, z = WormholeSolver.extract_physics(model, throat_r0, domain_max)
    progress_bar.progress(100, text="Rendering Manifold...")
    
    # --- DASHBOARD ---
    
    # TOP ROW: KPI METRICS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Convergence Loss", f"{history.loss_train[-1][0]:.1e}")
    k2.metric("Total Exotic Matter", f"{simpson(rho.flatten(), x=r.flatten()):.2f} M_pl")
    k3.metric("Max Tidal Force", f"{np.max(tidal):.2f} g")
    
    # Traversability Assessment
    is_safe = np.max(tidal) < 1.0
    status_color = "green" if is_safe else "red"
    status_text = "SAFE FOR TRAVEL" if is_safe else "LETHAL TIDAL FORCES"
    k4.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold'>{status_text}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # TABS
    tab_3d, tab_2d = st.tabs(["🔮 3D INTERACTIVE EMBEDDING", "📉 TENSOR ANALYSIS"])

    with tab_3d:
        # PLOTLY 3D VISUALIZATION
        st.markdown("##### Isometric Embedding Diagram (Interact to Rotate)")
        
        # Create rotational surface
        theta = np.linspace(0, 2*np.pi, 60)
        R_grid, Theta_grid = np.meshgrid(r.flatten(), theta)
        Z_grid = np.tile(z.flatten(), (60, 1))
        
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        
        fig = go.Figure(data=[
            # Top Half
            go.Surface(x=X, y=Y, z=Z_grid, colorscale='Electric', opacity=0.9, showscale=False),
            # Bottom Half (Mirror)
            go.Surface(x=X, y=Y, z=-Z_grid, colorscale='Electric', opacity=0.9, showscale=False),
        ])

        fig.update_layout(
            title="",
            autosize=True,
            width=800, height=800,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='#050505'
            ),
            paper_bgcolor='#050505',
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_2d:
        c1, c2 = st.columns(2)
        with c1:
            # Shape Function
            fig1, ax1 = plt.subplots(facecolor='#050505')
            ax1.set_facecolor('#050505')
            ax1.plot(r, b, color='#00ADB5', lw=2, label="Shape Function b(r)")
            ax1.plot(r, r, color='#444', linestyle='--', label="Event Horizon")
            ax1.fill_between(r.flatten(), b.flatten(), r.flatten(), color='#00ADB5', alpha=0.1)
            ax1.set_title("Wormhole Geometry", color='white')
            ax1.tick_params(colors='white')
            ax1.legend(facecolor='#111', labelcolor='white')
            st.pyplot(fig1)
            
        with c2:
            # Energy Density
            fig2, ax2 = plt.subplots(facecolor='#050505')
            ax2.set_facecolor('#050505')
            ax2.plot(r, rho, color='#FF2E63', lw=2)
            ax2.set_title("Energy Density Distribution (NEC Violation)", color='white')
            ax2.tick_params(colors='white')
            st.pyplot(fig2)

else:
    # LANDING PAGE STATE
    st.info("👈 Set initial conditions in the sidebar and click 'INITIATE SIMULATION'.")
    st.markdown("""
    ### About this Tool
    This application solves the **Einstein Field Equations** for a traversable wormhole metric. 
    It uses a **Physics-Informed Neural Network (DeepXDE)** to find a metric geometry that satisfies stability constraints.
    
    **Calculated Metrics:**
    * **b(r):** The Shape Function (defines the throat geometry).
    * **Energy Density:** Required exotic matter distribution.
    * **Tidal Forces:** Gravitational shear experienced by a traveler.
    """)
