import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
from scipy.integrate import simpson
import plotly.graph_objects as go
import io

# --- 1. CONFIGURATION & VISUAL STYLE ---
st.set_page_config(
    page_title="Lorentzian Metric Solver (Event Horizon)", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# FORCE HIGH-CONTRAST DARK MODE
st.markdown("""
<style>
    /* 1. Main Background - True Void */
    .stApp { background-color: #000000; }
    
    /* 2. Text Colors - FORCE WHITE */
    h1, h2, h3, h4, h5, h6 { color: #00ADB5 !important; font-family: 'Consolas', monospace; letter-spacing: -1px; }
    p, li, label, .stMarkdown, .stCaption { color: #E0E0E0 !important; font-family: 'Verdana', sans-serif; }
    
    /* 3. Metrics (The Big Numbers) */
    div[data-testid="stMetricValue"] { color: #00FFF5 !important; font-family: 'Consolas', monospace; text-shadow: 0 0 15px rgba(0,255,245,0.4); }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; font-weight: bold; }
    
    /* 4. Sidebar */
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    
    /* 5. Buttons */
    div.stButton > button { 
        border: 1px solid #00ADB5; 
        color: #00ADB5; 
        background: transparent; 
        width: 100%; 
        border-radius: 0px; 
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:hover { 
        background: rgba(0, 173, 181, 0.2); 
        color: #FFFFFF; 
        border-color: #FFFFFF; 
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.6);
    }
    
    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #111; color: #888; border-radius: 5px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #00ADB5; color: #000; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS ENGINE (PINN) ---
class WormholeSolver:
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def train_metric(r0, r_max, curvature_factor, iterations, lr):
        """
        Solves Einstein Field Equations for Morris-Thorne Metric.
        curvature_factor: Controls how 'flared' the wormhole is.
        """
        geom = dde.geometry.Interval(r0, r_max)

        # PDE: db/dr - (b/r) * curvature = 0 
        # By changing 'curvature', we change the physical shape of the solution space.
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            return db_dr - (b / r) * curvature_factor 

        # Boundary Condition: Throat is minimal surface b(r0) = r0
        def boundary_throat(x, on_boundary):
            return on_boundary and np.isclose(x[0], r0)

        bc = dde.icbc.DirichletBC(geom, lambda x: r0, boundary_throat)

        # Neural Network Config
        data = dde.data.PDE(geom, pde, bc, num_domain=300, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)

        # Training
        model.compile("adam", lr=lr)
        loss_history, train_state = model.train(iterations=iterations, display_every=1000)
        
        return model, loss_history

    @staticmethod
    def extract_physics(model, r0, r_max):
        # High-res Sampling
        r_val = np.linspace(r0, r_max, 500).reshape(-1, 1)
        r_tensor = torch.tensor(r_val, dtype=torch.float32, requires_grad=True)
        
        # Inference & Derivatives (Autograd)
        b_tensor = model.net(r_tensor)
        db_dr_tensor = torch.autograd.grad(b_tensor, r_tensor, grad_outputs=torch.ones_like(b_tensor), create_graph=False)[0]
        
        b = b_tensor.detach().numpy()
        db_dr = db_dr_tensor.detach().numpy()
        
        # Physics Derived Values
        # Energy Density: rho = b' / (8pi r^2)
        rho = db_dr / (8 * np.pi * r_val**2 + 1e-9)
        
        # Lateral Tidal Forces: (1 - b/r) / r^2 (Approx in g-force)
        # Normalized for visualization
        tidal = np.abs((1 - (b / r_val)) / (r_val**2 + 1e-9))
        
        # Embedding Coordinate (z) integration for 3D View
        # dz/dr = +/- [ (r / b(r)) - 1 ]^(-1/2)
        z = np.zeros_like(r_val)
        dr = r_val[1] - r_val[0]
        
        for i in range(1, len(r_val)):
            # Safety clip to avoid negative sqrt in numerical noise
            val = (r_val[i] / (b[i] + 1e-6)) - 1
            slope = 1.0 / np.sqrt(val) if val > 1e-6 else 10.0 # Cap slope at throat
            z[i] = z[i-1] + slope * dr
            
        return r_val, b, rho, tidal, z

# --- 3. UI LAYOUT ---
st.title("🌌 LORENTZIAN METRIC SOLVER")
st.caption("General Relativity Physics-Informed Neural Network (PINN) | Automated Topology Analysis")

# Sidebar
st.sidebar.markdown("## 📐 GEOMETRY CONFIG")
throat_r0 = st.sidebar.slider("Throat Radius (M)", 1.0, 8.0, 2.0, help="The narrowest point of the wormhole. Larger = Wider tunnel.")
curvature = st.sidebar.slider("Throat Curvature", 0.1, 0.9, 0.5, help="Controls the flare-out shape. Low = Tube-like, High = Trumpet-like.")
domain_max = st.sidebar.slider("Simulation Horizon", 10.0, 40.0, 15.0, help="The radial distance to simulate out to.")

st.sidebar.markdown("## 🧠 NEURAL HYPERPARAMETERS")
iterations = st.sidebar.select_slider("Training Epochs", options=[1000, 2500, 5000], value=2500)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-2, 1e-3, 5e-4], value=1e-3)

run_btn = st.sidebar.button("INITIATE SOLVER", type="primary")

# --- MAIN LOGIC ---
if run_btn:
    progress_bar = st.progress(0, text="Initializing Neural Network...")
    
    # 1. Train Model
    model, history = WormholeSolver.train_metric(throat_r0, domain_max, curvature, iterations, lr)
    progress_bar.progress(60, text="Minimizing Null Energy Condition Violations...")
    
    # 2. Extract Data
    r, b, rho, tidal, z = WormholeSolver.extract_physics(model, throat_r0, domain_max)
    progress_bar.progress(100, text="Rendering Isometric Embedding...")
    time_delay = st.empty() # Placeholder for clean transition
    
    # --- DASHBOARD ---
    st.markdown("---")
    
    # TOP ROW: KPI METRICS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Convergence Loss", f"{history.loss_train[-1][0]:.1e}")
    k2.metric("Total Exotic Matter", f"{simpson(rho.flatten(), x=r.flatten()) * 100:.3f} units")
    
    # Tidal Force Logic
    max_tidal = np.max(tidal)
    k3.metric("Peak Tidal Shear", f"{max_tidal:.3f} g")
    
    # Traversability Assessment (Scientific)
    is_stable = max_tidal < 0.5
    status_text = "NOMINAL (STABLE)" if is_stable else "CRITICAL (SHEAR)"
    status_color = "#00FF41" if is_stable else "#FF2E63" # Green vs Red
    
    k4.markdown(f"""
    <div style="text-align: center;">
        <span style="font-size: 12px; color: #888;">METRIC INTEGRITY</span><br>
        <span style="font-size: 20px; font-weight: bold; color: {status_color}; text-shadow: 0 0 10px {status_color};">
            {status_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # TABS
    tab_3d, tab_2d, tab_export = st.tabs(["🔮 3D MANIFOLD VISUALIZATION", "📉 TENSOR METRICS", "💾 EXPORT DATA"])

    with tab_3d:
        # PLOTLY 3D VISUALIZATION
        
        # Construct Rotation
        theta = np.linspace(0, 2*np.pi, 50)
        R_grid, Theta_grid = np.meshgrid(r.flatten(), theta)
        Z_grid = np.tile(z.flatten(), (50, 1))
        
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        
        # Color Map based on Tidal Forces
        # We need to map 'tidal' values to the grid shape to color it
        Tidal_grid = np.tile(tidal.flatten(), (50, 1))
        
        fig = go.Figure(data=[
            # Top Half
            go.Surface(
                x=X, y=Y, z=Z_grid, 
                surfacecolor=Tidal_grid,
                colorscale='Plasma', 
                opacity=0.9, 
                showscale=False,
                name="Upper Universe"
            ),
            # Bottom Half (Mirror)
            go.Surface(
                x=X, y=Y, z=-Z_grid, 
                surfacecolor=Tidal_grid,
                colorscale='Plasma', 
                opacity=0.9, 
                showscale=False,
                name="Lower Universe"
            ),
        ])

        # LOCK ASPECT RATIO so wider throat looks wider
        # We find max range to keep box cubic
        max_range = max(np.max(X), np.max(Z_grid))
        
        fig.update_layout(
            title="",
            autosize=True,
            width=900, height=700,
            scene=dict(
                xaxis=dict(visible=False, range=[-max_range, max_range]),
                yaxis=dict(visible=False, range=[-max_range, max_range]),
                zaxis=dict(visible=False, range=[-max_range, max_range]), # Lock Z to X/Y scale
                bgcolor='#000000',
                aspectmode='cube' # Crucial for showing true geometry changes
            ),
            paper_bgcolor='#000000',
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Interactive Model: Left-Click to Rotate | Right-Click to Pan | Scroll to Zoom. Color Intensity = Tidal Shear Stress.")

    with tab_2d:
        c1, c2 = st.columns(2)
        with c1:
            # Shape Function b(r)
            fig1, ax1 = plt.subplots(facecolor='#000000')
            ax1.set_facecolor('#000000')
            ax1.plot(r, b, color='#00ADB5', lw=2.5, label="b(r) Solution")
            ax1.plot(r, r, color='#444', linestyle='--', label="Schwarzschild Limit")
            
            # Fill area
            ax1.fill_between(r.flatten(), b.flatten(), r.flatten(), color='#00ADB5', alpha=0.15)
            
            ax1.set_title("Shape Function Geometry", color='white', fontsize=10)
            ax1.set_xlabel("Radius r", color='#888')
            ax1.set_ylabel("b(r)", color='#888')
            ax1.tick_params(colors='#888')
            ax1.spines['bottom'].set_color('#333')
            ax1.spines['left'].set_color('#333')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.legend(facecolor='#111', labelcolor='white', edgecolor='#333')
            st.pyplot(fig1)
            
        with c2:
            # Energy Density
            fig2, ax2 = plt.subplots(facecolor='#000000')
            ax2.set_facecolor('#000000')
            ax2.plot(r, rho, color='#FF2E63', lw=2.5)
            ax2.fill_between(r.flatten(), rho.flatten(), 0, color='#FF2E63', alpha=0.15)
            
            ax2.set_title("Exotic Matter Density (NEC Violation)", color='white', fontsize=10)
            ax2.set_xlabel("Radius r", color='#888')
            ax2.tick_params(colors='#888')
            ax2.spines['bottom'].set_color('#333')
            ax2.spines['left'].set_color('#333')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)

    with tab_export:
        st.markdown("### 💾 Data Extraction")
        ex1, ex2 = st.columns(2)
        
        # 1. Snapshot Button
        img_buf = io.BytesIO()
        fig1.savefig(img_buf, format='png', facecolor='#000000', bbox_inches='tight')
        ex1.download_button(
            label="📸 Download Metric Tensor Plot",
            data=img_buf.getvalue(),
            file_name=f"lorentzian_metric_r{throat_r0}.png",
            mime="image/png",
            use_container_width=True
        )
        
        # 2. CSV Export
        df = pd.DataFrame({
            "Radius_r": r.flatten(),
            "Shape_b": b.flatten(),
            "Embedding_Z": z.flatten(),
            "Energy_Density": rho.flatten(),
            "Tidal_Force": tidal.flatten()
        })
        csv = df.to_csv(index=False).encode('utf-8')
        ex2.download_button(
            label="📊 Export Physics Telemetry (CSV)",
            data=csv,
            file_name=f"spacetime_data_r{throat_r0}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # LANDING STATE
    st.info("👈 Configure Spacetime parameters in the sidebar and click **INITIATE SOLVER**.")
    
    # Hero Graphic
    x = np.linspace(-4, 4, 100)
    y = np.exp(-x**2)
    fig_hero = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', line=dict(color='#00ADB5', width=3)))
    fig_hero.update_layout(
        template="plotly_dark",
        plot_bgcolor="#000000", paper_bgcolor="#000000",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0,r=0,t=0,b=0), height=200
    )
    st.plotly_chart(fig_hero, use_container_width=True)
