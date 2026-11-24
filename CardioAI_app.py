"""
CardioAI – Explainable ECG Intelligence
Main Streamlit Application
Explainable Deep Learning for Multi-Label ECG Arrhythmia Classification
"""

import os
import sys
import streamlit as st

# Setup page configuration as the very first Streamlit call
st.set_page_config(
    page_title="CardioAI – Explainable ECG Intelligence",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load Custom Healthcare CSS
CSS_PATH = os.path.join(BASE_DIR, 'static', 'css', 'cardioai.css')
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import Pages
from pages_ui.home import render_home_page
from pages_ui.ecg_analysis import render_ecg_analysis_page
from pages_ui.prediction_results import render_prediction_results_page
from pages_ui.explainable_ai import render_explainable_ai_page
from pages_ui.model_performance import render_model_performance_page
from pages_ui.about_project import render_about_project_page
from models.trainer import check_keras_model_status

# Initialize Session State
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Home"

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0; border-bottom:1px solid rgba(0,229,255,0.15); margin-bottom:1.5rem;">
      <div style="font-size:2.4rem;">🫀</div>
      <div style="font-weight:700; font-size:1.15rem; color:#ffffff; letter-spacing:-0.01em;">CardioAI</div>
      <div style="font-size:0.75rem; color:#00e5ff; text-transform:uppercase; letter-spacing:0.06em;">Explainable ECG Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Home",
        "ECG Analysis",
        "Prediction Results",
        "Explainable AI",
        "Model Performance",
        "About Project"
    ]

    icons = {
        "Home": "🏠",
        "ECG Analysis": "📈",
        "Prediction Results": "🎯",
        "Explainable AI": "👁️",
        "Model Performance": "📊",
        "About Project": "📖"
    }

    selected = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in nav_options else 0,
        format_func=lambda x: f"{icons.get(x, '')} {x}",
        label_visibility="collapsed"
    )

    if selected != st.session_state["nav_selection"]:
        st.session_state["nav_selection"] = selected
        st.rerun()

    st.markdown("---")
    
    # Model Status Badge in Sidebar
    status = check_keras_model_status()
    if status['is_trained']:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:0.75rem; font-size:0.75rem;">
          <div style="color:#34d399; font-weight:600; display:flex; align-items:center; gap:0.35rem;">
            <span>🟢</span> 1D CNN + BiLSTM
          </div>
          <div style="color:#94a3b8; margin-top:0.25rem;">PTB-XL Checkpoint Active<br>Params: <b>155,018</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); border-radius:8px; padding:0.75rem; font-size:0.75rem;">
          <div style="color:#fbbf24; font-weight:600; display:flex; align-items:center; gap:0.35rem;">
            <span>🟡</span> Training Required
          </div>
          <div style="color:#94a3b8; margin-top:0.25rem;">Visit Model Performance to train</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#64748b; text-align:center;">
      CardioAI &bull; Academic Research Edition<br>
      PhysioNet PTB-XL Benchmark
    </div>
    """, unsafe_allow_html=True)

# Main Page Routing
current_page = st.session_state["nav_selection"]

if current_page == "Home":
    render_home_page()
elif current_page == "ECG Analysis":
    render_ecg_analysis_page()
elif current_page == "Prediction Results":
    render_prediction_results_page()
elif current_page == "Explainable AI":
    render_explainable_ai_page()
elif current_page == "Model Performance":
    render_model_performance_page()
elif current_page == "About Project":
    render_about_project_page()

