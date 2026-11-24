"""
CardioAI - Home Page
Landing page with project introduction, ECG illustrations, multi-label & XAI explanations,
informational cards, clinical workflow, and tech stack details.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from config import RHYTHM_CLASSES


def render_home_page():
    # Header Banner
    st.markdown("""
    <div class="cardioai-header">
      <div class="cardioai-title-wrap">
        <h1><span>&#10084;&#65039;</span> CardioAI – Explainable ECG Intelligence</h1>
        <p>Explainable Deep Learning for Multi-Label ECG Arrhythmia Classification &bull; College Academic Research Project</p>
      </div>
      <div>
        <span class="cardio-badge badge-teal">&#9889; PTB-XL 12-Lead Standard</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Section: Two Columns
    col_intro, col_illus = st.columns([1.1, 0.9], gap="large")

    with col_intro:
        st.markdown("""
        ### AI-Powered Multi-Label Cardiac Rhythm Intelligence
        
        Cardiovascular diseases remain the leading cause of global mortality. While standard automated ECG machines 
        rely on rigid heuristic rule trees or opaque black-box neural networks, **CardioAI** bridges the clinical trust gap 
        by combining **Hybrid 1D Convolutional Neural Networks with Bidirectional Long Short-Term Memory (BiLSTM)** 
        and **1D Gradient-Weighted Class Activation Mapping (Grad-CAM)**.
        
        Designed for academic research and clinical decision support, the platform interprets full **12-lead ECG recordings**, 
        identifying simultaneous, co-occurring arrhythmias while providing sample-level visual explanations.
        """)

        # Call-to-action button
        c_btn, _ = st.columns([1, 1])
        with c_btn:
            if st.button("🚀 Start ECG Analysis", type="primary", use_container_width=True):
                st.session_state["nav_selection"] = "ECG Analysis"
                st.rerun()

    with col_illus:
        # Interactive Medical ECG Waveform Illustration using Plotly
        t = np.linspace(0, 2.5, 250)
        # Normal Sinus Beat + Ventricular Ectopic Beat
        ecg_sample = (
            0.15 * np.exp(-((t - 0.25) ** 2) / (2 * 0.035 ** 2)) -
            0.15 * np.exp(-((t - 0.38) ** 2) / (2 * 0.015 ** 2)) +
            1.35 * np.exp(-((t - 0.42) ** 2) / (2 * 0.022 ** 2)) -
            0.30 * np.exp(-((t - 0.46) ** 2) / (2 * 0.018 ** 2)) +
            0.35 * np.exp(-((t - 0.70) ** 2) / (2 * 0.060 ** 2)) +
            # Ectopic PVC beat at 1.4s
            1.55 * np.exp(-((t - 1.40) ** 2) / (2 * 0.065 ** 2)) -
            0.75 * np.exp(-((t - 1.52) ** 2) / (2 * 0.055 ** 2)) -
            0.45 * np.exp(-((t - 1.75) ** 2) / (2 * 0.075 ** 2))
        )
        
        # Attribution color simulation
        heat_color = np.clip(np.exp(-((t - 1.42) ** 2) / 0.05), 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=ecg_sample,
            mode='lines',
            line=dict(color='#00e5ff', width=2.5),
            name='ECG Lead II'
        ))
        fig.add_trace(go.Scatter(
            x=t, y=heat_color * 1.6,
            mode='lines',
            line=dict(color='rgba(239, 68, 68, 0.4)', width=0),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.22)',
            name='1D Grad-CAM Salience (PVC)'
        ))
        fig.update_layout(
            title=dict(text="<b>Clinical 12-Lead Morphology & 1D Grad-CAM Overlay</b>", font=dict(color="#f8fafc", size=13)),
            paper_bgcolor='rgba(15, 28, 52, 0.7)',
            plot_bgcolor='rgba(7, 12, 24, 0.9)',
            margin=dict(l=20, r=20, t=40, b=20),
            height=260,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="Time (seconds)", color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="Amplitude (mV)", color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8", size=10))
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Core Scientific Concepts (Arrhythmia, Multi-Label, XAI)
    st.markdown("### 🔬 Academic & Clinical Foundations")
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🫀</span> What is Cardiac Arrhythmia?
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Cardiac arrhythmias represent disruptions in the heart's normal electrical conduction system. 
            Manifesting as rates that are too fast (<b>tachycardia</b>), too slow (<b>bradycardia</b>), 
            or irregular (<b>atrial fibrillation</b>), they can cause hemodynamic instability, thromboembolic stroke, or cardiac arrest.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🏷️</span> Why Multi-Label Classification?
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Real-world patients rarely exhibit only one isolated anomaly. A single 10-second ECG recording 
            can simultaneously display <b>Sinus Tachycardia + Atrial Fibrillation</b> or <b>Pacemaker rhythm + PVCs</b>. 
            Multi-label learning models independent sigmoid probabilities rather than mutually exclusive softmax classes.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>👁️</span> What is Explainable AI (XAI)?
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Deep neural networks are traditionally black boxes. <b>1D Grad-CAM</b> (Gradient-Weighted Class Activation Mapping) 
            computes gradients of the target diagnosis with respect to convolutional feature maps, projecting salience heatmaps 
            directly onto the ECG waveform to verify P-wave absence, QRS widening, or ectopic timings.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 5 Informational Cards Required by Specification
    st.markdown("### 📋 Core Platform Capabilities")
    
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>📈</span> 1. ECG Signal Processing
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Implements zero-phase forward-backward <b>Butterworth bandpass filtering</b> (0.5 – 45 Hz) to eliminate baseline wander 
            and high-frequency EMG tremor. Features an <b>IIR notch filter</b> for 50 Hz powerline hum and lead-wise Z-score normalization.
          </p>
        </div>

        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🧠</span> 3. Hybrid Deep Learning (CNN + BiLSTM)
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            <b>Conv1D layers (64 & 128 filters)</b> extract fine-grained morphological deflections across 12 leads, 
            while a <b>Bidirectional LSTM</b> captures long-range temporal beat transitions and heart rate rhythm cadence across both temporal directions.
          </p>
        </div>

        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>📊</span> 5. Rigorous Model Evaluation
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Evaluated on official patient-wise folds of the <b>PTB-XL PhysioNet benchmark</b> (Folds 1–8 Train, Fold 9 Val, Fold 10 Test). 
            Reports Macro/Micro ROC-AUC, Macro F1, Subset Exact Match, and Hamming Loss without data leakage.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🎯</span> 2. Multi-Label Arrhythmia Detection
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Supports 10 clinical rhythm categories from the PTB-XL database, including Sinus Rhythm (SR), Atrial Fibrillation (AFIB), 
            Sinus Tachycardia (STACH), Sinus Bradycardia (SBRAD), Atrial Flutter (AFLT), SVT, Ventricular Ectopy (PVC/BIGU), and Pacemaker rhythms.
          </p>
        </div>

        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🔍</span> 4. 1D Grad-CAM & SHAP Explainability
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Provides condition-specific attribution heatmaps across all 12 leads. Users can toggle which predicted arrhythmia to explain, 
            inspecting localized temporal intervals (milliseconds) and lead importance rankings that triggered the decision.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Workflow Section
    st.markdown("### 🔄 Clinical & Computational Workflow")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
        <div class="workflow-step">
          <div class="workflow-num">1</div>
          <div>
            <b style="color:#fff; font-size:0.9rem;">ECG Ingestion</b>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.25rem;">Upload CSV, WFDB (.dat/.hea), or select clinical benchmark presets.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="workflow-step">
          <div class="workflow-num">2</div>
          <div>
            <b style="color:#fff; font-size:0.9rem;">Preprocessing</b>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.25rem;">Bandpass 0.5–45Hz, 50Hz notch, 12-lead standardization & 100Hz resampling.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="workflow-step">
          <div class="workflow-num">3</div>
          <div>
            <b style="color:#fff; font-size:0.9rem;">CNN + BiLSTM</b>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.25rem;">Multi-scale convolutions and bidirectional recurrent sequence modeling.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown("""
        <div class="workflow-step">
          <div class="workflow-num">4</div>
          <div>
            <b style="color:#fff; font-size:0.9rem;">1D Grad-CAM XAI</b>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.25rem;">Gradient attribution maps, lead importance, and cardiological diagnostic audit.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Technologies Used
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.5rem;">
      <span class="cardio-badge badge-teal">Python 3.12</span>
      <span class="cardio-badge badge-teal">TensorFlow 2.21</span>
      <span class="cardio-badge badge-teal">Keras 3.15</span>
      <span class="cardio-badge badge-teal">Streamlit 1.67</span>
      <span class="cardio-badge badge-teal">WFDB (PhysioNet)</span>
      <span class="cardio-badge badge-teal">Plotly & Matplotlib</span>
      <span class="cardio-badge badge-teal">SciPy Signal</span>
      <span class="cardio-badge badge-teal">Scikit-learn</span>
      <span class="cardio-badge badge-teal">PTB-XL Dataset</span>
    </div>
    """, unsafe_allow_html=True)

