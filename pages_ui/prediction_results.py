"""
CardioAI - Prediction Results Page
Displays actual multi-label model predictions, probability distribution bars,
interactive classification threshold slider, risk tier, and preprocessing audit log.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import RHYTHM_CLASSES, CLASS_CODES, CLASS_NAMES


def render_prediction_results_page():
    st.markdown("""
    <div class="cardioai-header">
      <div class="cardioai-title-wrap">
        <h1><span>🎯</span> Multi-Label Arrhythmia Prediction Results</h1>
        <p>Independent sigmoid probabilities per arrhythmia class with dynamic threshold calibration</p>
      </div>
      <div>
        <span class="cardio-badge badge-teal">Sigmoid Multi-Label</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if 'predictions' not in st.session_state or not st.session_state.get('analysis_performed', False):
        st.markdown("""
        <div class="cardio-alert alert-info-custom">
          <div style="font-size:1.5rem;">ℹ️</div>
          <div>
            <b>No Active Analysis Found</b><br>
            Please navigate to the <b>ECG Analysis</b> page, select or upload an ECG recording, and click <b>"Analyze ECG"</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬅️ Go to ECG Analysis"):
            st.session_state["nav_selection"] = "ECG Analysis"
            st.rerun()
        return

    predictions = st.session_state['predictions']
    case_name = st.session_state.get('active_case_name', 'Patient Recording')
    true_labels = st.session_state.get('active_true_labels', [])

    st.markdown(f"### 📋 Diagnostic Summary: **{case_name}**")
    if true_labels:
        st.markdown(f"**Clinical Reference Labels:** `{' | '.join(true_labels)}`")

    # Threshold Slider
    st.markdown("<br>", unsafe_allow_html=True)
    c_thresh, c_desc = st.columns([1, 2])
    with c_thresh:
        threshold = st.slider("Classification Probability Threshold:", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    with c_desc:
        st.markdown(f"""
        <p style="font-size:0.83rem; color:#94a3b8; margin-top:1.5rem;">
          In multi-label learning, each arrhythmia neuron outputs an independent sigmoid probability. 
          Conditions with confidence &ge; <b>{threshold:.2f}</b> are classified as <b>DETECTED</b>.
        </p>
        """, unsafe_allow_html=True)

    # Recalculate detections
    detected_arrhythmias = []
    for p in predictions:
        p['is_detected'] = (p['probability'] >= threshold)
        if p['is_detected']:
            detected_arrhythmias.append(p)

    # Clinical Risk Stratification
    st.markdown("---")
    risk_col, detected_col = st.columns([1, 1.8], gap="medium")

    with risk_col:
        # Determine Risk
        detected_codes = [d['code'] for d in detected_arrhythmias]
        if 'AFIB' in detected_codes and 'STACH' in detected_codes:
            risk_tier = 'High Risk'
            risk_badge = 'badge-red'
            risk_desc = 'Atrial Fibrillation with Rapid Ventricular Response (RVR). Elevated stroke and decompensation risk.'
        elif 'BIGU' in detected_codes or 'PVC' in detected_codes:
            risk_tier = 'Moderate Risk'
            risk_badge = 'badge-amber'
            risk_desc = 'Ventricular ectopic activity observed. Risk of progression to complex ventricular ectopy.'
        elif len(detected_arrhythmias) >= 2:
            risk_tier = 'Moderate Risk'
            risk_badge = 'badge-amber'
            risk_desc = f'Multi-label co-occurrence: {len(detected_arrhythmias)} concurrent conditions identified.'
        elif len(detected_arrhythmias) == 1 and detected_codes[0] != 'SR':
            risk_tier = 'Moderate Risk'
            risk_badge = 'badge-amber'
            risk_desc = f'Isolated arrhythmia: {detected_arrhythmias[0]["name"]}.'
        else:
            risk_tier = 'Low Risk'
            risk_badge = 'badge-emerald'
            risk_desc = 'Physiological sinus rhythm without acute ectopic or reentrant rhythm anomalies.'

        st.markdown(f"""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🛡️</span> Risk Stratification
          </div>
          <div style="margin-bottom:0.75rem;">
            <span class="cardio-badge {risk_badge}" style="font-size:0.9rem; padding:0.35rem 0.9rem;">{risk_tier}</span>
          </div>
          <p style="font-size:0.83rem; color:#f8fafc; line-height:1.5;">{risk_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    with detected_col:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>🩺</span> Identified Positive Diagnoses
          </div>
        """, unsafe_allow_html=True)
        if detected_arrhythmias:
            for d in detected_arrhythmias:
                st.markdown(f"""
                <div style="padding:0.6rem; border:1px solid rgba(0,229,255,0.2); border-radius:8px; margin-bottom:0.5rem; background:rgba(0,119,182,0.1);">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="color:#00e5ff; font-size:0.95rem;">{d['name']} ({d['code']})</b>
                    <span class="cardio-badge badge-teal">Confidence: {d['probability']*100:.1f}%</span>
                  </div>
                  <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.25rem;">{d['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#94a3b8; font-size:0.83rem;'>No conditions exceeded the selected probability threshold.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Multi-Label Probability Distribution Chart
    st.markdown("### 📊 Multi-Label Prediction Probability Distribution")

    codes = [p['code'] for p in predictions]
    probs = [p['probability'] * 100 for p in predictions]
    names = [p['name'] for p in predictions]
    colors = ['#00e5ff' if p['is_detected'] else '#475569' for p in predictions]

    fig_probs = go.Figure(go.Bar(
        x=probs,
        y=[f"{n} ({c})" for n, c in zip(names, codes)],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(0,229,255,0.4)', width=1)),
        text=[f"{v:.1f}%" for v in probs],
        textposition='outside'
    ))
    fig_probs.add_vline(x=threshold * 100, line_dash="dash", line_color="#ef4444", annotation_text=f"Threshold ({threshold:.2f})", annotation_position="top right")
    fig_probs.update_layout(
        paper_bgcolor='#0b1528',
        plot_bgcolor='#070c18',
        height=420,
        margin=dict(l=20, r=40, t=30, b=30),
        xaxis=dict(title="Probability (%)", range=[0, 105], gridcolor='rgba(255,255,255,0.08)', color="#94a3b8"),
        yaxis=dict(autorange="reversed", color="#f8fafc", tickfont=dict(size=11)),
        title=dict(text="<b>Multi-Label Output Layer Neurons (Sigmoid Activations)</b>", font=dict(color="#f8fafc", size=13))
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    # Preprocessing Audit Trail
    if 'prep_audit' in st.session_state:
        with st.expander("🔍 View Preprocessing Pipeline Audit Trail"):
            st.markdown("Operations applied to ensure exact consistency with training data:")
            for item in st.session_state['prep_audit']:
                st.markdown(f"- `{item}`")

    # Navigation to XAI
    st.markdown("<br>", unsafe_allow_html=True)
    c_xai, _ = st.columns([1.2, 1])
    with c_xai:
        if st.button("👁️ Deep Dive into Explainable AI (1D Grad-CAM)", type="primary", use_container_width=True):
            st.session_state["nav_selection"] = "Explainable AI"
            st.rerun()

