"""
CardioAI - Explainable AI (XAI) Page
1D Grad-CAM Class Activation Mapping across 12 leads, Lead-wise attribution rankings,
temporal interval localization, and SHAP/Integrated Gradients evidence decomposition.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

from config import LEAD_NAMES, RHYTHM_CLASSES, CLASS_CODES, CLASS_NAMES
from models.trainer import load_keras_model
from xai.gradcam_1d import GradCAM1D_Keras
from xai.shap_explainer import compute_lead_attribution, compute_shap_temporal_attribution


def render_explainable_ai_page():
    st.markdown("""
    <div class="cardioai-header">
      <div class="cardioai-title-wrap">
        <h1><span>👁️</span> Explainable AI (XAI) Attribution Studio</h1>
        <p>1D Grad-CAM & Multi-Lead Saliency for Transparent Electrocardiogram Diagnostics</p>
      </div>
      <div>
        <span class="cardio-badge badge-teal">1D Grad-CAM + SHAP</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if 'clean_ecg' not in st.session_state or not st.session_state.get('analysis_performed', False):
        st.markdown("""
        <div class="cardio-alert alert-info-custom">
          <div style="font-size:1.5rem;">ℹ️</div>
          <div>
            <b>No Active ECG Recording Loaded</b><br>
            Please go to <b>ECG Analysis</b> and click <b>"Analyze ECG"</b> to generate XAI attributions.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬅️ Go to ECG Analysis"):
            st.session_state["nav_selection"] = "ECG Analysis"
            st.rerun()
        return

    clean_ecg = st.session_state['clean_ecg']  # (1000, 12)
    predictions = st.session_state['predictions']
    active_code = st.session_state.get('active_explain_code', predictions[0]['code'])

    # Class Selection Bar: Choose which diagnosis to explain
    st.markdown("### 🎛️ Diagnostic Target Class Selection")
    st.markdown("""
    Select which predicted arrhythmia condition to explain. 
    1D Grad-CAM extracts gradients of that specific output neuron to show which wave morphology triggered that diagnosis:
    """)

    # Pills / Buttons for all classes with their probability
    cols_btn = st.columns(len(predictions))
    for i, p in enumerate(predictions):
        with cols_btn[i]:
            is_active = (p['code'] == active_code)
            label_text = f"**{p['code']}**\n{p['probability']*100:.0f}%"
            if st.button(label_text, key=f"xai_pick_{p['code']}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state['active_explain_code'] = p['code']
                st.rerun()

    active_p = next((p for p in predictions if p['code'] == active_code), predictions[0])
    class_idx = CLASS_CODES.index(active_code)

    st.markdown(f"""
    <div style="padding:0.75rem 1rem; background:rgba(0,119,182,0.12); border-left:4px solid #00e5ff; border-radius:0 8px 8px 0; margin-top:1rem; margin-bottom:1.5rem;">
      <b style="color:#00e5ff; font-size:1rem;">Explaining: {active_p['name']} ({active_p['code']})</b> &bull; Model Probability: <b>{active_p['probability']*100:.1f}%</b><br>
      <span style="font-size:0.83rem; color:#94a3b8;">{active_p['description']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Compute or retrieve Grad-CAM for this class
    heatmaps_cache = st.session_state.get('gradcam_heatmaps', {})
    lead_imp_cache = st.session_state.get('lead_importance', {})
    intervals_cache = st.session_state.get('salient_intervals', {})

    if active_code not in heatmaps_cache:
        model = load_keras_model()
        gradcam = GradCAM1D_Keras(model, target_conv_layer_name="conv1d_stage2_target")
        cam_norm, lead_imp, intervals = gradcam.explain(clean_ecg, class_idx)
        heatmaps_cache[active_code] = cam_norm
        lead_imp_cache[active_code] = lead_imp
        intervals_cache[active_code] = intervals
        st.session_state['gradcam_heatmaps'] = heatmaps_cache
        st.session_state['lead_importance'] = lead_imp_cache
        st.session_state['salient_intervals'] = intervals_cache

    cam_array = heatmaps_cache[active_code]
    lead_importance = lead_imp_cache[active_code]
    salient_intervals = intervals_cache[active_code]

    # Display Controls
    col_lead_sel, col_opacity = st.columns([1, 1])
    with col_lead_sel:
        display_lead = st.selectbox("Inspect Waveform Lead:", LEAD_NAMES, index=1)
    with col_opacity:
        cam_opacity = st.slider("Heatmap Highlight Intensity:", 0.2, 1.0, 0.75, 0.05)

    lead_idx = LEAD_NAMES.index(display_lead)
    lead_signal = clean_ecg[:, lead_idx]
    time_axis = np.linspace(0, 10.0, 1000, endpoint=False)

    # Plotly 1D Grad-CAM Waveform Overlay
    fig_cam = go.Figure()

    # 1. ECG Trace
    fig_cam.add_trace(go.Scatter(
        x=time_axis, y=lead_signal,
        mode='lines',
        line=dict(color='#00e5ff', width=2.4),
        name=f'ECG Lead {display_lead}'
    ))

    # 2. Shaded Grad-CAM Attribution Ribbon
    scaled_cam = cam_array * (np.max(lead_signal) - np.min(lead_signal)) * 0.45 + np.min(lead_signal)
    fig_cam.add_trace(go.Scatter(
        x=time_axis, y=scaled_cam,
        mode='lines',
        line=dict(color='rgba(239, 68, 68, 0.0)'),
        fill='tozeroy',
        fillcolor=f'rgba(239, 68, 68, {cam_opacity * 0.35})',
        name='1D Grad-CAM Attribution Area'
    ))

    fig_cam.update_layout(
        title=dict(text=f"<b>1D Grad-CAM Neural Attribution Overlay &bull; Lead {display_lead}</b> (Target: {active_p['name']})", font=dict(color="#f8fafc", size=14)),
        paper_bgcolor='#0b1528',
        plot_bgcolor='#070c18',
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(showgrid=True, gridcolor='rgba(239,68,68,0.18)', title="Time (seconds)", color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor='rgba(239,68,68,0.18)', title="Normalized Voltage", color="#94a3b8"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_cam, use_container_width=True)

    # Two Columns: Lead-Wise Spatial Importance & Salient Temporal Intervals
    st.markdown("---")
    col_spat, col_temp = st.columns([1, 1], gap="medium")

    with col_spat:
        st.markdown("### 🌐 12-Lead Spatial Importance Ranking")
        st.markdown("Quantifies which anatomical lead axes exhibited the strongest diagnostic gradient sensitivity:")

        sorted_leads = sorted(lead_importance.items(), key=lambda x: x[1], reverse=True)
        l_names = [x[0] for x in sorted_leads]
        l_scores = [x[1] for x in sorted_leads]
        l_colors = ['#00e5ff' if s >= 12.0 else '#0077b6' if s >= 8.0 else '#334155' for s in l_scores]

        fig_leads = go.Figure(go.Bar(
            x=l_scores,
            y=l_names,
            orientation='h',
            marker=dict(color=l_colors),
            text=[f"{s:.1f}%" for s in l_scores],
            textposition='outside'
        ))
        fig_leads.update_layout(
            paper_bgcolor='#0b1528',
            plot_bgcolor='#070c18',
            height=340,
            margin=dict(l=10, r=40, t=20, b=20),
            xaxis=dict(title="Gradient Attribution Share (%)", color="#94a3b8", gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(autorange="reversed", color="#f8fafc")
        )
        st.plotly_chart(fig_leads, use_container_width=True)

    with col_temp:
        st.markdown("### ⏱️ Top Salient Temporal Intervals")
        st.markdown("Exact millisecond windows where the neural network detected definitive morphological anomalies:")

        if salient_intervals:
            for idx, item in enumerate(salient_intervals):
                st.markdown(f"""
                <div style="padding:0.75rem; border:1px solid rgba(0,229,255,0.2); border-radius:8px; margin-bottom:0.6rem; background:rgba(15,28,52,0.85);">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="color:#00e5ff; font-size:0.9rem;">Interval #{idx+1}: {item['start_time_sec']:.2f}s &ndash; {item['end_time_sec']:.2f}s</b>
                    <span class="cardio-badge badge-red">Peak Salience: {item['peak_attribution']*100:.1f}%</span>
                  </div>
                  <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.25rem;">
                    Duration: <b>{item['duration_ms']} ms</b> &bull; Samples: {item['start_sample']} to {item['end_sample']}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No localized interval exceeded the 0.45 peak threshold (diffuse baseline activation).")

    # SHAP Evidence Decomposition Section
    st.markdown("---")
    st.markdown("### ⚖️ SHAP / Integrated Gradients: Positive vs Negative Evidence")
    st.markdown("""
    Decomposes signal segments into **positive evidence** (driving diagnosis *up*) versus **negative evidence** (arguing *against* diagnosis):
    """)

    shap_res = compute_shap_temporal_attribution(load_keras_model(), clean_ecg, class_idx)
    fig_shap = go.Figure()
    fig_shap.add_trace(go.Scatter(
        x=time_axis, y=shap_res['pos_evidence'],
        mode='lines',
        line=dict(color='#10b981', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.25)',
        name='Positive Attribution (Supports Condition)'
    ))
    fig_shap.add_trace(go.Scatter(
        x=time_axis, y=shap_res['neg_evidence'],
        mode='lines',
        line=dict(color='#ef4444', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.25)',
        name='Negative Attribution (Contradicts Condition)'
    ))
    fig_shap.update_layout(
        paper_bgcolor='#0b1528',
        plot_bgcolor='#070c18',
        height=260,
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis=dict(title="Time (seconds)", color="#94a3b8", gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(title="SHAP Attribution Value", color="#94a3b8", gridcolor='rgba(255,255,255,0.08)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8", size=10))
    )
    st.plotly_chart(fig_shap, use_container_width=True)

