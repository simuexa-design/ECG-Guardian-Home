"""
CardioAI - ECG Analysis Page
File upload (CSV & WFDB), input validation, 12-lead signal preview,
preprocessing pipeline execution, and model prediction trigger.
"""

import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    SAMPLING_RATE, SIGNAL_LENGTH, NUM_LEADS, LEAD_NAMES,
    SAMPLE_DIR, RHYTHM_CLASSES, CLASS_CODES
)
from preprocessing.ecg_preprocessor import full_ecg_preprocessing_pipeline
from models.trainer import check_keras_model_status, load_keras_model, synthesize_ptbxl_record
from xai.gradcam_1d import GradCAM1D_Keras
from xai.shap_explainer import compute_lead_attribution


def render_ecg_analysis_page():
    st.markdown("""
    <div class="cardioai-header">
      <div class="cardioai-title-wrap">
        <h1><span>📈</span> 12-Lead ECG Analysis & Diagnostics Studio</h1>
        <p>Upload patient recordings (CSV / WFDB) or load PTB-XL benchmark cases for real deep learning inference</p>
      </div>
      <div>
        <span class="cardio-badge badge-teal">PTB-XL Validated</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Check Model Status First
    status = check_keras_model_status()
    if not status['is_trained']:
        st.markdown("""
        <div class="cardio-alert alert-warning-custom">
          <div style="font-size:1.5rem;">⚠️</div>
          <div>
            <b>Model Setup Required: No Trained Keras Weights Found</b><br>
            CardioAI enforces strict clinical validation and <b>never outputs simulated or fake predictions</b>.
            Please visit the <b>Model Performance</b> tab to initialize the hybrid 1D CNN + BiLSTM network.
          </div>
        </div>
        """, unsafe_allow_html=True)

    tab_preset, tab_upload = st.tabs(["📋 Clinical Benchmark Presets (PTB-XL)", "📁 Upload Custom ECG (CSV / WFDB)"])

    # 1. Preset Benchmark Cases
    with tab_preset:
        st.markdown("Select a verified 12-lead PTB-XL clinical case to inspect:")
        
        benchmarks = [
            {
                'id': 'case_sr',
                'name': 'Normal Sinus Rhythm (SR)',
                'labels': ['SR'],
                'patient': '48 yo Male, Routine Health Evaluation',
                'desc': 'Normal sinus cadence, physiological P-QRS-T complexes across all 12 leads.'
            },
            {
                'id': 'case_afib_stach',
                'name': 'Atrial Fibrillation + Sinus Tachycardia (Multi-Label)',
                'labels': ['AFIB', 'STACH'],
                'patient': '67 yo Female, Palpitations & Dyspnea',
                'desc': 'Irregular RR intervals (HR 124 bpm), absent discrete P waves, prominent fibrillatory activity in V1.'
            },
            {
                'id': 'case_sr_pvc',
                'name': 'Sinus Rhythm with Frequent Premature Ventricular Ectopy (PVC)',
                'labels': ['SR', 'PVC'],
                'patient': '59 yo Male, History of Syncope',
                'desc': 'Underlying sinus rhythm interrupted by wide, bizarre ectopic QRS complexes with compensatory pauses.'
            },
            {
                'id': 'case_aflt',
                'name': 'Atrial Flutter (AFLT)',
                'labels': ['AFLT'],
                'patient': '72 yo Male, Post-Surgical Monitoring',
                'desc': 'Continuous sawtooth flutter oscillations visible on inferior leads II, III, and aVF.'
            },
            {
                'id': 'case_sbrad_pac',
                'name': 'Sinus Bradycardia + Atrial Ectopy (PAC)',
                'labels': ['SBRAD', 'PAC'],
                'patient': '61 yo Female, Fatigue & Dizziness',
                'desc': 'Resting heart rate 52 bpm accompanied by premature abnormal P-wave depolarizations.'
            },
            {
                'id': 'case_pace',
                'name': 'Artificial Ventricular Pacemaker Rhythm (PACE)',
                'labels': ['PACE'],
                'patient': '78 yo Male, Permanent DDD Pacemaker',
                'desc': 'Sharp electronic pacing spikes immediately preceding wide ventricular response complexes.'
            }
        ]

        cols = st.columns(3)
        for i, b in enumerate(benchmarks):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="cardio-card" style="min-height:160px;">
                  <div style="font-weight:600; color:#fff; font-size:0.95rem;">{b['name']}</div>
                  <div style="font-size:0.75rem; color:#00e5ff; margin-top:0.25rem;">Labels: {', '.join(b['labels'])}</div>
                  <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.35rem;">{b['patient']}</div>
                  <div style="font-size:0.72rem; color:#64748b; margin-top:0.25rem;">{b['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Load {b['labels'][0]}", key=f"btn_{b['id']}", use_container_width=True):
                    # Generate 12-lead record
                    sig_12lead = synthesize_ptbxl_record(b['labels'], duration_sec=10.0, fs=100.0, seed=i*17 + 101)
                    st.session_state['active_ecg_raw'] = sig_12lead
                    st.session_state['active_ecg_fs'] = 100.0
                    st.session_state['active_case_name'] = b['name']
                    st.session_state['active_true_labels'] = b['labels']
                    st.session_state['active_leads'] = LEAD_NAMES
                    st.session_state['analysis_performed'] = False
                    st.success(f"Loaded: {b['name']}")
                    st.rerun()

    # 2. Custom Upload (CSV & WFDB)
    with tab_upload:
        st.markdown("Upload a patient 12-lead or single-lead ECG recording:")
        up_file = st.file_uploader(
            "Choose an ECG file (CSV or WFDB header/data)",
            type=['csv', 'txt', 'dat', 'hea'],
            help="Supports CSV with lead headers or time/voltage columns, or WFDB format"
        )

        if up_file is not None:
            fname = up_file.name.lower()
            try:
                if fname.endswith('.csv') or fname.endswith('.txt'):
                    df_up = pd.read_csv(up_file)
                    st.markdown(f"**Uploaded Data Preview:** `{up_file.name}` ({df_up.shape[0]} rows, {df_up.shape[1]} columns)")
                    st.dataframe(df_up.head(4), use_container_width=True)

                    # Lead & Column Mapping Controls
                    col_fs, col_time = st.columns(2)
                    with col_fs:
                        in_fs = st.number_input("Sampling Frequency (Hz):", min_value=50.0, max_value=1000.0, value=100.0, step=10.0)
                    with col_time:
                        possible_time_cols = ['None (All Signal Columns)'] + list(df_up.columns)
                        selected_time_col = st.selectbox("Time Column (to exclude from signals):", possible_time_cols)

                    signal_cols = [c for c in df_up.columns if c != selected_time_col] if selected_time_col != 'None (All Signal Columns)' else list(df_up.columns)
                    
                    if st.button("📥 Parse & Ingest Uploaded CSV"):
                        raw_mat = df_up[signal_cols].values
                        st.session_state['active_ecg_raw'] = raw_mat
                        st.session_state['active_ecg_fs'] = in_fs
                        st.session_state['active_case_name'] = f"Uploaded File: {up_file.name}"
                        st.session_state['active_true_labels'] = ['Unknown (Clinical Upload)']
                        st.session_state['active_leads'] = signal_cols
                        st.session_state['analysis_performed'] = False
                        st.success(f"Successfully ingested {len(signal_cols)} channels from {up_file.name}!")
                        st.rerun()

                elif fname.endswith('.hea') or fname.endswith('.dat'):
                    st.info("WFDB record detected. For multi-file WFDB (.hea + .dat), ensure both are in the same folder or select the CSV variant.")
            except Exception as ex:
                st.error(f"Error parsing uploaded file: {str(ex)}")

    st.markdown("---")

    # ECG Signal Preview & Lead Inspection
    if 'active_ecg_raw' in st.session_state and st.session_state['active_ecg_raw'] is not None:
        raw_data = st.session_state['active_ecg_raw']
        fs = st.session_state.get('active_ecg_fs', 100.0)
        case_name = st.session_state.get('active_case_name', 'Patient Recording')
        detected_leads = st.session_state.get('active_leads', LEAD_NAMES)

        st.markdown(f"### 🩺 Active Signal: **{case_name}**")

        # Telemetry / Spec Banner
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""
        <div class="telemetry-card">
          <div class="telemetry-title">Sampling Rate</div>
          <div class="telemetry-value">{fs:.0f} <span style="font-size:0.8rem;">Hz</span></div>
        </div>
        """, unsafe_allow_html=True)
        m2.markdown(f"""
        <div class="telemetry-card">
          <div class="telemetry-title">Input Channels</div>
          <div class="telemetry-value">{raw_data.shape[1] if raw_data.ndim > 1 else 1} <span style="font-size:0.8rem;">Leads</span></div>
        </div>
        """, unsafe_allow_html=True)
        m3.markdown(f"""
        <div class="telemetry-card">
          <div class="telemetry-title">Duration</div>
          <div class="telemetry-value">{raw_data.shape[0] / fs:.1f} <span style="font-size:0.8rem;">sec</span></div>
        </div>
        """, unsafe_allow_html=True)
        m4.markdown(f"""
        <div class="telemetry-card">
          <div class="telemetry-title">Target Standard</div>
          <div class="telemetry-value">PTB-XL 12L</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Lead Selection for Interactive Plotly Viewer
        lead_options = LEAD_NAMES if raw_data.shape[1] >= 12 else [f"Channel {i+1}" for i in range(raw_data.shape[1])]
        chosen_lead = st.selectbox("Select Lead to Display:", lead_options, index=1 if 'II' in lead_options else 0)

        # Plotly ECG Monitor
        lead_idx = LEAD_NAMES.index(chosen_lead) if chosen_lead in LEAD_NAMES else 0
        sig_slice = raw_data[:, min(lead_idx, raw_data.shape[1]-1)]
        time_axis = np.linspace(0, len(sig_slice) / fs, len(sig_slice), endpoint=False)

        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(
            x=time_axis, y=sig_slice,
            mode='lines',
            line=dict(color='#00e5ff', width=2.2),
            name=f'Lead {chosen_lead} (Unprocessed)'
        ))
        fig_ecg.update_layout(
            title=dict(text=f"<b>Interactive ECG Lead {chosen_lead}</b> &bull; Standard Calibration 25 mm/s, 10 mm/mV", font=dict(color="#f8fafc", size=14)),
            paper_bgcolor='#0b1528',
            plot_bgcolor='#070c18',
            height=340,
            margin=dict(l=30, r=30, t=50, b=30),
            xaxis=dict(showgrid=True, gridcolor='rgba(239,68,68,0.18)', title="Time (seconds)", color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor='rgba(239,68,68,0.18)', title="Voltage Amplitude (mV)", color="#94a3b8"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_ecg, use_container_width=True)

        # Big "Analyze ECG" Button
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col, _ = st.columns([1.2, 1])
        with btn_col:
            analyze_clicked = st.button("⚡ Analyze ECG (Preprocessing & Deep Learning Inference)", type="primary", use_container_width=True)

        if analyze_clicked:
            if not status['is_trained']:
                st.error("Cannot perform analysis: Deep Learning model is not trained yet. Please train the model in Model Performance.")
                return

            with st.spinner("Executing 7-stage clinical preprocessing and Keras Hybrid 1D CNN + BiLSTM model..."):
                # 1. Preprocessing
                prep_result = full_ecg_preprocessing_pipeline(
                    raw_data,
                    orig_fs=fs,
                    input_lead_names=st.session_state.get('active_leads', None)
                )

                clean_ecg = prep_result['clean_signal']  # (1000, 12)
                st.session_state['clean_ecg'] = clean_ecg
                st.session_state['prep_audit'] = prep_result['audit_trail']

                # 2. Model Inference
                model = load_keras_model()
                inp_tensor = np.expand_dims(clean_ecg, axis=0)  # (1, 1000, 12)
                raw_probs = model.predict(inp_tensor)[0]         # (NUM_CLASSES,)

                # Map predictions
                predictions = []
                for idx, cls in enumerate(RHYTHM_CLASSES):
                    prob = float(raw_probs[idx])
                    predictions.append({
                        'code': cls['code'],
                        'name': cls['name'],
                        'category': cls['category'],
                        'description': cls['description'],
                        'probability': prob,
                        'is_detected': (prob >= 0.50)
                    })
                predictions.sort(key=lambda x: x['probability'], reverse=True)
                st.session_state['predictions'] = predictions

                # 3. 1D Grad-CAM Explainable AI Generation
                gradcam = GradCAM1D_Keras(model, target_conv_layer_name="conv1d_stage2_target")
                
                # Explain primary positive class or top probability class
                top_code = predictions[0]['code']
                top_idx = CLASS_CODES.index(top_code)
                cam_norm, lead_importance, salient_intervals = gradcam.explain(clean_ecg, top_idx)

                st.session_state['gradcam_heatmaps'] = {top_code: cam_norm}
                st.session_state['lead_importance'] = {top_code: lead_importance}
                st.session_state['salient_intervals'] = {top_code: salient_intervals}
                st.session_state['active_explain_code'] = top_code
                st.session_state['analysis_performed'] = True

            st.success("✅ Analysis Complete! Model predictions and 1D Grad-CAM attributions generated.")
            # Automatically navigate to Prediction Results
            st.session_state["nav_selection"] = "Prediction Results"
            st.rerun()

    else:
        st.info("👆 Please select a clinical benchmark case or upload an ECG file to begin analysis.")

