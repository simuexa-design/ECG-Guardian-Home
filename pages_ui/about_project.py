"""
CardioAI - About Project Page
Comprehensive academic project documentation, layer-by-layer architectural justification,
loss function and XAI mathematical formulations, and PhysioNet references.
"""

import streamlit as st
from config import MODEL_CONFIG, NUM_CLASSES, SIGNAL_LENGTH, NUM_LEADS


def render_about_project_page():
    st.markdown("""
    <div class="cardioai-header">
      <div class="cardioai-title-wrap">
        <h1><span>📖</span> Academic Project & Methodology Documentation</h1>
        <p>Explainable Deep Learning for Multi-Label ECG Arrhythmia Classification &bull; Technical Specifications</p>
      </div>
      <div>
        <span class="cardio-badge badge-teal">Academic Project</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Project Title & Abstract
    st.markdown("""
    ### 🎓 Project Title
    **"Explainable Deep Learning for Multi-Label ECG Arrhythmia Classification"**
    
    #### 📝 Abstract & Clinical Motivation
    Automated electrocardiogram interpretation is a foundational diagnostic tool in modern cardiology. 
    However, conventional clinical AI applications suffer from two critical limitations:
    1. **Single-label assumption**: Many legacy models assume a single mutually exclusive diagnosis, 
       whereas real clinical patients frequently suffer from concurrent pathologies (e.g., Atrial Fibrillation co-occurring with Sinus Tachycardia or Pacemaker rhythm with Ventricular Ectopy).
    2. **Black-box opacity**: Standard deep neural networks provide raw probability scores without verifying 
       whether the decision was driven by genuine electrophysiological markers or unmodeled noise artifacts.
       
    **CardioAI** addresses these limitations by developing a **Hybrid 1D CNN + Bidirectional LSTM** network trained with 
    multi-label Binary Cross-Entropy loss and evaluated on the clinical **PhysioNet PTB-XL benchmark** across official patient-wise folds. 
    It integrates **1D Grad-CAM** and **SHAP** gradient attributions to illuminate the exact temporal segments and lead axes driving each prediction.
    """)

    st.markdown("---")

    # 2. Deep Learning Architecture: Layer-by-Layer Justification
    st.markdown("### 🧠 Deep Learning Architecture & Layer Rationale")
    st.markdown("The hybrid network combines local spatial-morphological feature extraction with bidirectional temporal sequence learning:")

    st.markdown("""
    | Layer Name | Type | Specifications | Architectural & Clinical Rationale |
    | :--- | :--- | :--- | :--- |
    | **`ecg_12lead_input`** | Input | `(1000, 12)` | Standardized 10-second 12-lead ECG recorded at 100 Hz. Ingests all 12 clinical lead axes simultaneously. |
    | **`conv1d_stage1`** | Conv1D | 64 Filters, Kernel=7 | Extracts low-level morphological wave deflections (P-wave, QRS spike, ST-elevation) across leads. |
    | **`bn_stage1` & `relu`** | Normalization | Batch Normalization | Stabilizes internal covariate shift and accelerates gradient propagation. |
    | **`maxpool_stage1`** | Max Pooling | Pool Size=2 | Downsamples temporal resolution by 50% (500 time steps) to impart temporal translation invariance. |
    | **`conv1d_stage2_target`** | Conv1D | 128 Filters, Kernel=5 | Captures complex multi-lead morphological patterns. **Designated target layer for 1D Grad-CAM**. |
    | **`bn_stage2` & `relu`** | Normalization | Batch Normalization | Enhances non-linear representational capacity before recurrent temporal aggregation. |
    | **`maxpool_stage2`** | Max Pooling | Pool Size=2 | Further compresses temporal representation to 250 steps, allowing the BiLSTM to operate on beat-level features. |
    | **`bidirectional_lstm`** | BiLSTM | 64 Units (128 total) | Models rhythm cadence and long-range dependencies in **both forward and backward time directions**. Critical for detecting irregular RR variability (AFib) and compensatory pauses. |
    | **`dropout_classifier`**| Dropout | Rate=0.35 | Mitigates neuron co-adaptation and prevents overfitting on dominant baseline classes. |
    | **`dense_embedding`** | Dense | 64 Units, ReLU, L2 reg | Projects bidirectional recurrent state into a compact latent diagnostic feature vector. |
    | **`multilabel_sigmoid`** | Dense | 10 Units, Sigmoid | **Multi-Label Output Layer**. Uses Sigmoid activations to model independent class probabilities (Softmax is strictly avoided). |
    """)

    st.markdown("---")

    # 3. Mathematical Formulations
    st.markdown("### 📐 Mathematical Formulations")
    
    col_math1, col_math2 = st.columns(2, gap="medium")
    with col_math1:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>📉</span> Multi-Label Loss (Binary Cross-Entropy)
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            Because cardiac conditions can co-occur, the network optimizes independent binary cross-entropy across all $C = 10$ classes:
          </p>
          <div style="background:#070c18; padding:0.75rem; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#00e5ff; margin-bottom:0.5rem;">
            L_BCE = - (1/C) * &Sigma; [ y_c * log(&sigma;(z_c)) + (1 - y_c) * log(1 - &sigma;(z_c)) ]
          </div>
          <p style="font-size:0.75rem; color:#64748b;">
            Where &sigma;(z_c) = 1 / (1 + e^(-z_c)) is the sigmoid activation for class c.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_math2:
        st.markdown("""
        <div class="cardio-card">
          <div class="cardio-card-header">
            <span>👁️</span> 1D Grad-CAM Attribution Math
          </div>
          <p style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">
            1D Grad-CAM computes gradients of target score $y^c$ with respect to feature maps $A^k$ of the final conv layer:
          </p>
          <div style="background:#070c18; padding:0.75rem; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#00e5ff; margin-bottom:0.5rem;">
            &alpha;_k^c = (1 / L') * &Sigma;_t (&part;y^c / &part;A_k(t))<br>
            L_GradCAM^c(t) = ReLU( &Sigma;_k &alpha;_k^c * A_k(t) )
          </div>
          <p style="font-size:0.75rem; color:#64748b;">
            Upsampled back to 1000 samples and normalized to [0, 1] for waveform overlay.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Official PhysioNet PTB-XL Dataset Reference
    st.markdown("### 📚 Benchmark Dataset & Clinical References")
    st.markdown("""
    This project is benchmarked against the **PTB-XL ECG Dataset** hosted on PhysioNet:
    
    1. **Wagner, P., Strodthoff, N., Bousseljot, R. D., et al.** (2020).  
       *"PTB-XL, a large publicly available electrocardiography dataset."* **Scientific Data (Nature)**, 7(1), 154.  
       DOI: [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)
       
    2. **Goldberger, A. L., Amaral, L. A., Glass, L., et al.** (2000).  
       *"PhysioNet: Components of a new research resource for complex physiologic signals."* **Circulation**, 101(23), e215-e220.
       
    3. **Selvaraju, R. R., Cogswell, M., Das, A., et al.** (2017).  
       *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization."* **IEEE International Conference on Computer Vision (ICCV)**, pp. 618-626.
       
    4. **Pan, J., & Tompkins, W. J.** (1985).  
       *"A real-time QRS detection algorithm."* **IEEE Transactions on Biomedical Engineering**, 32(3), 230-236.
    """)

