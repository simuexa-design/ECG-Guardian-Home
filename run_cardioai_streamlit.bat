@echo off
title CardioAI - Explainable ECG Intelligence (Streamlit)
echo =========================================================================
echo   CardioAI - Explainable ECG Intelligence
echo   College Academic Research Project
echo   Hybrid 1D CNN + BiLSTM & 1D Grad-CAM (PTB-XL PhysioNet)
echo =========================================================================
echo Launching Streamlit Healthcare Dashboard...
"C:\Users\sonal\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run D:\ecg\CardioAI_app.py --server.port 8501 --server.headless true
pause

