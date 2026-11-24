@echo off
title Explainable Deep Learning ECG Arrhythmia Classification
echo =========================================================================
echo   EXPLAINABLE DEEP LEARNING FOR MULTI-LABEL ECG ARRHYTHMIA CLASSIFICATION
echo   College Academic Research Project
echo =========================================================================
echo Starting Python Backend Server...
start "" "http://127.0.0.1:5000"
"C:\Users\sonal\AppData\Local\Programs\Python\Python312\python.exe" D:\ecg\run_server.py
pause

