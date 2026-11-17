@echo off
title Iniciando AGD-Sentinel
echo Iniciando el motor predictivo, por favor espere...
cd /d "%~dp0"
rem Portable runtime (motor_python) if present; otherwise the local virtual environment, if any.
if exist motor_python\scripts\env.bat (
    call motor_python\scripts\env.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
rem The deep surrogate runs with NumPy weights (models\dl_lstm.npz): no deep-learning framework required.
streamlit run app.py --server.port 8505