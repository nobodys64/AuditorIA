@echo off
cd /d "%~dp0"
title AuditorIA Universitaria

echo ===================================================
echo   AuditorIA Universitaria - Entorno Windows
echo ===================================================

python --version
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    pause
    exit /b 1
)

if not exist "venv\Scripts\streamlit.exe" (
    echo Creando entorno virtual e instalando dependencias...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\pip.exe" install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Fallo la instalacion de requirements.txt.
        pause
        exit /b 1
    )
)

set "HF_HOME=%~dp0modelos_cache"
echo Iniciando servidor local en http://localhost:8501 ...
"venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats false
pause