@echo off
cd /d "%~dp0"
set HF_HOME=%~dp0modelos_cache

echo ===================================================
echo          AuditorIA - Sistema de Auditoria
echo ===================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado en PATH. Instale Python 3.10 o superior.
    pause
    exit /b 1
)

if not exist "venv\Scripts\streamlit.exe" (
    echo [AuditorIA] Creando entorno virtual...
    python -m venv venv
    echo [AuditorIA] Instalando dependencias (esto puede tardar varios minutos)...
    venv\Scripts\pip.exe install --upgrade pip
    venv\Scripts\pip.exe install -r requirements.txt
    echo [AuditorIA] Instalacion completada.
    echo.
)

echo Iniciando AuditorIA en http://localhost:8501
echo Presione Ctrl+C en esta ventana para detener el servidor.
echo.

"venv\Scripts\streamlit.exe" run app.py --server.maxUploadSize 500 --browser.gatherUsageStats false

pause