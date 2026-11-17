#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export HF_HOME="$(pwd)/modelos_cache"

echo "[AuditorIA] Iniciando sistema..."
echo "[AuditorIA] HF_HOME=$HF_HOME"

if [ ! -f "venv_linux/bin/streamlit" ]; then
    echo "[AuditorIA] Creando entorno virtual Python..."
    python3 -m venv venv_linux
    echo "[AuditorIA] Instalando dependencias (puede tardar varios minutos)..."
    ./venv_linux/bin/pip install --upgrade pip
    ./venv_linux/bin/pip install -r requirements.txt
    echo "[AuditorIA] Instalacion completada."
    echo
fi

echo "[AuditorIA] Servidor disponible en http://localhost:8501"
echo "[AuditorIA] Presione Ctrl+C para detener."
echo
./venv_linux/bin/streamlit run app.py --server.maxUploadSize 500 --browser.gatherUsageStats false