#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export HF_HOME="$(pwd)/modelos_cache"

echo "[AuditorIA] Iniciando sistema en macOS..."
echo "[AuditorIA] HF_HOME=$HF_HOME"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 no encontrado. Instala Python desde https://python.org"
    read -p "Presiona Enter para cerrar..."
    exit 1
fi

if [ ! -f "venv_mac/bin/streamlit" ]; then
    echo "[AuditorIA] Creando entorno virtual Python..."
    python3 -m venv venv_mac
    echo "[AuditorIA] Instalando dependencias (puede tardar varios minutos)..."
    ./venv_mac/bin/pip install --upgrade pip
    ./venv_mac/bin/pip install -r requirements.txt
    echo "[AuditorIA] Instalacion completada."
    echo
fi

echo "[AuditorIA] Servidor disponible en http://localhost:8501"
echo "[AuditorIA] Presione Ctrl+C para detener."
echo
./venv_mac/bin/streamlit run app.py --server.maxUploadSize 500 --browser.gatherUsageStats false