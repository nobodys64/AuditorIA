"""
app.py – Interfaz Institucional de AuditorIA.
Sistema de Integridad Académica y Detección de Contenido Generado por IA.

Versión Corporativa Definitiva:
- 100% libre de emojis con iconografía y logotipos vectoriales SVG profesionales.
- Soporte OCR offline integrado para PDFs escaneados e imágenes (.png, .jpg, .jpeg) con RapidOCR y PyMuPDF.
- Cuadro de mandos superior con 4 tarjetas KPI institucionales.
- Estilos adaptativos de alto contraste para temas claro y oscuro.
- Tabla interactiva con clasificación de riesgo académica y descarga a Excel con openpyxl.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import chardet
import pandas as pd
import streamlit as st
import torch

# ---- Importar motores ----
from motor_texto import (
    analizar_texto,
    generar_html_subrayado,
    cargar_modelo,
    extraer_texto_pdf,
    extraer_texto_imagen,
    MODELO_POR_DEFECTO,
)
from motor_codigo import (
    analizar_codigo,
    comparacion_cruzada_tfidf,
    generar_html_codigo,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constantes y Extensiones
# ═══════════════════════════════════════════════════════════════════════════

EXTENSIONES_TEXTO = {".pdf", ".docx", ".txt", ".tex"}
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg"}
EXTENSIONES_CODIGO = {".adb", ".ads", ".java", ".c", ".h", ".r", ".rmd", ".wl", ".nb"}
EXTENSIONES_VALIDAS = EXTENSIONES_TEXTO | EXTENSIONES_IMAGEN | EXTENSIONES_CODIGO | {".zip"}

PERFILES_HARDWARE = {
    "Gama Media / Portátil Moderno (Recomendado)": {
        "clave": "media",
        "descripcion": "bloom-560m INT8 · Batching completo en CPU · Alta velocidad y precisión.",
    },
    "Gama Baja / PC Oficina Antiguo (CPU / <=8 GB RAM)": {
        "clave": "baja",
        "descripcion": "bloom-560m INT8 · Muestreo de 5 bloques clave · Mínimo consumo de RAM.",
    },
    "Gama Alta / Estación de Trabajo (GPU / Precisión Completa)": {
        "clave": "alta",
        "descripcion": "bloom-560m sin cuantizar (CUDA float16 / CPU float32) · Ventana deslizante 100%.",
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Iconografía Vectorial SVG
# ═══════════════════════════════════════════════════════════════════════════

SVG_LOGO = """
<svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="44" height="44" rx="8" fill="#1E3A8A"/>
  <path d="M22 10L12 14.5V21.25C12 27.4 16.27 33.13 22 34.75C27.73 33.13 32 27.4 32 21.25V14.5L22 10Z" stroke="#93C5FD" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M18 22L21 25L27 19" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_CARD_TOTAL = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
"""

SVG_CARD_ALERTA = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
"""

SVG_CARD_OCR = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0891b2" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
"""

SVG_CARD_MEDIA = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
"""


# ═══════════════════════════════════════════════════════════════════════════
# Detección de Hardware
# ═══════════════════════════════════════════════════════════════════════════

def detectar_hardware() -> tuple[bool, str, int, float | None]:
    """Detecta GPU, cantidad de hilos CPU y memoria RAM total del sistema."""
    cuda_disponible = torch.cuda.is_available()
    gpu_nombre = torch.cuda.get_device_name(0) if cuda_disponible else ""
    cpus = os.cpu_count() or 1
    ram_gb = None

    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            ram_gb = round(stat.ullTotalPhys / (1024**3), 1)
    except Exception:
        pass

    return cuda_disponible, gpu_nombre, cpus, ram_gb


# ═══════════════════════════════════════════════════════════════════════════
# Funciones de extracción de texto multiformato con soporte OCR
# ═══════════════════════════════════════════════════════════════════════════

def _leer_bytes_como_texto(raw: bytes) -> str:
    """Decodifica bytes intentando detectar la codificación."""
    det = chardet.detect(raw)
    encoding = det.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def _extraer_docx(raw: bytes) -> str:
    """Extrae texto de un archivo .docx."""
    from docx import Document
    doc = Document(io.BytesIO(raw))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extraer_contenido(nombre: str, raw: bytes) -> tuple[str, str]:
    """Extrae texto de cualquier archivo compatible y reporta el método utilizado.

    Devuelve: (texto, metodo_descripcion)
    """
    ext = Path(nombre).suffix.lower()

    if ext == ".pdf":
        texto, uso_ocr = extraer_texto_pdf(raw, nombre_archivo=nombre)
        metodo = "PDF (OCR)" if uso_ocr else "PDF (Nativo)"
        return texto, metodo
    elif ext in EXTENSIONES_IMAGEN:
        texto = extraer_texto_imagen(raw, nombre_archivo=nombre)
        return texto, "Imagen (OCR)"
    elif ext == ".docx":
        return _extraer_docx(raw), "DOCX (Nativo)"
    elif ext == ".tex":
        return _leer_bytes_como_texto(raw), "LaTeX (Nativo)"
    elif ext == ".txt":
        return _leer_bytes_como_texto(raw), "TXT (Nativo)"
    else:
        return _leer_bytes_como_texto(raw), f"Código ({ext.lstrip('.').upper()})"


def _clasificar(nombre: str) -> str:
    """Clasifica el archivo en 'texto', 'codigo' o 'ignorar'."""
    ext = Path(nombre).suffix.lower()
    if ext in (EXTENSIONES_TEXTO | EXTENSIONES_IMAGEN):
        return "texto"
    elif ext in EXTENSIONES_CODIGO:
        return "codigo"
    return "ignorar"


def _expandir_zip(nombre_zip: str, raw: bytes, profundidad: int = 0) -> list[tuple[str, bytes]]:
    """Extrae recursivamente archivos compatibles de un ZIP ignorando __MACOSX y archivos ocultos."""
    if profundidad > 3:
        return []
    archivos: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                partes = Path(info.filename).parts
                if any(p.startswith(".") or p == "__MACOSX" for p in partes):
                    continue
                nombre_base = Path(info.filename).name
                if nombre_base.startswith("."):
                    continue

                ext = Path(info.filename).suffix.lower()
                data = zf.read(info.filename)

                if ext == ".zip":
                    alias_zip = f"{Path(nombre_zip).stem}/{nombre_base}"
                    archivos.extend(_expandir_zip(alias_zip, data, profundidad + 1))
                elif ext in (EXTENSIONES_TEXTO | EXTENSIONES_IMAGEN | EXTENSIONES_CODIGO):
                    alias = f"{Path(nombre_zip).stem}/{nombre_base}"
                    archivos.append((alias, data))
    except Exception:
        pass
    return archivos


# ═══════════════════════════════════════════════════════════════════════════
# Interfaz Streamlit
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="AuditorIA — Sistema de Integridad Académica",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inyección de CSS Corporativo / Institucional
    st.markdown(
        """
<style>
  /* Ocultar barra superior y pie de página de Streamlit */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Tipografía e Interfaz General */
  .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1280px;
  }

  /* Tarjetas KPI Superiores */
  .kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .kpi-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #64748b;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
  }
  .kpi-icon {
    width: 44px;
    height: 44px;
    border-radius: 8px;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Modo oscuro para tarjetas KPI */
  @media (prefers-color-scheme: dark) {
    .kpi-card {
      background: #1e293b !important;
      border-color: rgba(255, 255, 255, 0.1) !important;
    }
    .kpi-title { color: #94a3b8 !important; }
    .kpi-value { color: #f8fafc !important; }
    .kpi-icon { background: rgba(255, 255, 255, 0.06) !important; }
  }

  /* Botones primarios corporativos */
  .stButton > button {
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.3px;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s ease;
  }

  /* Badges de Estado */
  .badge-tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
  }
</style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Cabecera Corporativa con Logotipo SVG ----
    st.markdown(
        f"""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
  {SVG_LOGO}
  <div>
    <h1 style="margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.5px; color: #1e3a8a;">
      AUDITORIA
    </h1>
    <div style="font-size: 13px; font-weight: 500; color: #64748b; margin-top: 2px;">
      Sistema de Integridad Académica y Detección de Contenido Generado por IA · 100% Offline
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ---- Sidebar: Configuración Institucional ----
    with st.sidebar:
        st.markdown(
            """
<div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #1e3a8a; margin-bottom: 12px;">
  Configuración del Sistema
</div>
            """,
            unsafe_allow_html=True,
        )

        cuda_ok, gpu_name, cpus, ram_gb = detectar_hardware()

        st.markdown("**Recursos del Servidor / PC**")
        if cuda_ok:
            st.success(f"GPU: {gpu_name}\nAceleración CUDA activa")
            indice_sugerido = 2
        else:
            ram_info = f" · {ram_gb} GB RAM" if ram_gb else ""
            st.info(f"CPU: {cpus} hilos{ram_info}")
            indice_sugerido = 1 if (ram_gb and ram_gb <= 8.0) else 0

        perfil_seleccionado_label = st.selectbox(
            "Perfil de Procesamiento",
            list(PERFILES_HARDWARE.keys()),
            index=indice_sugerido,
            help="Seleccione el perfil adecuado para la potencia de cálculo del equipo docente.",
        )

        perfil_info = PERFILES_HARDWARE[perfil_seleccionado_label]
        perfil_clave = perfil_info["clave"]
        st.caption(perfil_info["descripcion"])

        st.divider()
        st.markdown(
            "**Formatos Admitidos**\n\n"
            "• **Texto y Documentos:** `.pdf`, `.docx`, `.txt`, `.tex`\n\n"
            "• **Escaneos / Imágenes:** `.png`, `.jpg`, `.jpeg` *(OCR Automático)*\n\n"
            "• **Código Fuente:** `.c`, `.h`, `.java`, `.adb`, `.ads`, `.r`, `.wl`, `.nb`\n\n"
            "• **Comprimidos:** `.zip` *(Descompresión por alumno)*"
        )
        st.divider()
        st.markdown(
            "**Escala de Clasificación**\n\n"
            "• **Alto Riesgo IA:** $\\ge 70\\%$\n\n"
            "• **Revisión Recomendada:** $40\\% - 69\\%$\n\n"
            "• **Autoría Humana:** $< 40\\%$"
        )

    # ---- Carga masiva de archivos ----
    archivos = st.file_uploader(
        "Carga de entregas (hasta 500 archivos simultáneos o paquete comprimido en .zip)",
        type=[e.lstrip(".") for e in EXTENSIONES_VALIDAS],
        accept_multiple_files=True,
    )

    if not archivos:
        st.info("Seleccione o arrastre los archivos de los alumnos para iniciar el proceso de auditoría.")
        st.stop()

    if len(archivos) > 500:
        st.error("Límite de procesamiento: máximo 500 archivos por lote.")
        st.stop()

    # Expandir archivos y ZIPs
    items: list[tuple[str, bytes]] = []
    for f in archivos:
        if f.name.lower().endswith(".zip"):
            items.extend(_expandir_zip(f.name, f.read()))
        else:
            items.append((f.name, f.read()))

    if not items:
        st.warning("No se encontraron documentos válidos en el conjunto cargado.")
        st.stop()

    st.success(f"Lote preparado: **{len(items)} archivo(s)** listos para auditar con el perfil **{perfil_seleccionado_label.split('/')[0].strip()}**.")

    # ---- Botón de Ejecución ----
    if st.button("Ejecutar Auditoría Académica", type="primary", use_container_width=True):
        resultados_texto: list[dict] = []
        resultados_codigo: list[dict] = []
        errores: list[str] = []

        barra = st.progress(0)
        texto_progreso = st.empty()
        texto_progreso.markdown("*Iniciando motores de análisis y OCR offline...*")

        # Cargar modelo si hay archivos de texto o imágenes
        tiene_texto = any(_clasificar(n) == "texto" for n, _ in items)
        if tiene_texto:
            with st.spinner(f"Inicializando modelo de perplejidad ({perfil_clave.upper()})..."):
                try:
                    cargar_modelo(MODELO_POR_DEFECTO, perfil=perfil_clave)
                except Exception as e:
                    st.error(f"Error al inicializar el modelo: {e}")
                    st.stop()

        total = len(items)
        for idx, (nombre, raw) in enumerate(items):
            texto_progreso.markdown(
                f"**Procesando archivo {idx+1} de {total}:** `{nombre}`..."
            )
            barra.progress((idx + 1) / total)

            tipo = _clasificar(nombre)
            try:
                if tipo == "texto":
                    texto, metodo = extraer_contenido(nombre, raw)
                    if len(texto.strip()) < 30:
                        errores.append(
                            f"{nombre}: Texto insuficiente ({len(texto.strip())} caracteres). "
                            "Verifique si el documento es legible o contiene texto seleccionable."
                        )
                        continue
                    res = analizar_texto(
                        texto,
                        nombre,
                        MODELO_POR_DEFECTO,
                        perfil=perfil_clave,
                        metodo_extraccion=metodo,
                    )
                    resultados_texto.append(res)
                elif tipo == "codigo":
                    ext = Path(nombre).suffix.lower()
                    codigo = _leer_bytes_como_texto(raw)
                    if len(codigo.strip()) < 10:
                        errores.append(f"{nombre}: Archivo de código fuente vacío o corrupto.")
                        continue
                    res = analizar_codigo(codigo, nombre, ext)
                    res["metodo_extraccion"] = f"Código ({ext.lstrip('.').upper()})"
                    resultados_codigo.append(res)
            except Exception as e:
                errores.append(f"{nombre}: {e}")

        barra.empty()
        texto_progreso.empty()

        st.session_state["resultados_texto"] = resultados_texto
        st.session_state["resultados_codigo"] = resultados_codigo
        st.session_state["errores"] = errores

        if len(resultados_codigo) >= 2:
            comp = comparacion_cruzada_tfidf(resultados_codigo)
            st.session_state["comparacion_cruzada"] = comp
        else:
            st.session_state["comparacion_cruzada"] = None

    # ---- Resultados en pantalla ----
    resultados_texto = st.session_state.get("resultados_texto", [])
    resultados_codigo = st.session_state.get("resultados_codigo", [])
    errores = st.session_state.get("errores", [])
    comparacion = st.session_state.get("comparacion_cruzada")

    if not resultados_texto and not resultados_codigo:
        st.stop()

    # ---- Tarjetas KPI Superiores ----
    todos = resultados_texto + resultados_codigo
    total_auditados = len(todos)
    alto_riesgo_cnt = sum(1 for r in todos if r.get("score_ia", 0) >= 70.0)
    ocr_cnt = sum(1 for r in resultados_texto if "OCR" in r.get("metodo_extraccion", ""))
    media_ia = (sum(r.get("score_ia", 0) for r in todos) / total_auditados) if total_auditados > 0 else 0.0

    st.markdown(
        f"""
<div class="kpi-container">
  <div class="kpi-card">
    <div>
      <div class="kpi-title">Entregas Auditadas</div>
      <div class="kpi-value">{total_auditados}</div>
    </div>
    <div class="kpi-icon">{SVG_CARD_TOTAL}</div>
  </div>
  <div class="kpi-card">
    <div>
      <div class="kpi-title">Alto Riesgo IA</div>
      <div class="kpi-value" style="color:#b91c1c;">{alto_riesgo_cnt}</div>
    </div>
    <div class="kpi-icon">{SVG_CARD_ALERTA}</div>
  </div>
  <div class="kpi-card">
    <div>
      <div class="kpi-title">Procesados con OCR</div>
      <div class="kpi-value" style="color:#0891b2;">{ocr_cnt}</div>
    </div>
    <div class="kpi-icon">{SVG_CARD_OCR}</div>
  </div>
  <div class="kpi-card">
    <div>
      <div class="kpi-title">Media Global IA</div>
      <div class="kpi-value" style="color:#6d28d9;">{media_ia:.1f} %</div>
    </div>
    <div class="kpi-icon">{SVG_CARD_MEDIA}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Avisos de Extracción / Documentos No Legibles ----
    if errores:
        with st.expander(f"Incidencias en {len(errores)} archivo(s)", expanded=False):
            for e in errores:
                st.warning(e)

    # ---- Tabla Resumen de Auditoría ----
    st.subheader("Informe General de Entregas")

    filas: list[dict] = []
    for r in resultados_texto:
        ppl = r.get("perplejidad_media")
        ppl_str = f"{ppl:.2f}" if (ppl is not None and isinstance(ppl, (int, float))) else "—"
        burst = r.get("burstiness_media")
        burst_str = f"{burst:.3f}" if (burst is not None and isinstance(burst, (int, float))) else "—"
        score = round(float(r["score_ia"]), 1)

        filas.append({
            "Archivo": r["archivo"],
            "Tipo": "Texto",
            "Método": r.get("metodo_extraccion", "Nativo"),
            "Score IA (%)": score,
            "Perplejidad": ppl_str,
            "Burstiness": burst_str,
            "Patrones LLM": "—",
        })

    for r in resultados_codigo:
        score = round(float(r["score_ia"]), 1)
        filas.append({
            "Archivo": r["archivo"],
            "Tipo": "Código",
            "Método": r.get("metodo_extraccion", "Nativo"),
            "Score IA (%)": score,
            "Perplejidad": "—",
            "Burstiness": "—",
            "Patrones LLM": str(len(r.get("hallazgos", []))),
        })

    if filas:
        df = pd.DataFrame(filas).sort_values("Score IA (%)", ascending=False)
        df = df.reset_index(drop=True)
        df.index += 1

        def _colorear_score(val):
            try:
                v = float(str(val).replace("%", "").strip())
            except (ValueError, TypeError):
                return ""
            if v >= 70.0:
                return "background-color: #fee2e2; color: #111111; font-weight: bold;"
            elif v >= 40.0:
                return "background-color: #fef3c7; color: #111111; font-weight: bold;"
            else:
                return "background-color: #d1fae5; color: #111111;"

        styler = df.style.format({
            "Score IA (%)": "{:.1f} %",
        })

        if hasattr(styler, "map"):
            styled = styler.map(_colorear_score, subset=["Score IA (%)"])
        else:
            styled = styler.applymap(_colorear_score, subset=["Score IA (%)"])

        st.dataframe(styled, use_container_width=True, height=min(len(df) * 38 + 50, 600))

        # Descarga a Excel con openpyxl
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Resumen", index_label="#")
            if comparacion and len(comparacion["pares_sospechosos"]) > 0:
                df_pares = pd.DataFrame(
                    [
                        {
                            "Archivo A": a,
                            "Archivo B": b,
                            "Similitud Estructural": f"{sim*100:.1f} %",
                        }
                        for a, b, sim in comparacion["pares_sospechosos"]
                    ]
                )
                df_pares.to_excel(writer, sheet_name="Similitud código", index=False)
            else:
                df_vacio = pd.DataFrame(
                    [{"Información": "No se detectaron pares de código con similitud superior al 70%"}]
                )
                df_vacio.to_excel(writer, sheet_name="Similitud código", index=False)

        st.download_button(
            label="Descargar Informe de Auditoría (.xlsx)",
            data=buffer_excel.getvalue(),
            file_name="auditoria_academica.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ---- Comparación Cruzada de Código ----
    if comparacion and comparacion["pares_sospechosos"]:
        st.subheader("Detección de Similitud Cruzada en Código (TF-IDF Estructural)")
        st.info(
            f"Se han identificado {len(comparacion['pares_sospechosos'])} pares de entregas "
            "con concordancia estructural superior al 70%, con independencia del renombramiento de variables."
        )
        for a, b, sim in comparacion["pares_sospechosos"]:
            st.markdown(f"• **{a}** $\\longleftrightarrow$ **{b}** : `{sim:.1%}` de concordancia estructural")

    # ---- Visor Detallado por Entrega ----
    st.subheader("Inspección Individualizada de Entregas")

    nombres_archivos = [r["archivo"] for r in todos]
    if nombres_archivos:
        seleccion = st.selectbox(
            "Seleccione el documento para examinar el desglose por bloques y métricas:",
            nombres_archivos,
        )
        resultado_sel = next(r for r in todos if r["archivo"] == seleccion)

        if resultado_sel["tipo"] == "texto":
            html = generar_html_subrayado(resultado_sel)
        else:
            html = generar_html_codigo(resultado_sel)

        st.components.v1.html(html, height=720, scrolling=True)


if __name__ == "__main__":
    main()
