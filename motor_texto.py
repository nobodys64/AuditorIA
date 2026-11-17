"""
motor_texto.py – Motor de análisis de texto y extracción documental para AuditorIA.

Incluye:
- Extracción de PDFs híbrida: texto nativo con PyMuPDF y OCR offline automático
  (RapidOCR con ONNX Runtime) para páginas escaneadas o con <20 caracteres.
- Extracción directa por OCR de imágenes (.png, .jpg, .jpeg).
- Filtrado de transparencias, encabezados de diapositivas y fórmulas aisladas.
- Análisis de perplejidad y burstiness multilingüe con bigscience/bloom-560m.
- Soporte para 3 perfiles de hardware (Gama Baja INT8, Gama Media INT8, Gama Alta float).
- Generación de informes HTML corporativos con logotipos y badges vectoriales SVG (sin emojis).
"""

from __future__ import annotations

import io
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Configurar ruta de caché local autocontenida para modelos (portabilidad USB/offline)
_DIR_PROYECTO = Path(__file__).resolve().parent
_CACHE_LOCAL = _DIR_PROYECTO / "modelos_cache"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_CACHE_LOCAL)

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELO_POR_DEFECTO = "bigscience/bloom-560m"


# ---------------------------------------------------------------------------
# Motor OCR Offline (RapidOCR con ONNX Runtime)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _obtener_instancia_ocr():
    """Inicializa y cachea el motor de OCR offline RapidOCR."""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def extraer_texto_imagen(raw_bytes: bytes, nombre_archivo: str = "") -> str:
    """Extrae texto de una imagen (.png, .jpg, .jpeg) usando RapidOCR offline."""
    ocr = _obtener_instancia_ocr()
    result, _ = ocr(raw_bytes)
    if not result:
        raise ValueError(
            f"El documento de imagen '{nombre_archivo}' no contiene texto reconocible por OCR."
        )
    lineas = [r[1].strip() for r in result if r[1] and r[1].strip()]
    texto = "\n".join(lineas).strip()
    if len(texto) < 15:
        raise ValueError(
            f"El documento '{nombre_archivo}' contiene texto insuficiente para su análisis ({len(texto)} caracteres)."
        )
    return texto


def _limpiar_diapositivas_pdf(paginas_texto: list[str]) -> str:
    """Filtra encabezados/pies repetitivos entre páginas de diapositivas o apuntes."""
    if len(paginas_texto) < 2:
        return "\n\n".join(paginas_texto)

    contador_lineas = Counter()
    for pag in paginas_texto:
        lineas_unicas = set(l.strip() for l in pag.splitlines() if len(l.strip()) > 3)
        for l in lineas_unicas:
            contador_lineas[l] += 1

    umbral_repeticion = max(2, int(len(paginas_texto) * 0.40))
    lineas_repetidas = {l for l, c in contador_lineas.items() if c >= umbral_repeticion}

    paginas_limpias = []
    for pag in paginas_texto:
        lineas_validas = []
        for l in pag.splitlines():
            strip = l.strip()
            if strip in lineas_repetidas:
                continue
            if re.match(r"^(p[aá]g\.?|p\.|p[aá]gina)?\s*\d+(\s*/\s*\d+)?$", strip, re.I):
                continue
            lineas_validas.append(l)
        paginas_limpias.append("\n".join(lineas_validas))

    return "\n\n".join(paginas_limpias)


def extraer_texto_pdf(raw_bytes: bytes, nombre_archivo: str = "") -> tuple[str, bool]:
    """Extrae texto de un archivo PDF usando PyMuPDF.

    Si una página contiene menos de 20 caracteres de texto seleccionable,
    renderiza automáticamente la página a imagen (150 DPI) y aplica OCR offline.

    Devuelve: (texto_extraido, se_uso_ocr)
    """
    import pymupdf

    try:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"El archivo '{nombre_archivo}' está dañado o no es un PDF válido ({e}).")

    if doc.is_encrypted:
        try:
            doc.authenticate("")
        except Exception:
            raise ValueError(f"El PDF '{nombre_archivo}' está protegido con contraseña y no se puede leer.")

    if len(doc) == 0:
        raise ValueError(f"El PDF '{nombre_archivo}' no contiene páginas legibles.")

    paginas_texto: list[str] = []
    ocr_activado = False
    ocr_engine = None

    for page in doc:
        txt_nativo = page.get_text() or ""
        txt_limpio = txt_nativo.strip()

        # Si la página tiene menos de 20 caracteres de texto seleccionable -> Renderizar y aplicar OCR
        if len(txt_limpio) < 20:
            if ocr_engine is None:
                ocr_engine = _obtener_instancia_ocr()
            ocr_activado = True
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            result, _ = ocr_engine(img_bytes)
            if result:
                txt_ocr = "\n".join(r[1].strip() for r in result if r[1] and r[1].strip())
                paginas_texto.append(txt_ocr)
            else:
                paginas_texto.append("")
        else:
            paginas_texto.append(txt_nativo)

    doc.close()

    texto_total = "\n\n".join(paginas_texto).strip()
    if len(texto_total) < 20:
        raise ValueError(
            f"El PDF '{nombre_archivo}' es un documento escaneado que no contiene texto seleccionable ni caracteres reconocibles por OCR."
        )

    return _limpiar_diapositivas_pdf(paginas_texto), ocr_activado


# ---------------------------------------------------------------------------
# Carga de modelo causal con caché de recursos de Streamlit
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cargar_modelo(nombre_modelo: str = MODELO_POR_DEFECTO, perfil: str = "media"):
    """Carga y cachea en memoria el modelo y su tokenizador según el perfil de hardware."""
    usar_cuda = (perfil == "alta") and torch.cuda.is_available()
    device = torch.device("cuda" if usar_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if usar_cuda else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(nombre_modelo, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(nombre_modelo, torch_dtype=dtype)

    if perfil in ("baja", "media"):
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            pass
        model.to("cpu")
        device = torch.device("cpu")
    else:
        model.to(device)

    model.eval()
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Filtrado de bloques y cálculo de métricas
# ---------------------------------------------------------------------------

def es_bloque_prosa_valido(bloque: str) -> bool:
    """Verifica si un bloque es texto en prosa analizable y no una lista telegráfica o fórmulas."""
    palabras = bloque.split()
    if len(palabras) < 20:
        return False

    lineas = [l.strip() for l in bloque.splitlines() if l.strip()]
    if not lineas:
        return False

    lineas_vineta = sum(
        1 for l in lineas if re.match(r"^([•\-\*>]|\d+[\.\)]|[a-zA-Z][\.\)])\s+", l)
    )
    if lineas_vineta / len(lineas) > 0.55 and len(palabras) < 50:
        return False

    caracteres_letras = sum(1 for c in bloque if c.isalpha())
    if caracteres_letras / max(len(bloque), 1) < 0.45:
        return False

    return True


def dividir_en_bloques(texto: str, min_palabras: int = 30) -> list[str]:
    """Divide el texto en bloques de prosa continua de al menos min_palabras."""
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    bloques: list[str] = []
    buf = ""
    for p in parrafos:
        buf = f"{buf} {p}".strip() if buf else p
        if len(buf.split()) >= min_palabras:
            bloques.append(buf)
            buf = ""
    if buf:
        if bloques:
            bloques[-1] += " " + buf
        else:
            bloques.append(buf)

    bloques_validos = [b for b in bloques if es_bloque_prosa_valido(b)]
    return bloques_validos if bloques_validos else (bloques or [texto])


def seleccionar_bloques_representativos(bloques: list[str], n_max: int = 5) -> list[str]:
    """Selecciona hasta n_max bloques representativos distribuidos en el documento (Gama Baja)."""
    if len(bloques) <= n_max:
        return bloques

    indices = [
        0,
        len(bloques) // 4,
        len(bloques) // 2,
        (3 * len(bloques)) // 4,
        len(bloques) - 1,
    ]
    vistos = set()
    seleccionados = []
    for idx in indices:
        if idx not in vistos and idx < len(bloques):
            vistos.add(idx)
            seleccionados.append(bloques[idx])
    return seleccionados


def _perplejidad_bloque(
    texto: str, tokenizer, model, device, max_tokens: int = 512
) -> float | None:
    """Perplejidad de un bloque individual."""
    inputs = tokenizer(
        texto, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    ids = inputs.input_ids.to(device)
    if ids.shape[1] < 2:
        return None
    with torch.no_grad():
        out = model(ids, labels=ids)
    val = out.loss.item()
    return math.exp(val) if val < 100 else 1000.0


def _perplejidad_lote(
    textos: list[str],
    tokenizer,
    model,
    device,
    max_tokens: int = 512,
    batch_size: int = 4,
) -> list[float | None]:
    """Calcula la perplejidad de una lista de bloques en lotes vectorizados."""
    if not textos:
        return []

    resultados: list[float | None] = []
    for i in range(0, len(textos), batch_size):
        chunk = textos[i : i + batch_size]
        if len(chunk) == 1:
            resultados.append(_perplejidad_bloque(chunk[0], tokenizer, model, device, max_tokens))
            continue

        try:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(device)

            labels = inputs.input_ids.clone()
            labels[inputs.attention_mask == 0] = -100

            with torch.no_grad():
                outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())

                mask = (shift_labels != -100).float()
                seq_loss = (loss * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

                for l in seq_loss:
                    val = l.item()
                    resultados.append(math.exp(val) if val < 100 else 1000.0)

        except Exception:
            for t in chunk:
                resultados.append(_perplejidad_bloque(t, tokenizer, model, device, max_tokens))

    return resultados


def _perplejidad_ventana_deslizante(
    texto: str,
    tokenizer,
    model,
    device,
    window_size: int = 512,
    stride: int = 256,
) -> float | None:
    """Calcula la perplejidad mediante ventana deslizante con solapamiento (Gama Alta)."""
    encodings = tokenizer(texto, return_tensors="pt").to(device)
    seq_len = encodings.input_ids.size(1)
    if seq_len < 2:
        return None
    if seq_len <= window_size:
        return _perplejidad_bloque(texto, tokenizer, model, device, window_size)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + window_size, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_loss = torch.stack(nlls).sum() / seq_len
    return math.exp(total_loss.item())


def calcular_burstiness(texto: str) -> float:
    """Coeficiente de variación de longitudes de oración (burstiness)."""
    oraciones = [s.strip() for s in re.split(r"[.!?¿¡]+", texto) if s.strip()]
    lens = [len(s.split()) for s in oraciones]
    if len(lens) < 2:
        return 0.0
    mu = float(np.mean(lens))
    return float(np.std(lens) / mu) if mu > 0 else 0.0


# ---------------------------------------------------------------------------
# Detección de conectores LLM y elementos técnicos
# ---------------------------------------------------------------------------

PATRONES_CONECTORES_ES = re.compile(
    r"\b(en primer lugar|en segundo lugar|por [uú]ltimo|a modo de conclusi[oó]n|"
    r"en conclusi[oó]n|en resumen|en s[ií]ntesis|en definitiva|"
    r"cabe (destacar|se[ñn]alar|mencionar|resaltar)|"
    r"es (importante|relevante|fundamental|crucial|menester) (destacar|se[ñn]alar|mencionar|notar|recordar|resaltar)|"
    r"por un lado|por otro lado|por una parte|por otra parte|"
    r"asimismo|del mismo modo|de igual manera|de forma an[aá]loga|"
    r"en este sentido|en consonancia con|a ra[ií]z de esto|como resultado de ello|"
    r"como se ha mencionado|como se puede (observar|apreciar)|a continuaci[oó]n se presenta|"
    r"no obstante|sin embargo|por consiguiente|por ende|en consecuencia|"
    r"juega un papel (fundamental|crucial|clave)|desempe[ñn]a un papel (fundamental|crucial|clave))\b",
    re.IGNORECASE,
)


def calcular_densidad_tecnica(texto: str) -> float:
    """Calcula la proporción de elementos técnicos (fórmulas matemáticas, viñetas, expresiones científicas)."""
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    if not lineas:
        return 0.0
    simbolos_math = len(re.findall(r"[\$%=+\\^_{}\[\]\<\>]", texto))
    lineas_vineta = sum(
        1 for l in lineas if re.match(r"^([•\-\*>]|\d+[\.\)]|[a-zA-Z][\.\)])\s+", l)
    )
    palabras = max(len(texto.split()), 1)
    ratio_math = min(1.0, simbolos_math / palabras)
    ratio_vineta = lineas_vineta / len(lineas)
    return min(1.0, ratio_math * 0.6 + ratio_vineta * 0.4)


def calcular_score_ia_texto(
    ppl_media: float | None,
    burst_media: float,
    texto: str = "",
    perfil: str = "media",
) -> float:
    """Calcula la probabilidad global de IA adaptada a la cuantización y calibrada para cero falsos positivos."""
    if ppl_media is None:
        return 50.0

    # Calibración sigmoide sobre PPL
    if perfil in ("baja", "media"):
        score_ppl = 100.0 / (1.0 + math.exp((ppl_media - 850.0) / 160.0))
    else:
        score_ppl = 100.0 / (1.0 + math.exp((ppl_media - 22.0) / 4.5))

    # Calibración sigmoide sobre Burstiness
    score_burst = 100.0 / (1.0 + math.exp((burst_media - 0.42) / 0.08))

    # Conectores típicos de modelos LLM en español
    if texto:
        n_conectores = len(PATRONES_CONECTORES_ES.findall(texto))
        palabras = max(len(texto.split()), 1)
        densidad_conectores = (n_conectores / palabras) * 100
        score_conectores = min(100.0, (densidad_conectores / 1.5) * 100.0)
    else:
        score_conectores = 0.0

    score_base = score_ppl * 0.50 + score_burst * 0.30 + score_conectores * 0.20

    # Penalización técnica anti-falsos positivos (apuntes con fórmulas, viñetas, ejercicios)
    if texto:
        densidad_tec = calcular_densidad_tecnica(texto)
        if densidad_tec > 0.08:
            factor_tecnico = max(0.20, 1.0 - densidad_tec * 1.5)
            score_base *= factor_tecnico

    return round(max(0.0, min(100.0, score_base)), 1)


def analizar_texto(
    texto: str,
    nombre_archivo: str = "",
    nombre_modelo: str = MODELO_POR_DEFECTO,
    perfil: str = "media",
    metodo_extraccion: str = "Nativo",
) -> dict:
    """Analiza un documento de texto aplicando el perfil de hardware especificado."""
    tokenizer, model, device = cargar_modelo(nombre_modelo, perfil=perfil)
    todos_bloques = dividir_en_bloques(texto)

    if perfil == "baja":
        bloques = seleccionar_bloques_representativos(todos_bloques, n_max=5)
        ppls = _perplejidad_lote(bloques, tokenizer, model, device, max_tokens=256, batch_size=2)
        import gc
        gc.collect()
    elif perfil == "alta":
        bloques = todos_bloques
        ppls = [_perplejidad_ventana_deslizante(b, tokenizer, model, device) for b in bloques]
    else:  # media
        bloques = todos_bloques
        ppls = _perplejidad_lote(bloques, tokenizer, model, device, max_tokens=512, batch_size=4)

    resultados_bloques: list[dict] = []
    for idx, b in enumerate(bloques):
        burst = calcular_burstiness(b)
        ppl = ppls[idx] if idx < len(ppls) else None
        resultados_bloques.append(
            {"texto": b, "perplejidad": ppl, "burstiness": burst}
        )

    ppls_validos = [r["perplejidad"] for r in resultados_bloques if r["perplejidad"] is not None]
    bursts = [r["burstiness"] for r in resultados_bloques]

    ppl_media = float(np.mean(ppls_validos)) if ppls_validos else None
    burst_media = float(np.mean(bursts)) if bursts else 0.0

    score_ia = calcular_score_ia_texto(ppl_media, burst_media, texto=texto, perfil=perfil)

    return {
        "archivo": nombre_archivo,
        "tipo": "texto",
        "metodo_extraccion": metodo_extraccion,
        "perfil_usado": perfil,
        "total_bloques_doc": len(todos_bloques),
        "bloques_analizados": len(bloques),
        "bloques": resultados_bloques,
        "perplejidad_media": round(ppl_media, 2) if ppl_media is not None else None,
        "burstiness_media": round(burst_media, 3),
        "score_ia": score_ia,
    }


# ---------------------------------------------------------------------------
# Generación de HTML corporativo con logotipos y badges SVG (sin emojis)
# ---------------------------------------------------------------------------

def generar_html_subrayado(resultado: dict) -> str:
    """Genera HTML con diseño institucional y badges vectoriales SVG sin emojis."""
    bloques = resultado.get("bloques", [])
    if not bloques:
        return "<p style='color:#64748b;font-family:sans-serif;'>No se encontró texto suficiente para auditar.</p>"

    ppls = [b["perplejidad"] for b in bloques if b["perplejidad"] is not None]
    mediana = float(np.median(ppls)) if ppls else 20.0
    perfil = resultado.get("perfil_usado", "media")

    ppl_media_str = f"{resultado['perplejidad_media']:.2f}" if resultado.get("perplejidad_media") is not None else "—"
    burst_media_str = f"{resultado['burstiness_media']:.3f}" if resultado.get("burstiness_media") is not None else "—"
    score_ia = resultado.get("score_ia", 0.0)
    metodo = resultado.get("metodo_extraccion", "Nativo")

    if score_ia >= 70.0:
        badge_global = '<span class="badge-status badge-alto">ALTO RIESGO IA</span>'
    elif score_ia >= 40.0:
        badge_global = '<span class="badge-status badge-revision">REVISIÓN RECOMENDADA</span>'
    else:
        badge_global = '<span class="badge-status badge-humano">AUTORÍA HUMANA</span>'

    html_parts = [
        """
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; padding: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b; background-color: transparent;
  }
  @media (prefers-color-scheme: dark) {
    body { color: #f1f5f9; }
    .metrica-card {
      background: rgba(30, 41, 59, 0.7) !important;
      border: 1px solid rgba(255, 255, 255, 0.12) !important;
      color: #ffffff !important;
    }
    .metrica-label { color: #94a3b8 !important; }
    .bloque-ia {
      background: #2b1114 !important; border-left: 4px solid #ef4444 !important;
      color: #fecaca !important;
    }
    .bloque-sospechoso {
      background: #2b2308 !important; border-left: 4px solid #f59e0b !important;
      color: #fef3c7 !important;
    }
    .bloque-humano {
      background: #0d2818 !important; border-left: 4px solid #10b981 !important;
      color: #d1fae5 !important;
    }
    .badge-sub { color: #94a3b8 !important; }
  }
  .metrica-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .metrica-label { color: #64748b; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
  .badge-status {
    display: inline-block; padding: 4px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.8px;
  }
  .badge-alto { background: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }
  .badge-revision { background: #fef3c7; color: #b45309; border: 1px solid #fbbf24; }
  .badge-humano { background: #d1fae5; color: #047857; border: 1px solid #34d399; }
  .bloque-ia {
    background: #fff1f2; border-left: 4px solid #e11d48;
    color: #1e293b; padding: 14px 18px; margin: 12px 0;
    border-radius: 6px; line-height: 1.7; font-size: 14px;
  }
  .bloque-sospechoso {
    background: #fffbeb; border-left: 4px solid #d97706;
    color: #1e293b; padding: 14px 18px; margin: 12px 0;
    border-radius: 6px; line-height: 1.7; font-size: 14px;
  }
  .bloque-humano {
    background: #f0fdf4; border-left: 4px solid #059669;
    color: #1e293b; padding: 14px 18px; margin: 12px 0;
    border-radius: 6px; line-height: 1.7; font-size: 14px;
  }
  .badge-sub { font-size: 11px; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .icon-svg { vertical-align: middle; margin-right: 6px; }
</style>
        """,
        f"""
<div style="max-width:860px;margin:auto;">
  <div class="metrica-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-bottom:1px solid rgba(128,128,128,0.2);padding-bottom:10px;">
      <div style="display:flex;align-items:center;">
        <svg class="icon-svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
        <span style="font-size:16px;font-weight:700;">{resultado['archivo']}</span>
      </div>
      <div>{badge_global}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;font-size:13px;">
      <div>
        <div class="metrica-label">Perplejidad Media</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;">{ppl_media_str}</div>
      </div>
      <div>
        <div class="metrica-label">Burstiness</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;">{burst_media_str}</div>
      </div>
      <div>
        <div class="metrica-label">Score IA</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;">{score_ia:.1f} %</div>
      </div>
      <div>
        <div class="metrica-label">Extracción</div>
        <div style="font-size:14px;font-weight:600;margin-top:4px;">{metodo}</div>
      </div>
    </div>
  </div>
  <div style="font-size:12px;color:#64748b;margin-bottom:14px;display:flex;gap:18px;">
    <span><span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;margin-right:4px;"></span>Alta probabilidad IA</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border-radius:2px;margin-right:4px;"></span>Revisión recomendada</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#10b981;border-radius:2px;margin-right:4px;"></span>Autoría humana</span>
  </div>
        """,
    ]

    if perfil in ("baja", "media"):
        umbral_ia = 420.0
        umbral_sospecha = 700.0
    else:
        umbral_ia = 16.0
        umbral_sospecha = 25.0

    for b in bloques:
        ppl = b["perplejidad"]
        burst = b["burstiness"]

        if ppl is None:
            clase = "bloque-humano"
            etiqueta = "TEXTO INFORMATIVO (MUESTRA)"
        elif ppl < umbral_ia or (ppl < mediana * 0.70):
            clase = "bloque-ia"
            etiqueta = "ALTO RIESGO IA"
        elif ppl < umbral_sospecha or (ppl < mediana * 0.90):
            clase = "bloque-sospechoso"
            etiqueta = "REVISIÓN RECOMENDADA"
        else:
            clase = "bloque-humano"
            etiqueta = "AUTORÍA HUMANA"

        ppl_str = f"{ppl:.1f}" if ppl is not None else "N/A"
        html_parts.append(
            f'<div class="{clase}">'
            f'<div class="badge-sub">{etiqueta} &nbsp;|&nbsp; PPL: {ppl_str} &nbsp;|&nbsp; BURSTINESS: {burst:.2f}</div>'
            f'{b["texto"]}</div>'
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)
