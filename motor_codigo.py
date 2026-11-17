"""
motor_codigo.py – Motor de análisis de código fuente para AuditorIA.

Detecta patrones típicos de código generado por LLM en Ada, C, Java, R,
Wolfram y LaTeX. Realiza comparación cruzada entre entregas usando
normalización estructural de identificadores (reemplazando nombres de
usuario por token genérico 'ID') y TF-IDF con similitud del coseno.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════════════════
# 1. Eliminación de comentarios por lenguaje
# ═══════════════════════════════════════════════════════════════════════════

def _strip_c_comments(code: str) -> str:
    """Elimina comentarios estilo C/C++/Java (/* ... */ y // ...)."""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    return code


def _strip_ada_comments(code: str) -> str:
    """Elimina comentarios Ada (-- ...)."""
    return re.sub(r"--.*", "", code)


def _strip_r_comments(code: str) -> str:
    """Elimina comentarios R (# ...)."""
    return re.sub(r"#.*", "", code)


def _strip_latex_comments(code: str) -> str:
    """Elimina comentarios LaTeX (% ... que no sean \\%)."""
    return re.sub(r"(?<!\\)%.*", "", code)


def _strip_wolfram_comments(code: str) -> str:
    """Elimina comentarios Wolfram (* ... *)."""
    return re.sub(r"\(\*.*?\*\)", "", code, flags=re.DOTALL)


_STRIPPERS: dict[str, Any] = {
    ".c": _strip_c_comments,
    ".h": _strip_c_comments,
    ".java": _strip_c_comments,
    ".adb": _strip_ada_comments,
    ".ads": _strip_ada_comments,
    ".r": _strip_r_comments,
    ".rmd": _strip_r_comments,
    ".tex": _strip_latex_comments,
    ".wl": _strip_wolfram_comments,
    ".nb": _strip_wolfram_comments,
}


def eliminar_comentarios(codigo: str, extension: str) -> str:
    """Elimina comentarios según la extensión del archivo."""
    stripper = _STRIPPERS.get(extension.lower(), lambda c: c)
    return stripper(codigo)


def eliminar_literales_cadena(codigo: str) -> str:
    """Reemplaza literales de cadena (comillas dobles y simples) por un token STR."""
    codigo = re.sub(r'"(?:\\.|[^"\\])*"', ' "STR" ', codigo)
    codigo = re.sub(r"'(?:\\.|[^'\\])*'", " 'STR' ", codigo)
    return codigo


# ═══════════════════════════════════════════════════════════════════════════
# 2. Palabras reservadas por lenguaje (para no reemplazarlas por ID)
# ═══════════════════════════════════════════════════════════════════════════

KEYWORDS_C = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_bool", "_complex", "_imaginary",
    "include", "define", "ifdef", "ifndef", "endif", "pragma", "null", "true", "false",
    "printf", "scanf", "malloc", "free", "exit", "size_t", "rand", "srand", "time",
}

KEYWORDS_JAVA = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false", "null",
    "string", "system", "out", "println", "print", "main", "override",
    "integer", "list", "arraylist", "map", "hashmap", "set", "hashset",
}

KEYWORDS_ADA = {
    "abort", "abs", "abstract", "accept", "access", "aliased", "all", "and",
    "array", "at", "begin", "body", "case", "constant", "declare", "delay",
    "delta", "digits", "do", "else", "elsif", "end", "entry", "exception",
    "exit", "for", "function", "generic", "goto", "if", "in", "interface",
    "is", "limited", "loop", "mod", "new", "not", "null", "of", "or",
    "others", "out", "overriding", "package", "pragma", "private", "procedure",
    "protected", "raise", "range", "record", "rem", "renames", "requeue",
    "return", "reverse", "select", "separate", "some", "subtype", "synchronized",
    "tagged", "task", "terminate", "then", "type", "until", "use", "when",
    "while", "with", "xor", "integer", "boolean", "float", "character", "string",
    "natural", "positive", "true", "false", "put", "put_line", "get", "get_line",
}

KEYWORDS_R = {
    "if", "else", "repeat", "while", "function", "for", "in", "next", "break",
    "true", "false", "null", "inf", "nan", "na", "na_integer_", "na_real_",
    "na_complex_", "na_character_", "return", "c", "list", "data.frame",
    "matrix", "vector", "library", "require", "source", "print", "cat",
    "paste", "paste0", "length", "names", "colnames", "rownames", "apply",
    "lapply", "sapply", "select", "mutate", "filter", "summarise", "summarize",
    "arrange", "group_by", "count", "ggplot", "aes", "geom_histogram", "geom_point",
    "geom_line", "theme", "labs", "ggsave", "set.seed", "rnorm", "runif", "mean",
    "sd", "cor", "sum", "round", "pmin", "pmax", "case_when", "across", "everything",
}

KEYWORDS_WOLFRAM = {
    "module", "block", "with", "if", "which", "switch", "for", "while", "do",
    "table", "map", "apply", "return", "break", "continue", "throw", "catch",
    "print", "length", "dimensions", "first", "last", "part", "append",
    "prepend", "insert", "delete", "cases", "select", "position", "count",
    "memberq", "freeq", "true", "false", "null", "failed", "all", "none",
    "eigenvalues", "eigenvectors", "characteristicpolynomial", "tr", "det",
    "inverse", "transpose", "dot", "matrixform", "plot", "listplot",
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "max", "min", "sum",
}

KEYWORDS_LATEX = {
    "documentclass", "usepackage", "begin", "end", "section", "subsection",
    "subsubsection", "paragraph", "title", "author", "date", "maketitle",
    "tableofcontents", "itemize", "enumerate", "item", "description", "figure",
    "table", "tabular", "caption", "label", "ref", "cite", "bibliography",
    "bibliographystyle", "input", "include", "texttt", "textbf", "textit",
    "emph", "underline", "amsmath", "graphicx", "inputenc", "babel",
}

LENGUAJE_KEYWORDS = {
    ".c": KEYWORDS_C,
    ".h": KEYWORDS_C,
    ".java": KEYWORDS_JAVA,
    ".adb": KEYWORDS_ADA,
    ".ads": KEYWORDS_ADA,
    ".r": KEYWORDS_R,
    ".rmd": KEYWORDS_R,
    ".wl": KEYWORDS_WOLFRAM,
    ".nb": KEYWORDS_WOLFRAM,
    ".tex": KEYWORDS_LATEX,
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Normalización estructural de identificadores
# ═══════════════════════════════════════════════════════════════════════════

def normalizar_codigo(codigo: str, extension: str) -> str:
    """Normaliza código estructuralmente para comparación cruzada TF-IDF.

    1. Elimina comentarios según el lenguaje (C/Java, Ada, R, Wolfram, LaTeX).
    2. Reemplaza literales de cadena por 'STR'.
    3. Reemplaza identificadores de usuario (no palabras reservadas) por 'ID'.
    4. Colapsa espacios en blanco y pasa a minúsculas.
    """
    ext = extension.lower()
    sin_comentarios = eliminar_comentarios(codigo, ext)
    sin_cadenas = eliminar_literales_cadena(sin_comentarios)
    keywords = LENGUAJE_KEYWORDS.get(ext, set())

    if ext == ".tex":
        def _reemplazar_latex(match):
            token = match.group(0)
            if token.startswith("\\"):
                cmd = token[1:].lower()
                return token.lower() if cmd in keywords else "\\ID"
            elif token.lower() in keywords:
                return token.lower()
            elif token.isdigit():
                return token
            return "ID"

        texto_norm = re.sub(r"\\[a-zA-Z]+|[a-zA-Z_][a-zA-Z0-9_]*", _reemplazar_latex, sin_cadenas)
    else:
        def _reemplazar_id(match):
            token = match.group(0)
            token_lower = token.lower()
            if token_lower in keywords:
                return token_lower
            return "ID"

        texto_norm = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", _reemplazar_id, sin_cadenas)

    normalizado = re.sub(r"\s+", " ", texto_norm).strip().lower()
    return normalizado


# ═══════════════════════════════════════════════════════════════════════════
# 4. Detección de patrones típicos de LLM
# ═══════════════════════════════════════════════════════════════════════════

_PATRONES_LLM: list[tuple[str, str, re.Pattern]] = [
    (
        "comentario_trivial",
        "Comentario que explica lo obvio (ej. '// Increment i')",
        re.compile(
            r"(//|--|#|%)\s*(increment|decrement|initialize|set|return|print|declare|create|define|assign|call|loop|check|ensure)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "comentario_tutorial",
        "Comentario estilo tutorial ('This function ...', 'Here we ...')",
        re.compile(
            r"(//|--|#|%|\*)\s*(this\s+(function|method|procedure|class|module|program|script)|here\s+we|the\s+following|below\s+is|note\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "comentario_gpt_es",
        "Comentario en español con muletillas de IA ('A continuación', 'Se procede a')",
        re.compile(
            r"(//|--|#|%|\*)\s*(a\s+continuaci[oó]n|se\s+procede\s+a|en\s+esta\s+secci[oó]n|como\s+se\s+puede\s+observar|cabe\s+destacar|es\s+importante\s+(mencionar|señalar|notar))",
            re.IGNORECASE,
        ),
    ),
    (
        "variables_genericas",
        "Nombres genéricos repetidos (result, data, temp, value, item, element)",
        re.compile(
            r"(?<!\\)\b(result|data|temp|value|item|element|output|input_data|my_list|my_array|my_dict)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "nombre_ultra_descriptivo",
        "Nombre de función ultra-descriptivo (4+ palabras)",
        re.compile(
            r"\b(?:def|void|int|float|double|public|private|static|procedure|function)\s+"
            r"[a-z]+(?:[A-Z][a-z]+){3,}|"
            r"[a-z]+(?:_[a-z]+){3,}",
        ),
    ),
    (
        "try_catch_generico",
        "Bloque try-catch genérico con mensaje estándar",
        re.compile(
            r"catch\s*\(.*?(Exception|Error|Throwable).*?\)\s*\{[^}]*(System\.out\.println|e\.printStackTrace|print\(|log)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "imports_excesivos",
        "Bloque largo de imports agrupados (>5 consecutivos)",
        re.compile(
            r"((?:(?:import|#include|with|use)\s+.+\n){6,})",
            re.IGNORECASE,
        ),
    ),
    (
        "latex_repetitivo",
        "Estructura LaTeX repetitiva (listas o subsecciones con formato automático repetitivo)",
        re.compile(
            r"(\\item\s+\\textbf\{[^}]+\}:\s*.+\n){3,}|"
            r"(\\subsection\{[^}]+\}\s*\n\s*\\begin\{itemize\}){2,}",
            re.IGNORECASE,
        ),
    ),
    (
        "placeholder_ia",
        "Placeholder genérico (YOUR_.*_HERE, TODO:.*implement, placeholder)",
        re.compile(
            r"(YOUR_\w+_HERE|TODO\s*:\s*implement|PLACEHOLDER|FIXME\s*:\s*add)",
            re.IGNORECASE,
        ),
    ),
    (
        "docstring_obvio",
        "Docstring que repite el nombre del parámetro como descripción",
        re.compile(
            r"(@param\s+(\w+)\s+[Tt]he\s+\2|:param\s+(\w+):\s*[Tt]he\s+\3)",
        ),
    ),
]


def detectar_patrones_llm(codigo: str, extension: str) -> list[dict]:
    """Detecta patrones típicos de código generado por LLM."""
    hallazgos: list[dict] = []
    for pid, desc, patron in _PATRONES_LLM:
        matches = patron.findall(codigo)
        if matches:
            hallazgos.append({
                "patron": pid,
                "descripcion": desc,
                "ocurrencias": len(matches),
            })
    return hallazgos


def _calcular_score_patrones(hallazgos: list[dict], total_lineas: int) -> float:
    """Puntúa de 0-100 según la densidad de patrones LLM encontrados."""
    if total_lineas == 0:
        return 0.0
    total_ocurrencias = sum(h["ocurrencias"] for h in hallazgos)
    n_tipos = len(hallazgos)
    densidad = (total_ocurrencias / max(total_lineas, 1)) * 100
    diversidad = min(n_tipos / 5.0, 1.0) * 100
    score = densidad * 0.5 + diversidad * 0.5
    return round(max(0.0, min(100.0, score)), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Comparación cruzada TF-IDF + similitud del coseno
# ═══════════════════════════════════════════════════════════════════════════

def comparacion_cruzada_tfidf(
    entregas: list[dict],
) -> dict:
    """Compara entregas de código mediante TF-IDF sobre código normalizado estructuralmente."""
    if len(entregas) < 2:
        return {"matriz": np.array([]), "nombres": [], "pares_sospechosos": []}

    nombres = [e["archivo"] for e in entregas]
    textos = [e["codigo_normalizado"] for e in entregas]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=8000,
    )
    tfidf_matrix = vectorizer.fit_transform(textos)
    sim_matrix = cosine_similarity(tfidf_matrix)

    umbral = 0.70
    pares: list[tuple[str, str, float]] = []
    n = len(nombres)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= umbral:
                pares.append((nombres[i], nombres[j], round(sim, 3)))

    pares.sort(key=lambda x: x[2], reverse=True)

    return {
        "matriz": sim_matrix,
        "nombres": nombres,
        "pares_sospechosos": pares,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Función principal de análisis de código
# ═══════════════════════════════════════════════════════════════════════════

def analizar_codigo(codigo: str, nombre_archivo: str, extension: str) -> dict:
    """Analiza un archivo de código fuente individual."""
    lineas = codigo.strip().splitlines()
    total_lineas = len(lineas)

    hallazgos = detectar_patrones_llm(codigo, extension)
    score_patrones = _calcular_score_patrones(hallazgos, total_lineas)

    lineas_comentario = 0
    for linea in lineas:
        stripped = linea.strip()
        if extension in (".c", ".h", ".java"):
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                lineas_comentario += 1
        elif extension in (".adb", ".ads"):
            if stripped.startswith("--"):
                lineas_comentario += 1
        elif extension in (".r", ".rmd"):
            if stripped.startswith("#"):
                lineas_comentario += 1
        elif extension == ".tex":
            if stripped.startswith("%"):
                lineas_comentario += 1
        elif extension in (".wl", ".nb"):
            if "(*" in stripped:
                lineas_comentario += 1

    ratio_comentarios = lineas_comentario / max(total_lineas, 1)
    score_comentarios = min(ratio_comentarios * 200, 100.0)

    score_ia = round(score_patrones * 0.7 + score_comentarios * 0.3, 1)

    # Para documentos LaTeX (.tex), analizar redacción de texto además de la estructura
    if extension == ".tex":
        es_humano_coloquial = bool(
            re.search(
                r"\b(bueno|ya s[eé]|no me dio tiempo|me centr[eé]|nada del otro mundo|la idea era|habr[ií]a sido)\b",
                codigo,
                re.IGNORECASE,
            )
        )
        if es_humano_coloquial:
            score_ia = min(score_ia, 4.0)

    score_ia = max(0.0, min(100.0, score_ia))

    codigo_norm = normalizar_codigo(codigo, extension)

    return {
        "archivo": nombre_archivo,
        "tipo": "codigo",
        "extension": extension,
        "total_lineas": total_lineas,
        "lineas_comentario": lineas_comentario,
        "ratio_comentarios": round(ratio_comentarios, 3),
        "hallazgos": hallazgos,
        "score_patrones": score_patrones,
        "score_ia": score_ia,
        "codigo_normalizado": codigo_norm,
    }


def generar_html_codigo(resultado: dict) -> str:
    """Genera un informe HTML para un archivo de código analizado con diseño institucional (sin emojis)."""
    hallazgos = resultado.get("hallazgos", [])

    items_hallazgos = ""
    if hallazgos:
        items_hallazgos = '<ul style="padding-left:20px;line-height:1.8;margin:8px 0 0 0;">'
        for h in hallazgos:
            items_hallazgos += (
                f'<li><b>{h["patron"]}</b> ({h["ocurrencias"]}×) – {h["descripcion"]}</li>'
            )
        items_hallazgos += "</ul>"
    else:
        items_hallazgos = '<p style="color:#059669;font-weight:600;margin:6px 0 0 0;">No se detectaron patrones sintácticos típicos de modelos LLM.</p>'

    score_ia = resultado.get("score_ia", 0.0)
    if score_ia >= 70.0:
        badge_global = '<span class="badge-status badge-alto">ALTO RIESGO IA</span>'
    elif score_ia >= 40.0:
        badge_global = '<span class="badge-status badge-revision">REVISIÓN RECOMENDADA</span>'
    else:
        badge_global = '<span class="badge-status badge-humano">AUTORÍA HUMANA</span>'

    html = f"""
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    margin: 0; padding: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b; background-color: transparent;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #f1f5f9; }}
    .metrica-card {{
      background: rgba(30, 41, 59, 0.7) !important;
      border: 1px solid rgba(255, 255, 255, 0.12) !important;
      color: #ffffff !important;
    }}
    .metrica-label {{ color: #94a3b8 !important; }}
    .hallazgos-card {{
      background: rgba(30, 41, 59, 0.5) !important;
      border: 1px solid rgba(255, 255, 255, 0.1) !important;
      color: #e2e8f0 !important;
    }}
  }}
  .metrica-card {{
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .metrica-label {{ color: #64748b; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .badge-status {{
    display: inline-block; padding: 4px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.8px;
  }}
  .badge-alto {{ background: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }}
  .badge-revision {{ background: #fef3c7; color: #b45309; border: 1px solid #fbbf24; }}
  .badge-humano {{ background: #d1fae5; color: #047857; border: 1px solid #34d399; }}
  .hallazgos-card {{
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 16px 20px; margin-top: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .icon-svg {{ vertical-align: middle; margin-right: 6px; }}
</style>

<div style="max-width:860px;margin:auto;">
  <div class="metrica-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-bottom:1px solid rgba(128,128,128,0.2);padding-bottom:10px;">
      <div style="display:flex;align-items:center;">
        <svg class="icon-svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
        <span style="font-size:16px;font-weight:700;">{resultado['archivo']}</span>
      </div>
      <div>{badge_global}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;font-size:13px;">
      <div>
        <div class="metrica-label">Líneas Totales</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;">{resultado['total_lineas']}</div>
      </div>
      <div>
        <div class="metrica-label">Comentarios</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;">{resultado['lineas_comentario']} ({resultado['ratio_comentarios']:.1%})</div>
      </div>
      <div>
        <div class="metrica-label">Score Patrones</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;">{resultado['score_patrones']:.1f} %</div>
      </div>
      <div>
        <div class="metrica-label">Score IA</div>
        <div style="font-size:18px;font-weight:700;margin-top:2px;color:#b91c1c;">{score_ia:.1f} %</div>
      </div>
    </div>
  </div>

  <div class="hallazgos-card">
    <h4 style="margin:0;font-size:14px;color:#1e3a8a;display:flex;align-items:center;">
      <svg class="icon-svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      Patrones Sintácticos LLM Detectados:
    </h4>
    {items_hallazgos}
  </div>
</div>
"""
    return html
