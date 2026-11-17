# AuditorIA — Sistema de Integridad Academica

**Deteccion automatizada de contenido generado por IA en entregas universitarias.**  
Analisis NLP multilingue, auditoria estructural de codigo y OCR offline para examenes escaneados.

---

## Capacidades del Sistema

### Deteccion Textual (NLP Multilingue)
- Calculo de **perplejidad** mediante `bigscience/bloom-560m` (modelo multilingue, optimizado para textos en espanol)
- Calculo de **burstiness** (varianza de longitud de oraciones)
- **Score IA combinado** con clasificacion en tres niveles: Alto Riesgo, Revision Recomendada, Autoria Humana
- Tres **perfiles de hardware** para adaptarse a cualquier equipo:
  - Gama Baja / PC Oficina (CPU lenta, <=8 GB RAM) — BLOOM cuantizado INT8, 5 bloques representativos
  - Gama Media / Portatil Moderno — BLOOM cuantizado INT8, analisis completo por lotes
  - Gama Alta / PC con GPU — Precision float32/float16 completa, ventana deslizante

### Auditoria Estructural de Codigo
- Similitud cruzada entre entregas mediante **TF-IDF + Similitud del Coseno**
- **Normalizacion de identificadores** (variables, funciones y parametros reemplazados por token `ID`)
- Deteccion de patrones tipicos de LLMs en lenguajes academicos: **Ada (.adb, .ads), C/C++ (.c, .h), Java (.java), R (.r, .rmd), Wolfram (.wl, .nb) y LaTeX (.tex)**
- En documentos **LaTeX (.tex)**, analiza tanto el texto redactado como la estructura sintactica
- Resistente a renombrado de variables: si dos alumnos usan el mismo codigo IA con variables distintas, la similitud estructural supera el 95%

### OCR Offline para Documentos Escaneados
- Soporte para **PDFs escaneados** (imagenes incrustadas sin texto seleccionable)
- Soporte para **imagenes sueltas** (.png, .jpg, .jpeg)
- Motor OCR: `RapidOCR` (100% offline, sin dependencias externas como Tesseract)
- Renderizado automatico de paginas a imagen con `PyMuPDF` (150 DPI) cuando hay menos de 20 caracteres de texto nativo
- La tabla resumen indica si el documento fue leido via **PDF (nativo)** u **OCR**

---

## Requisitos

- **Python 3.10 o superior** (3.12 recomendado)
- Conexion a internet **solo la primera vez** para descargar dependencias y modelos
- Espacio en disco: ~2 GB (modelos + entorno virtual)

---

## Inicio Rapido

### Windows (doble clic)

```
iniciar.bat
```

o desde cmd/PowerShell:

```bat
iniciar_windows.bat
```

El script comprueba si existe el entorno virtual; si no, lo crea e instala todas las dependencias automaticamente. A continuacion abre `http://localhost:8501`.

---

### Linux

```bash
chmod +x iniciar_linux.sh
./iniciar_linux.sh
```

El script crea `venv_linux/` la primera vez e instala todo automaticamente.

---

### macOS

```bash
chmod +x iniciar_mac.command
./iniciar_mac.command
```

O haz doble clic en `iniciar_mac.command` desde el Finder (puede necesitar permiso en Preferencias de Seguridad la primera vez).

---

## Uso Portable en USB sin Internet

1. Ejecuta AuditorIA **una vez con conexion** para que se descarguen los modelos en `modelos_cache/`.
2. Copia **toda la carpeta del proyecto** al USB, incluyendo:
   - `modelos_cache/` — modelos Hugging Face offline (~1.1 GB)
   - `venv/` / `venv_linux/` / `venv_mac/` — entorno Python preinstalado
3. En el equipo destino (sin internet), ejecuta el lanzador de tu SO.
4. El sistema cargara los modelos desde `modelos_cache/` sin necesidad de red.

> **Nota:** Si el USB se copia a un equipo con un SO distinto, usa el lanzador correspondiente para que cree su propio `venv_*/` la primera vez (si hay internet) o copia tambien ese entorno si es el mismo SO.

---

## Estructura del Proyecto

```
AuditorIA/
├── app.py                  # Interfaz Streamlit (UI corporativa)
├── motor_texto.py          # Motor NLP: perplejidad, burstiness, OCR
├── motor_codigo.py         # Motor de auditoria de codigo
├── requirements.txt        # Dependencias Python
├── iniciar.bat             # Lanzador Windows (doble clic)
├── iniciar_windows.bat     # Lanzador Windows (alias)
├── iniciar_linux.sh        # Lanzador Linux
├── iniciar_mac.command     # Lanzador macOS (doble clic desde Finder)
├── modelos_cache/          # Cache local de modelos HF (gitignored)
│   └── README_OFFLINE.md   # Instrucciones de pre-descarga
├── pruebas/                # Archivos de prueba (.pdf, .py, .tex, .png)
├── test_sistema.py         # Suite completa de pruebas automatizadas
├── test_modelo.py          # Verificacion de perplejidad y generacion HTML
└── test_estres_zip.py      # Simulacion de carga masiva y robustez ante fallos
```

---

## Dependencias Principales

| Libreria | Version minima | Uso |
|---|---|---|
| streamlit | 1.32.0 | Interfaz web |
| transformers | 4.40.0 | Modelo BLOOM NLP |
| torch | 2.2.0 | Backend PyTorch |
| rapidocr-onnxruntime | 1.3.0 | OCR offline |
| pymupdf | 1.23.0 | Lectura de PDFs |
| scikit-learn | 1.4.0 | TF-IDF + Similitud |
| pandas | 2.2.0 | Tabla de resultados |
| openpyxl | 3.1.0 | Exportacion Excel |

---

## Licencia

Proyecto academico desarrollado para uso institucional universitario.  
Uso interno — no distribuir sin autorizacion.
