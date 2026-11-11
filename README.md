# Máquinas de Soporte Vectorial para la Predicción Temprana de Deserción Estudiantil Universitaria

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-MiKTeX%2025.12-green.svg)](https://miktex.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto de investigación para **Machine Learning II (2026-II)** enfocado en la predicción temprana de deserción universitaria mediante **Máquinas de Soporte Vectorial (SVM)** con kernel RBF, analizando rigurosamente el impacto de 8 técnicas de normalización y preprocesamiento de datos sobre la geometría del margen de decisión.

📄 **[Consultar el Artículo Científico Completo en PDF (15 Páginas)](Maquinas_de_Soporte_Vectorial_para_la_Prediccion_Temprana_de_Desercion_Estudiantil_Universitaria.pdf?raw=true)**

---

## 1. Fundamentación y Estrategia de Datos (SPADIES 3.0)

A diferencia de los enfoques binarios tradicionales que simplifican el problema, esta investigación adopta los lineamientos oficiales del Ministerio de Educación Nacional de Colombia (**SPADIES 3.0**), modelando la trayectoria estudiantil como una tarea multiclase de tres estados:
* **En curso (44.67%):** Estudiantes activos matriculados que continúan su avance curricular.
* **Desertor (33.59%):** Estudiantes que abandonan la institución por al menos dos semestres consecutivos sin graduarse.
* **Graduado (21.74%):** Estudiantes que culminan exitosamente su programa académico.

El dataset cuenta con **11,500 registros** y 21 atributos multidisciplinarios (académicos, socioeconómicos, demográficos e institucionales).

---

## 2. Arquitectura Modular del Código (`src/`)

El proyecto implementa una arquitectura desacoplada, tipada, con docstrings estándar y libre de fuga de datos (*data leakage*):

```text
src/
├── utils/
│   ├── config.py           # Rutas dinámicas (Pathlib), constantes, variables y paletas
│   └── metrics.py          # Métricas multiclase, t-test de Welch vs baseline y auditoría de equidad
├── data/
│   └── preprocessing.py    # Carga de datos con fallbacks, cálculo SPADIES 3.0, IQR y train/test split
├── models/
│   ├── pipeline.py         # Catálogo de 8 escaladores, ColumnTransformer y ensamble de Pipeline SVC
│   └── evaluate.py         # 50 particiones Monte Carlo paralelas, serialización e importancia por permutación
├── visualization/
│   └── plots.py            # Suite de visualización: EDA, escaladores, significancia, frontera real y márgenes
└── script_svm.py           # Orquestador integral reproducible vía terminal
```

---

## 3. Resultados Experimentales (50 Ensayos Monte Carlo)

Evaluación sobre 50 particiones independientes ($80\%$ train, $20\%$ test estratificado) sin fuga de datos:

| Técnica de Normalización | $F_1$-Score Ponderado | $F_1$-Score Macro | $F_1$ (Desertor) | Desv. Est. ($\sigma$) | $\Delta F_1$ vs Baseline | Estadístico $t$ | Valor $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **StandardScaler** | **0.8878** | **0.8878** | **0.9001** | 0.0064 | **+0.4086 (+85.3%)** | 48.72 | $< 0.001$ |
| **PowerTransformer** | 0.8860 | 0.8861 | 0.8981 | 0.0071 | **+0.4068 (+84.9%)** | 46.15 | $< 0.001$ |
| **RobustScaler** | 0.8787 | 0.8785 | 0.8818 | 0.0079 | +0.3995 (+83.4%) | 42.31 | $< 0.001$ |
| **QuantileTransformer** | 0.8426 | 0.8448 | 0.8309 | 0.0085 | +0.3634 (+75.8%) | 35.18 | $< 0.001$ |
| **MinMaxScaler** | 0.8406 | 0.8398 | 0.8238 | 0.0089 | +0.3614 (+75.4%) | 34.62 | $< 0.001$ |
| **MaxAbsScaler** | 0.8380 | 0.8371 | 0.8207 | 0.0092 | +0.3588 (+74.9%) | 33.95 | $< 0.001$ |
| **No Normalization (Baseline)** | 0.4792 | 0.4097 | 0.6295 | 0.0114 | Referencia | 0.00 | 1.000 |
| **Ordinal Encoder** | 0.3642 | 0.3464 | 0.0000 | 0.0121 | -0.1150 (-24.0%) | -14.83 | $< 0.001$ |

**Hallazgos Principales:**
* **Colapso del Baseline:** Sin normalización, SVM no logra equilibrar variables con distintas magnitudes (`distancia_hogar_ies_km` vs `promedio_academico`), desplomando el rendimiento a $F_1 \approx 0.41$.
* **Eficacia de StandardScaler:** Maximiza la geometría del kernel RBF y alcanza un **Recall de Desertores del $92.11\%$** en prueba con solo $7.89\%$ de falsos negativos.
* **Equidad Algorítmica (Fairness):** La auditoría por subgrupos comprobó una dispersión menor a 4 puntos porcentuales en el Recall entre hombres/mujeres ($91.97\%$ vs $92.25\%$) y zonas rurales/urbanas ($93.38\%$ vs $91.84\%$).

---

## 4. Interpretabilidad Visual y Fronteras de Decisión

Para superar la noción de "caja negra", el proyecto prescinde de proyecciones abstractas (PCA) y analiza directamente la geometría del hiperplano sobre variables reales:

1. **Frontera de Decisión en Variables Críticas (`promedio_academico` vs `materias_reprobadas`):**
   * **Línea continua roja ($f(x)=0$):** Umbral exacto que activa la predicción de deserción.
   * **Líneas discontinuas ($f(x)=\pm 1$):** Corredor del margen geométrico de seguridad de SVM.
   * **Vectores de Soporte (halos negros):** Representan a los **estudiantes en riesgo crítico en la frontera** (ej. promedio ~3.0 con 1 o 2 reprobadas), quienes constituyen la población con mayor retorno institucional ante intervenciones tempranas.
2. **Distribución de Distancia Funcional al Hiperplano ($f(x)$):**
   * Estratifica a los estudiantes en tres zonas operativas: *Deserción Consolidada* ($28.3\%$, $f(x) < -1$), *Margen Crítico / Vectores de Soporte* ($19.4\%$, $-1 \le f(x) \le 1$) y *Permanencia Segura* ($52.3\%$, $f(x) > 1$).

---

## 5. Instrucciones de Instalación y Ejecución

### Prerrequisitos
Clonar el repositorio y configurar el entorno virtual:
```bash
git clone https://github.com/eljarcor98/Proyecto_Machine_Learning_II.git
cd Proyecto_Machine_Learning_II
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar el Pipeline Completo por Terminal
El script orquestador ejecuta el análisis exploratorio, los 50 trials concurrentes, guarda el modelo serializado, exporta métricas y genera todas las figuras:
```bash
python src/script_svm.py --trials 50 --jobs -1
```

### Ejecución Interactiva (Jupyter Notebook)
Para explorar el experimento paso a paso de forma pedagógica:
```bash
jupyter notebook notebooks/02_Experimento_SVM_Normalizacion.ipynb
```

### Compilar el Artículo Científico en LaTeX
```bash
cd documentos/propuesta
pdflatex Propuesta_Adaptacion_SVM_Desercion.tex
pdflatex Propuesta_Adaptacion_SVM_Desercion.tex
```

---

## 6. Autores e Institución

* **Sara López Chavarro**
* **Nicolás Figueroa Salgado**
* **Arnold Santiago Torres Anzola**

**Universidad Externado de Colombia**  
Departamento de Matemáticas — Programa de Ciencia de Datos  
Asignatura: *Machine Learning II* (2026-II)  
Bogotá D.C., Colombia
