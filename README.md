# Predicción Multiclase de la Deserción Estudiantil mediante Máquinas de Soporte Vectorial: Análisis Comparativo de Métodos de Normalización

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://essential-bridges-newer-measure.trycloudflare.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://essential-bridges-newer-measure.trycloudflare.com/)
[![Cloudflare Tunnel](https://img.shields.io/badge/Demo%20en%20Vivo-Cloudflare%20Tunnel-F38020.svg?logo=cloudflare&logoColor=white)](https://essential-bridges-newer-measure.trycloudflare.com/)
[![LaTeX](https://img.shields.io/badge/LaTeX-MiKTeX%2025.12-green.svg)](https://miktex.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto de investigación para la asignatura **Machine Learning II (2026-II)** en el pregrado de Ciencia de Datos de la **Universidad Externado de Colombia**. El estudio evalúa el comportamiento de **Máquinas de Soporte Vectorial (SVM)** bajo dos funciones kernel (**RBF** y **Polinómico grado 3**), contrastando ocho configuraciones de normalización y preprocesamiento de datos sobre una tarea de clasificación multiclase inspirada en los lineamientos de **SPADIES 3.0**.

🌐 **[Acceder a la Aplicación Web en Vivo (Demo Cloudflare)](https://essential-bridges-newer-measure.trycloudflare.com/)** | 📄 **[Consultar el Artículo Final Machine Learning II](Articulo%20final%20Machine%20Learning%20II.pdf?raw=true)**

---

## 1. Fundamentación y Estrategia de Datos (SPADIES 3.0)

A diferencia de los enfoques binarios tradicionales que simplifican el problema, esta investigación adopta el marco conceptual del Sistema para la Prevención de la Deserción de la Educación Superior (**SPADIES 3.0**) del Ministerio de Educación Nacional de Colombia (MEN), modelando la trayectoria estudiantil en **tres estados académicos mutuamente excluyentes**:

* **En curso (44.67% · 5,137 registros):** Estudiantes activos matriculados que continúan su avance curricular regular.
* **Desertor (33.59% · 3,863 registros):** Estudiantes que abandonan la institución por al menos dos semestres consecutivos sin graduarse.
* **Graduado (21.74% · 2,500 registros):** Estudiantes que culminan exitosamente los créditos exigidos por su programa.

```text
Distribución del Dataset (N = 11,500):
  ├── En curso:  [██████████████████████] 44.67%
  ├── Desertor:  [████████████████]       33.59%
  └── Graduado:  [███████████]            21.74%
```

El conjunto de datos comprende **11,500 registros y 21 variables predictoras** (11 cuantitativas y 10 categóricas), construido mediante simulación paramétrica autorizada con fines docentes, reflejando factores académicos, demográficos, institucionales y socioeconómicos.

---

## 2. Arquitectura del Repositorio y del Código Modular

El código está estructurado bajo principios de ingeniería de software para aprendizaje automático: tipado estático, aislamiento estricto de particiones para impedir fuga de datos (*data leakage*), pipelines reproducibles de Scikit-Learn y herramientas de explicabilidad:

```text
Proyecto_Machine_Learning_II/
├── app.py                  # Aplicación interactiva en Streamlit para consulta en tiempo real
├── datos/
│   ├── brutos/             # Dataset sintético 11,500 y dataset base UCI
│   └── procesados/         # Almacenamiento intermedio de particiones
├── documentos/
│   ├── Articulo final Machine Learning II.pdf # Artículo científico final
│   ├── articulo/           # Artículo base de referencia IEEE BigData (Boteju et al., 2024)
│   ├── borradores/         # Entregas intermedias y documentos de trabajo
│   ├── presentacion/       # Diapositivas oficiales de sustentación
│   └── propuesta/          # Propuesta adaptada e informe técnico de 15 páginas
├── modelos/
│   ├── dummy_classifier.joblib        # Baseline trivial (Most Frequent)
│   ├── svm_pipeline_poly3.joblib      # Pipeline ganador en F1-Macro (Poly-3 + PowerTransformer)
│   ├── svm_pipeline_rbf.joblib        # Pipeline ganador en Recall Desertor (RBF + StandardScaler)
│   └── svm_pipeline_reproducible.joblib
├── notebooks/
│   ├── 01_EDA_KDD.ipynb               # Análisis Exploratorio de Datos y correlaciones
│   └── 02_Experimento_SVM_Normalizacion.ipynb # Validación y evaluación experimental
├── resultados/
│   ├── figuras/            # Curvas, PDP, SHAP summary, LIME y matrices de confusión
│   ├── metricas/           # CSVs de ensayos Monte Carlo, auditoría por subgrupos y pruebas
│   └── informe_resultados.md
└── src/
    ├── data/
    │   └── preprocessing.py# Limpieza, codificación y particionamiento estratificado
    ├── models/
    │   ├── pipeline.py     # Catálogo de los 8 preprocesamientos y ensamblado de Pipeline SVC
    │   ├── evaluate.py     # 50 repeticiones Monte Carlo, evaluación en test y auditoría
    │   └── interpretability.py # Explicabilidad del modelo: LIME, SHAP y PDP
    ├── utils/
    │   ├── config.py       # Configuración global, variables predictoras y rutas dinámicas
    │   └── metrics.py      # Métricas multiclase, intervalos de confianza y significancia
    ├── visualization/
    │   └── plots.py        # Generación integral de gráficas diagnósticas y explicativas
    └── script_svm.py       # Orquestador integral por línea de comandos
```

---

## 3. Protocolo Experimental y Resultados Cuantitativos

El diseño experimental consta de dos etapas rigurosas:
1. **Selección de Preprocesamiento (Fase de Desarrollo · 9,200 registros · 80%):** Se ejecutaron **50 repeticiones de Monte Carlo Repeated Holdout** (80/20 estratificado) sin fuga de datos, evaluando el impacto de 8 configuraciones sobre kernels **RBF** y **Poly-3**.
2. **Evaluación Definitiva en Test Reservado (2,300 registros · 20%):** Los mejores pipelines seleccionados se evaluaron una única vez sobre el conjunto ciego de prueba, contrastándolos contra un `DummyClassifier` (estrategia *most frequent*).

### 3.1. Comparativa de Preprocesamiento en Monte Carlo (50 Ensayos)

Resultados de $F_1$-Macro promedio ($\pm$ desviación estándar) a lo largo de las 50 iteraciones:

| Configuración de Preprocesamiento | SVM RBF ($F_1$-Macro) | SVM Poly-3 ($F_1$-Macro) | Comportamiento Geométrico |
| :--- | :---: | :---: | :--- |
| **Power Transformer (Yeo-Johnson)** | $0.8782 \pm 0.0072$ | **$0.8897 \pm 0.0059$** | **Mejor configuración para Poly-3**; estabiliza colas y asimetría |
| **Standard Scaler (Z-Score)** | **$0.8808 \pm 0.0072$** | $0.8866 \pm 0.0058$ | **Mejor configuración para RBF**; centra varianza unitaria |
| **Robust Scaler** | $0.8685 \pm 0.0078$ | $0.8744 \pm 0.0067$ | Mitiga valores atípicos mediante rango intercuartílico |
| **Ordinal Encoder** | $0.8376 \pm 0.0080$ | $0.8532 \pm 0.0071$ | Codificación ordinal sobre variables categóricas |
| **Quantile Transformer (Uniforme)** | $0.8328 \pm 0.0071$ | $0.8367 \pm 0.0074$ | Distorsiona distancias euclidianas en los márgenes |
| **Min-Max Scaler** | $0.8296 \pm 0.0081$ | $0.8313 \pm 0.0071$ | Acota al intervalo $[0, 1]$; sensible a extremos |
| **Max Absolute Scaler** | $0.8254 \pm 0.0086$ | $0.8348 \pm 0.0074$ | Preserva ceros pero comprime la escala relativa |
| **No Normalization (Línea Base)** | $0.4094 \pm 0.0061$ | $0.4977 \pm 0.0107$ | **Colapso severo:** dominancia de atributos con mayor escala |

### 3.2. Rendimiento sobre el Conjunto Ciego de Prueba (2,300 Registros)

| Modelo y Configuración | Accuracy | $F_1$-Score Macro | $F_1$-Score Ponderado | **Recall Desertor** |
| :--- | :---: | :---: | :---: | :---: |
| **SVM Poly-3 + PowerTransformer** | **0.8922** | **0.8923** | **0.8921** | 0.9082 |
| **SVM RBF + StandardScaler** | 0.8891 | 0.8885 | 0.8890 | **0.9211** |
| **DummyClassifier (Línea Base)** | 0.4465 | 0.2058 | 0.2757 | 0.0000 |

### 3.3. Desempeño Detallado por Estado Académico (Test Final)

| Modelo | Estado Académico | Precisión | Recall | $F_1$-Score | Matriz de Confusión (Aciertos) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM RBF** | Graduado | 0.8561 | 0.9040 | 0.8794 | 452 / 500 correctos |
| | En curso | 0.9101 | 0.8578 | 0.8832 | 881 / 1,027 correctos |
| | **Desertor** | 0.8856 | **0.9211** | **0.9030** | **712 / 773 correctos** (solo 61 FN) |
| **SVM Poly-3** | Graduado | 0.8821 | 0.8980 | 0.8900 | 449 / 500 correctos |
| | En curso | 0.9001 | 0.8773 | 0.8886 | 901 / 1,027 correctos |
| | **Desertor** | 0.8886 | 0.9082 | 0.8983 | 702 / 773 correctos (71 FN) |

> **Conclusión del Trade-off:**  
> * Si el objetivo institucional es el **equilibrio global**, **SVM Poly-3** es superior ($F_1$-Macro: **0.8923**).  
> * Si el objetivo prioritario de bienestar es la **detección temprana y rescate de estudiantes en riesgo**, **SVM RBF** es el modelo indicado, alcanzando un **Recall de Desertores del 92.11%** frente al $0.00\%$ del clasificador trivial.

---

## 4. Interpretabilidad del Modelo (XAI): LIME, SHAP y PDP

Para superar la limitación de opacidad de las SVM, el proyecto implementó una suite exhaustiva de explicabilidad global y local en `src/models/interpretability.py`:

1. **Importancia de Características por Permutación:**
   En ambos clasificadores, las variables con mayor capacidad discriminativa fueron:
   * `semestre_cursado`: $0.2268$ (RBF) y $0.2184$ (Poly-3)
   * `promedio_academico`: $0.1885$ (RBF) y $0.2016$ (Poly-3)
   * `materias_reprobadas`: $0.1800$ (RBF) y $0.1809$ (Poly-3)
   * `nivel_formacion`: $0.0951$ (RBF) y $0.1005$ (Poly-3)

2. **Gráficos de Dependencia Parcial (PDP):**
   * El puntaje de riesgo de deserción decrece marcadamente al superar un promedio de $3.0$ (reflejando el punto de inflexión académico).
   * El riesgo aumenta de forma casi monótona con cada materia reprobada adicional.

3. **Explicabilidad Local (LIME y SHAP):**
   * **LIME:** Al auditar estudiantes desertores individuales, condiciones como `promedio_academico <= 2.98` aportan pesos positivos locales de $+0.39$ (Poly-3) y $+0.36$ (RBF), mientras que `1.00 < materias_reprobadas <= 2.00` suma $+0.21$ y $+0.16$.
   * **SHAP:** Las fuerzas locales coinciden en situar a `materias_reprobadas` ($> +0.70$ en Poly-3, $+0.60$ en RBF) y `promedio_academico` como los motores principales de la predicción de abandono.

---

## 5. Auditoría Descriptiva por Subgrupos (Equidad Algorítmica)

Se examinó la consistencia del Recall para la categoría **Desertor** a través de subgrupos poblacionales para descartar sesgos desproporcionados:

* **Por Sexo:**
  * SVM RBF: Hombres $93.10\%$ · Mujeres $91.29\%$ (brecha: $1.81\%$)
  * SVM Poly-3: Hombres $90.23\%$ · Mujeres $91.29\%$ (brecha: $1.06\%$)
* **Por Zona de Residencia:**
  * SVM RBF: Rural $93.06\%$ · Urbana $91.89\%$ (brecha: $1.17\%$)
  * SVM Poly-3: Rural $93.75\%$ · Urbana $90.14\%$ (brecha: $3.61\%$)
* **Por Metodología de Estudio:**
  * SVM RBF: Virtual $96.70\%$ · Presencial $91.65\%$ · Distancia tradicional $90.84\%$
  * SVM Poly-3: Virtual $92.31\%$ · Presencial $90.74\%$ · Distancia tradicional $90.08\%$

---

## 6. Aplicación Web Interactiva (`app.py`)

🌐 **Enlace Demo en Vivo:** [https://essential-bridges-newer-measure.trycloudflare.com/](https://essential-bridges-newer-measure.trycloudflare.com/)

La aplicación web desarrollada en **Streamlit** se encuentra contenerizada mediante **Docker** y desplegada sobre un nodo ARM64 (Raspberry Pi), expuesta de manera segura a Internet a través de un túnel cifrado de **Cloudflare Tunnel (`cloudflared`)**:

* **Pestaña 1 · Predicción Individual:** Sliders y selectores interactivos para ingresar el perfil de un estudiante, contrastar las salidas de RBF y Poly-3, y generar explicaciones LIME y SHAP inmediatas.
* **Pestaña 2 · Métricas y Matrices:** Visualizador de matrices de confusión interactivas, métricas comparativas y curvas ROC.
* **Pestaña 3 · Interpretabilidad Global:** Gráficos interactivos de Dependencia Parcial (PDP) e importancia de variables por permutación.
* **Pestaña 4 · Auditoría por Subgrupos:** Desglose interactivo de equidad descriptiva.

---

## 7. Instrucciones de Instalación y Reproducibilidad

### 7.1. Prerrequisitos y Configuración del Entorno Local
```bash
git clone https://github.com/eljarcor98/Proyecto_Machine_Learning_II.git
cd Proyecto_Machine_Learning_II
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2. Despliegue con Docker y Cloudflare Tunnel (Producción / Edge Computing)
El repositorio incluye archivos listos para contenedores (`Dockerfile` y `docker-compose.yml`):
```bash
# Construir y levantar el contenedor en segundo plano (puerto 8502)
docker compose up -d --build

# Exponer públicamente a Internet con Cloudflare Tunnel
docker run -d --name cf_tunnel --restart unless-stopped --network host cloudflare/cloudflared:latest tunnel --url http://localhost:8502
```

### 7.3. Ejecutar la Aplicación Web en Local (Streamlit)
```bash
streamlit run app.py
```

### 7.4. Ejecutar el Pipeline Completo por Terminal
El script ejecuta el preprocesamiento, los 50 trials de Monte Carlo para ambos kernels, la evaluación en test, las explicaciones LIME/SHAP/PDP y guarda los modelos serializados:
```bash
python src/script_svm.py --trials 50 --jobs -1
```

### 7.5. Cuadernos Interactivos (Jupyter)
```bash
jupyter notebook notebooks/02_Experimento_SVM_Normalizacion.ipynb
```

---

## 8. Limitaciones del Estudio

1. **Naturaleza Sintética de los Datos:** El dataset de 11,500 registros fue construido mediante simulación paramétrica con fines docentes bajo reglas determinísticas basadas en SPADIES 3.0; por lo tanto, no describe estudiantes reales ni certifica causalidad.
2. **Evaluación de Hiperparámetros:** Se mantuvieron los valores por defecto ($C=1.0$, $\gamma=\text{scale}$); una búsqueda exhaustiva vía Grid/Bayesian Search sobre datos reales representa el paso natural de trabajo futuro.
3. **Explicabilidad:** Los valores SHAP y pesos LIME son aproximaciones numéricas descriptivas de las funciones de decisión, no probabilidades calibradas ni diagnósticos clínicos.

---

## 9. Autores e Información Institucional

* **Sara López Chavarro**
* **Nicolás Figueroa Salgado**
* **Arnold Santiago Torres Anzola**

**Universidad Externado de Colombia**  
Facultad de Matemáticas — Pregrado en Ciencia de Datos  
Asignatura: *Machine Learning II* (2026-II)  
Bogotá D.C., Colombia

### Referencias Principales
* [1] G. Boteju, L. Tang, y M. Scott Brown, *"Support Vector Machine for Predicting Student Dropout Under Different Normalization Methods"*, IEEE BigData, pp. 8633–8638, 2024.
* [2] Ministerio de Educación Nacional de Colombia, *"SPADIES 3.0: Cambio Metodológico"*, MEN, Bogotá D.C., 2024.
