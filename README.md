# Predicción Temprana de Deserción Estudiantil Universitaria mediante Máquinas de Soporte Vectorial (SVM) y Evaluación de Métodos de Normalización

Este repositorio contiene el desarrollo del proyecto de investigación y modelamiento predictivo para la **Machine Learning II (2026-II)**. El trabajo aborda el problema de la deserción universitaria mediante la aplicación de **Máquinas de Soporte Vectorial (SVM)**, analizando de manera sistemática el impacto de múltiples técnicas de normalización y transformación de datos sobre el rendimiento del clasificador y la geometría del hiperplano de separación.

---

## 1. Descripción del Problema

La deserción en la educación superior es un fenómeno multidimensional desencadenado por factores académicos, socioeconómicos, personales e institucionales. La identificación temprana de estudiantes en condición de vulnerabilidad permite a las instituciones implementar estrategias preventivas y priorizar la asignación de recursos de tutoría y apoyo financiero.

Desde la perspectiva del aprendizaje automático, este problema se formula como una tarea de **clasificación binaria dependiente del tiempo**. Las Máquinas de Soporte Vectorial son una técnica adecuada para este problema debido a su capacidad para encontrar el hiperplano óptimo de separación maximizando el margen entre clases en espacios de alta dimensión. Sin embargo, al basarse en el cálculo de distancias geométricas en el espacio de características, la escala y la distribución de las variables afectan significativamente la localización del hiperplano y los vectores de soporte seleccionados.

---

## 2. Descripción de los Datos

El análisis se fundamenta en un conjunto de datos sintético representativo de una población universitaria, complementado con un dataset estandarizado de referencia internacional:

### Dataset Principal (`estudiantes_sinteticos_11500.csv`)
* **Dimensión:** 11,500 registros de estudiantes.
* **Variables Académicas:** Promedio acumulado, cantidad de asignaturas reprobadas, créditos matriculados por semestre, tasa de cancelación de cursos.
* **Variables Socioeconómicas:** Estrato socioeconómico, nivel de ingresos familiares, tipo de colegio de procedencia (oficial/privado), distancia física al campus universitario.
* **Variables Conductuales y de Apoyo:** Frecuencia de asistencia a tutorías, uso de plataformas virtuales de aprendizaje, posesión de beca o auxilio económico.
* **Variable Objetivo ($Y \in \{0, 1\}$):** Estado final del estudiante (0: Persistente / Matriculado, 1: Desertor).

### Dataset de Validación Externa (UCI Repository)
* **Dataset:** *Predict Students Dropout and Academic Success*.
* **Propósito:** Validar la capacidad de generalización del pipeline de normalización y modelamiento sobre datos reales de instituciones de educación superior internacional.

---

## 3. Metodología y Modelamiento

### Formulación Matemática del Clasificador SVM
El objetivo del clasificador SVM de margen blando (*Soft-Margin SVM*) consiste en resolver el siguiente problema de optimización cuadrática:

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \xi_i$$

sujeto a las restricciones de clasificación con holgura:

$$y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i \in \{1, \dots, N\}$$

donde $w$ define el vector normal al hiperplano, $C > 0$ es el parámetro de regularización, y $\xi_i$ representa la variable de holgura para observaciones violadoras del margen.

### Evaluación de Métodos de Normalización
Dado que la escala de los predictores influye directamente en el término $\|w\|^2$, se evaluaron experimentalmente 8 métodos de transformación:
1. **Sin Normalización (Baseline):** Evaluación con variables en su escala original.
2. **Standard Scaler:** Estandarización a media cero y varianza unitaria ($z = \frac{x - \mu}{\sigma}$).
3. **Min-Max Scaler:** Escalamiento acotado al intervalo $[0, 1]$.
4. **Max-Abs Scaler:** Escalamiento proporcional al valor absoluto máximo.
5. **Ordinal Encoder:** Transformación secuencial de características categóricas.
6. **One-Hot Encoder:** Codificación binaria vectorial para variables categóricas.
7. **Quantile Transformer:** Transformación no lineal hacia una distribución uniforme o normal basada en mapeo de cuantiles.
8. **Power Transformer (Yeo-Johnson):** Transformación estabilizadora de varianza y corrección de asimetría (*skewness*).

---

## 4. Resultados Experimentales

La evaluación se realizó mediante validación cruzada ($k=5$), reportando el puntaje $F_1$, la desviación estándar y las pruebas de significancia estadística ($t$-test de Student de muestras independientes) frente al modelo baseline.

| Método de Normalización | F1-Score Promedio | Desviación Estándar | Valor $p$ ($t$-test vs Baseline) |
| :--- | :---: | :---: | :---: |
| **Quantile Transformer** | **0.7686** | 0.0201 | $p < 0.001$ |
| **Power Transformer** | **0.7683** | 0.0220 | $p < 0.001$ |
| **One-Hot Encoder** | 0.7613 | 0.0222 | $p < 0.001$ |
| **Standard Scaler** | 0.7599 | 0.0222 | $p < 0.001$ |
| **Min-Max Scaler** | 0.7340 | 0.0246 | $p < 0.001$ |
| **Max-Abs Scaler** | 0.7334 | 0.0230 | $p < 0.001$ |
| **Ordinal Encoder** | 0.5605 | 0.0216 | $p < 0.001$ |
| **Sin Normalización (Baseline)** | **0.0000** | 0.0000 | Referencia ($p = 1.0$) |

### Conclusiones Principales:
1. **Colapso del Modelo Baseline:** El modelo sin normalizar obtuvo un $F_1 = 0.0000$, lo cual confirma de forma empírica que las variables con mayores magnitudes dominan la función de distancia, colapsando la capacidad discriminativa del hiperplano.
2. **Ventaja de las Transformaciones No Lineales:** Las técnicas `QuantileTransformer` ($F_1 = 0.7686$) y `PowerTransformer` ($F_1 = 0.7683$) registraron el mejor comportamiento al mitigar la influencia de valores atípicos (*outliers*) y corregir la asimetría de variables socioeconómicas.
3. **Importancia por Permutación (*Permutation Importance*):** Los factores con mayor contribución a la reducción de entropía e impacto en la frontera de decisión fueron el **promedio académico acumulado**, las **asignaturas reprobadas** y los **ingresos familiares**.

---

## 5. Estructura del Proyecto

El código y los artefactos del proyecto se organizan bajo la siguiente estructura modular:

```text
Proyecto/
├── README.md                   # Documentación principal del proyecto
├── datos/
│   ├── brutos/                 # Datasets originales (estudiantes_sinteticos_11500.csv y uci_dataset.csv)
│   └── procesados/             # Datasets filtrados y transformados
├── documentos/
│   ├── articulo/               # Artículos científicos y reporte técnico en PDF
│   └── propuesta/              # Documentos de propuesta formal en formato .docx y .tex
├── modelos/                    # Artefactos de modelos serializados (.joblib, .pkl)
├── notebooks/                  # Cuadernos Jupyter experimentales
│   ├── 01_EDA_KDD.ipynb        # Análisis Exploratorio de Datos (EDA)
│   └── 02_Experimento_SVM_Normalizacion.ipynb
├── resultados/                 # Salidas y evidencias del experimento
│   ├── figuras/                # Visualizaciones (hiperplano PCA, matriz de confusión, distribuciones)
│   └── metricas/               # Archivos CSV con tablas comparativas de desempeño
└── src/                        # Código fuente modular en Python
    ├── data/                   # Funciones para ingesta y limpieza de datos
    ├── models/                 # Scripts para entrenamiento, sintonía e inferencia de SVM
    ├── utils/                  # Funciones auxiliares y cálculo de métricas
    ├── visualization/          # Scripts para generación de gráficos del hiperplano
    └── script_svm.py           # Pipeline ejecutable del experimento completo
```

---

## 6. Instrucciones de Ejecución

### Requisitos Previos
* Python 3.9 o superior.
* Entorno virtual recomendado (`venv` o `conda`).

### Configuración y Ejecución
1. Clonar el repositorio y acceder al directorio raíz:
   ```bash
   git clone https://github.com/eljarcor98/Proyecto_Machine_Learning_II.git
   cd Proyecto_Machine_Learning_II
   ```

2. Ejecutar el script principal de entrenamiento y evaluación de SVM:
   ```bash
   python src/script_svm.py
   ```

3. Las figuras resultantes se exportarán automáticamente a `resultados/figuras/` y la tabla de métricas a `resultados/metricas/resultados_svm_normalizacion.csv`.

---
