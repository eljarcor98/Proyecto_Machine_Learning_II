# Predicción Temprana de Deserción Estudiantil Universitaria mediante Máquinas de Soporte Vectorial (SVM) y Evaluación de Métodos de Normalización

Este repositorio contiene el desarrollo del proyecto de investigación y modelamiento predictivo para la **Machine Learning II (2026-II)**. El trabajo aborda el problema de la deserción universitaria aplicando **Máquinas de Soporte Vectorial (SVM)** y evaluando de manera sistemática el impacto de diferentes técnicas de normalización y transformación de datos sobre el rendimiento del clasificador.

### Documento Principal del Proyecto (PDF)
El artículo formal completo se encuentra disponible directamente en la raíz de este repositorio para su consulta inmediata:
- **[Maquinas_de_Soporte_Vectorial_para_la_Prediccion_Temprana_de_Desercion_Estudiantil_Universitaria.pdf](Maquinas_de_Soporte_Vectorial_para_la_Prediccion_Temprana_de_Desercion_Estudiantil_Universitaria.pdf)**

---

## 1. Descripción del Problema

La deserción en la educación superior es un fenómeno multidimensional influenciado por factores académicos, socioeconómicos y personales. La identificación temprana de estudiantes en riesgo permite a las instituciones implementar estrategias preventivas y optimizar la asignación de recursos de apoyo.

Desde el aprendizaje automático, este problema se modela como una tarea de **clasificación binaria dependiente del tiempo**. Se implementaron modelos de **Máquinas de Soporte Vectorial (SVM)** para encontrar la frontera óptima de separación entre estudiantes persistentes y en riesgo de deserción.

---

## 2. Descripción de los Datos

El análisis se fundamenta en un conjunto de datos representativo y un dataset de validación externa:

- **Dataset Principal (`estudiantes_sinteticos_11500.csv`):** 11,500 registros con variables académicas (promedio acumulado, asignaturas reprobadas, créditos matriculados), socioeconómicas (estrato, ingresos, tipo de colegio) y conductuales (asistencia a tutorías, uso de plataformas). La variable objetivo distingue entre estudiantes persistentes (0) y desertores (1).
- **Dataset de Validación Externa (UCI Repository):** Utilizado para contrastar y validar la generalización del pipeline sobre datos reales internacionales.

---

## 3. Metodología y Evaluación de Normalización

Dado que la escala de los predictores influye directamente en el cálculo de distancias de las SVM, se evaluaron experimentalmente 8 métodos de normalización y transformación de características:
1. **Sin Normalización (Baseline)**
2. **Standard Scaler**
3. **Min-Max Scaler**
4. **Max-Abs Scaler**
5. **Ordinal Encoder**
6. **One-Hot Encoder**
7. **Quantile Transformer**
8. **Power Transformer (Yeo-Johnson)**

---

## 4. Resultados Experimentales

La evaluación se realizó mediante validación cruzada ($k=5$), midiendo el puntaje $F_1$, desviación estándar y pruebas de significancia ($t$-test vs. baseline).

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
1. **Colapso del Modelo Baseline:** Sin normalización, el modelo obtuvo un $F_1 = 0.0000$ debido a la dominancia de las variables con mayores magnitudes numéricas.
2. **Superioridad No Lineal:** Las técnicas `QuantileTransformer` y `PowerTransformer` lograron el mejor desempeño al mitigar el impacto de valores atípicos y corregir asimetrías.
3. **Variables Críticas:** El promedio acumulado, las asignaturas reprobadas y los ingresos familiares resultaron ser los predictores con mayor peso en la decisión del modelo.

---

## 5. Estructura del Repositorio

```text
Proyecto/
├── README.md                      # Documentación del proyecto
├── datos/
│   ├── brutos/                    # Datasets originales
│   └── procesados/                # Datasets filtrados y transformados
├── documentos/                    # Artículos científicos y reportes en PDF / Word
├── modelos/                       # Artefactos de modelos serializados (.joblib, .pkl)
├── notebooks/                     # Cuadernos Jupyter experimentales (EDA y Modelamiento)
├── resultados/                    # Figuras, gráficos del hiperplano y tablas de métricas
└── src/                           # Código fuente modular en Python
    ├── data/                      # Ingesta y limpieza
    ├── models/                    # Entrenamiento e inferencia SVM
    ├── utils/                     # Métricas auxiliares
    ├── visualization/             # Generación de gráficos
    └── script_svm.py              # Pipeline ejecutable completo
