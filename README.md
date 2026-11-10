# Predicción Temprana de Deserción Estudiantil (SVM y Normalización)

Proyecto de investigación para **Machine Learning II (2026-II)** enfocado en predecir la deserción universitaria mediante **Máquinas de Soporte Vectorial (SVM)**, evaluando el impacto de diferentes técnicas de normalización sobre el rendimiento del clasificador.

📄 **[Artículo Científico Principal (PDF)](Maquinas_de_Soporte_Vectorial_para_la_Prediccion_Temprana_de_Desercion_Estudiantil_Universitaria.pdf)**

---

## 1. Datos y Enfoque

El problema se modela como una tarea de clasificación binaria dependiente del tiempo utilizando dos fuentes:
- **Dataset Principal:** 11,500 registros sintéticos con variables académicas, socioeconómicas y conductuales.
- **Dataset UCI:** Repositorio externo para validación de generalización.

---

## 2. Evaluación de Normalización

Se evaluaron 8 técnicas de transformación para mitigar el impacto de las escalas en la geometría del hiperplano de las SVM:
1. Sin Normalización (Baseline)
2. Standard Scaler
3. Min-Max Scaler
4. Max-Abs Scaler
5. Ordinal Encoder
6. One-Hot Encoder
7. Quantile Transformer
8. Power Transformer (Yeo-Johnson)

---

## 3. Resultados Experimentales

Evaluación mediante validación cruzada ($k=5$) y métrica $F_1$:

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

**Conclusiones clave:**
- El modelo sin normalizar colapsa ($F_1 = 0.0000$) por la dominancia de magnitudes.
- Las transformaciones no lineales (`Quantile` y `Power Transformer`) obtuvieron el mejor rendimiento al corregir asimetrías y valores atípicos.

---

## 4. Estructura del Repositorio

```text
Proyecto/
├── README.md
├── datos/              # Brutos y procesados
├── documentos/         # Artículos y reportes (PDF/Word)
├── modelos/            # Artefactos serializados (.joblib, .pkl)
├── notebooks/          # Jupyter Notebooks experimentales
├── resultados/         # Figuras y tablas de métricas
└── src/                # Código fuente modular en Python
