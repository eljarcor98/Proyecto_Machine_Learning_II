# Informe de Integración y Mejoras: Experimento SVM Multiclase (SPADIES 3 Clases)

**Proyecto:** Predicción de Deserción Estudiantil sobre Dataset Sintético (11,500 Estudiantes)  
**Autor:** Arnold Santiago Torres | **Curso:** Machine Learning II (2026-II)  
**Archivos Relevantes:**
- [`script_svm.py`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/script_svm.py)
- [`Experimento_SVM_Normalizacion.ipynb`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/Experimento_SVM_Normalizacion.ipynb)
- [`resultados_svm_normalizacion.csv`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/resultados_svm_normalizacion.csv)

---

## 1. Justificación del Enfoque Exclusivo en Máquinas de Vectores de Soporte (SVM)

> [!NOTE]
> **Aclaración Metodológica**: Se descartaron explícitamente los modelos de ensamble basados en árboles (*Random Forest*, *Decision Trees*, *XGBoost*). El estudio y la propuesta se centran **única y exclusivamente en la arquitectura SVM**, evaluando el comportamiento de su hiperplano de separación frente a distintos métodos de normalización.

* **Importancia de Variables Directa sobre SVM**: Para determinar la relevancia de cada característica socioeconómica y académica, no se emplean árboles ni medidas MDI de impureza Gini. En su lugar, se utiliza **Permutation Feature Importance** evaluada directamente sobre el `Pipeline` de SVM ajustado en el conjunto de prueba no visto.

---

## 2. Origen y Construcción de la Variable Objetivo Multiclase (`estado_final`)

En el dataset sintético (`estudiantes_sinteticos_11500.csv`), la condición final del estudiante se formuló según los criterios del **Sistema de Prevención de la Deserción en Educación Superior (SPADIES 3.0 - MEN Colombia)** mediante 3 categorías mutuamente excluyentes (`Graduado`, `En curso`, `Desertor`):

### A. Índice de Riesgo Ponderado Sintético ($R_i \in [0, 7.5]$):
$$R_i = 2.5 \cdot \mathbb{I}(\text{promedio} < 3.0) + 2.5 \cdot \mathbb{I}(\text{reprobadas} \ge 2) + 1.0 \cdot \mathbb{I}(\text{icfes} < 40) + 1.0 \cdot \mathbb{I}(\text{trabaja} = 1) + 0.5 \cdot \mathbb{I}(\text{beca} = 0) + 0.5 \cdot \mathbb{I}(\text{estrato} \le 2)$$

### B. Distribución de Clases SPADIES:
- **Desertor** ($R_i \ge 3.5$): **3,863 observaciones (33.59%)** — Estudiantes en condición de deserción por riesgo crítico acumulado.
- **Graduado** ($R_i < 3.5 \text{ y } \text{semestre} \ge \text{duración nominal} \text{ y } \text{promedio} \ge 3.2$): **2,500 observaciones (21.74%)** — Estudiantes con titulación satisfactoria.
- **En curso** (Demás casos): **5,137 observaciones (44.67%)** — Estudiantes activos en avance lectivo.

---

## 3. Resumen de Resultados Empíricos Multiclase (50 Trials sin Fuga de Datos)

A continuación se presentan los resultados consolidados tras ejecutar 50 particiones independientes ($N=50$ *trials*, evaluando 400 modelos SVM en total) mediante `Pipeline` + `ColumnTransformer`:

| Método de Normalización | $F_1$-Score Ponderado (Weighted) | $F_1$-Score Macro | $F_1$-Score (Desertor) | Desviación Estándar ($\sigma$) | $p$-value vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard Scaler** | **0.8878** | **0.8872** | **0.9015** | 0.0061 | $< 0.000001$ |
| **Power Transformer** | **0.8877** | **0.8870** | **0.9016** | 0.0058 | $< 0.000001$ |
| **Robust Scaler** | **0.8765** | **0.8759** | **0.8846** | 0.0060 | $< 0.000001$ |
| **Min Max Scaler** | **0.8443** | **0.8431** | **0.8352** | 0.0052 | $< 0.000001$ |
| **Quantile Transformer** | **0.8422** | **0.8429** | **0.8361** | 0.0060 | $< 0.000001$ |
| **Max Absolute Scaler** | **0.8414** | **0.8398** | **0.8314** | 0.0054 | $< 0.000001$ |
| **No Normalization (Baseline)** | **0.4792** | **0.4097** | **0.6258** | 0.0064 | 1.000000 |
| **Ordinal Encoder** | **0.3699** | **0.3512** | **0.0000** | 0.0052 | $< 0.000001$ |

### Principales Hallazgos Empíricos:
1. **Liderazgo de Standard Scaler y Power Transformer**: Ambos métodos alcanzaron el mejor desempeño multiclase global con un **$F_1$-Score Macro de 0.8872** y un **Recall/F1 sobre Desertores superior al 90.1%**.
2. **Robustez de `RobustScaler`**: Obtuvo un **$F_1$ Macro de 0.8759**, siendo altamente recomendado cuando existen observaciones atípicas extremas (como en `distancia_hogar_ies_km`).
3. **Alto Recall en la Clase Crítica (`Desertor`)**: El modelo `SVC(class_weight='balanced')` logra un **Recall del 92.11%** sobre la clase de riesgo (Desertores), minimizando la tasa de falsos negativos.

---

## 4. Diagnóstico EDA e Importancia de Variables en SVM

### A. Diagnóstico Estadístico de Outliers (IQR)
| Variable Continua | Media | Mediana | IQR | Conteo Outliers | % Outliers | Skewness (Sesgo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `distancia_hogar_ies_km` | 11.12 | 7.43 | 9.53 | 824 | **7.17%** | **+5.05** |
| `estrato_socioeconomico` | 2.61 | 2.00 | 1.00 | 959 | **8.34%** | +0.67 |
| `beneficiario_icetex` | 0.22 | 0.00 | 0.00 | 2516 | **21.88%** | +1.36 |
| `beneficiario_beca` | 0.19 | 0.00 | 0.00 | 2147 | **18.67%** | +1.61 |

### B. Importancia de Variables en SVM (Permutation Importance)
1. `semestre_cursado`: **0.2010**
2. `promedio_academico`: **0.1849**
3. `materias_reprobadas`: **0.1817**
4. `nivel_formacion`: **0.0815**
5. `trabaja_mientras_estudia`: **0.0340**
6. `estrato_socioeconomico`: **0.0268**
7. `puntaje_icfes_percentil`: **0.0209**

---

## 5. Visualizaciones Generadas

Todas las figuras están guardadas en formato de alta resolución (300 DPI) en [`Archivos/images`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/images):
1. [`eda_distribucion_correlaciones.png`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/images/eda_distribucion_correlaciones.png): Distribución de 3 clases y panel comparativo Pearson vs Spearman.
2. [`eda_efecto_escaladores.png`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/images/eda_efecto_escaladores.png): Demostración visual del comportamiento de 6 transformadores sobre `distancia_hogar_ies_km`.
3. [`svm_normalizacion_vs_baseline.png`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/images/svm_normalizacion_vs_baseline.png): Diagrama de barras del ranking multiclase de normalizadores vs Baseline (50 trials).
4. [`svm_hiperplano_pca_2d.png`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/images/svm_hiperplano_pca_2d.png): Hiperplano de separación SVM multiclase en 2D (PCA).
5. [`svm_matriz_confusion_roc.png`](file:///c:/Users/Santiago%20T/Documents/Ubiversidad%202026/2026%20-%20II/Machine%20Learning%202/Proyecto/Archivos/images/svm_matriz_confusion_roc.png): Matriz de confusión multiclase $3 \times 3$.
