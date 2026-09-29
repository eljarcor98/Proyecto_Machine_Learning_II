# Adaptación de Máquinas de Soporte Vectorial (SVM) para la Predicción de Deserción Estudiantil bajo Diferentes Métodos de Normalización

**Asignatura:** Machine Learning II  
**Marco de Referencia:** *Support Vector Machine for Predicting Student Dropout Under Different Normalization Methods* (Boteju, Tang & Brown, **IEEE BigData 2024**)  
**Contexto de Aplicación:** Educación Superior en Colombia — Sistema de Prevención de la Deserción (**SPADIES 3.0**, Ministerio de Educación Nacional de Colombia)  
**Dataset:** 11,500 estudiantes (dimensiones académicas, socioeconómicas e institucionales)  

---

## Resumen Ejecutivo

Este proyecto replica y adapta la metodología científica del artículo internacional de **IEEE BigData 2024** (*Boteju et al.*) para evaluar el impacto de **9 técnicas de normalización y codificación de datos** sobre el rendimiento predictivo de una **Máquina de Soporte Vectorial (SVM)** con **Kernel Polinómico de grado 3 ($d=3$)**, utilizando una cohorte colombiana de **11,500 estudiantes**.

A través de un protocolo experimental riguroso de **50 repeticiones Monte Carlo independientes** con particiones $70\%$ entrenamiento / $30\%$ prueba, se evaluó la estabilidad, capacidad de generalización y significancia estadística (mediante la **Prueba T de Student de dos muestras**) de cada método frente a un **Baseline de control sin normalizar**.

### Hallazgos Principales:
1. **Superioridad del Escalado Acotado y Robusto:** Las técnicas **Ordinal Encoder + MinMax** ($F1 = 0.9100 \pm 0.0045$), **Scale / Robust Scaler** ($F1 = 0.9093 \pm 0.0047$) y **Quantile Transformer** ($F1 = 0.8983 \pm 0.0046$) alcanzaron el rendimiento más alto, con mejoras estadísticamente significativas sobre el modelo sin normalizar ($p < 10^{-20}$).
2. **Capacidad Discriminante Excepcional:** El mejor modelo SVM alcanzó un **ROC-AUC de 0.9746 (97.46%)**, una precisión del **91.93%** en la detección de desertores y una tasa de acierto global (*Accuracy*) del **90.81%**.
3. **Sensibilidad Matemática del Kernel Polinómico:** A diferencia de algoritmos basados en árboles, el SVM es altamente vulnerable a la asimetría extrema ($Skew > 5.0$) y valores atípicos severos en variables como `distancia_hogar_ies_km`, demostrando la necesidad imperativa de preprocesamiento acotado o robusto.

---

## 1. Fundamentación Teórica y Formulación Matemática

### 1.1 Máquinas de Soporte Vectorial (SVM) con Kernel Polinómico
Las Máquinas de Soporte Vectorial buscan encontrar el hiperplano óptimo de separación $w^T \phi(x) + b = 0$ que maximiza el margen geométrico $\frac{2}{\|w\|}$ entre clases mientras penaliza los errores de clasificación mediante variables de holgura $\xi_i$:

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{sujeto a: } y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

Para capturar interacciones no lineales complejas entre factores socioeconómicos y académicos, se implementa el **Kernel Polinómico de grado $d=3$** (siguiendo estrictamente el paper base):

$$K(x_i, x_j) = (\gamma \langle x_i, x_j \rangle + r)^d$$

### 1.2 Justificación de la Sensibilidad a la Escala
El cálculo del producto interno $\langle x_i, x_j \rangle = \sum_{k=1}^{p} x_{ik} x_{jk}$ depende directamente de la magnitud absoluta de cada característica $k$. Cuando variables continuas difieren en órdenes de magnitud (por ejemplo, distancias de hasta $250\text{ km}$ frente a promedios acumulados de $0.0 \text{ a } 5.0$), los términos de mayor escala **dominan exponencialmente la función de decisión polinomial**, anulando la influencia de factores académicos críticos y ralentizando la convergencia del optimizador cuadrático.

---

## 2. Diagnóstico y Análisis Exploratorio de Datos (EDA)

El conjunto de datos comprende **11,500 registros** y 21 atributos estructurados bajo el marco conceptual del **SPADIES 3.0**.

### 2.1 Diagnóstico Estadístico de Escalas, Asimetría y Valores Atípicos

| Variable | Media | Desv. Est. | Mínimo | Máximo | Rango | Asimetría (*Skew*) | Curtosis | Outliers (IQR) | % Outliers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`edad_ingreso`** | 19.57 | 2.96 | 15.00 | 32.00 | 17.00 | +0.316 | -0.406 | 3 | 0.03% |
| **`promedio_academico`** | 3.41 | 0.63 | 0.87 | 5.00 | 4.13 | -0.055 | -0.113 | 43 | 0.37% |
| **`materias_reprobadas`** | 1.09 | 1.20 | 0.00 | 12.00 | 12.00 | **+1.240** | 1.920 | 39 | 0.34% |
| **`distancia_hogar_ies_km`** | 11.12 | 12.52 | 0.22 | 250.00 | **249.78** | **+5.045** | **54.172** | **824** | **7.17%** |
| **`puntaje_icfes_percentil`** | 57.96 | 17.80 | 1.00 | 100.00 | 99.00 | -0.034 | -0.191 | 36 | 0.31% |
| **`semestre_cursado`** | 6.53 | 3.44 | 1.00 | 12.00 | 11.00 | -0.015 | -1.206 | 0 | 0.00% |
| **`estrato_socioeconomico`** | 2.61 | 1.20 | 1.00 | 6.00 | 5.00 | +0.669 | 0.039 | 959 | 8.34% |
| **`anio_registro`** | 2.31 | 1.27 | 0.00 | 4.00 | 4.00 | -0.260 | -0.987 | 0 | 0.00% |
| **`trabaja_mientras_estudia`** | 0.38 | 0.48 | 0.00 | 1.00 | 1.00 | +0.510 | -1.740 | 0 | 0.00% |
| **`beneficiario_icetex`** | 0.22 | 0.41 | 0.00 | 1.00 | 1.00 | +1.360 | -0.149 | 2516 | 21.88% |
| **`beneficiario_beca`** | 0.19 | 0.39 | 0.00 | 1.00 | 1.00 | +1.608 | 0.586 | 2147 | 18.67% |

> **Conclusión del Diagnóstico:** `distancia_hogar_ies_km` presenta un coeficiente de asimetría extremo ($Skew = 5.045$) y una curtosis de $54.172$ con $824$ valores atípicos. Esto justifica teóricamente la superioridad de transformaciones como **RobustScaler** (inmune a outliers por rango intercuartil) y **QuantileTransformer**.

---

### 2.2 Análisis de Multicolinealidad (Correlación de Spearman)
* `materias_reprobadas` tiene una fuerte correlación positiva con la deserción ($r_s \approx +0.62$).
* `promedio_academico` tiene una fuerte correlación inversa ($r_s \approx -0.65$).
* No se observan correlaciones colineales excesivas ($|r_s| > 0.85$) entre predictores, lo que descarta redundancia directa que requiera eliminación forzada previa.

---

### 2.3 Cruces Socioeconómicos del Contexto Colombiano (SPADIES)
* **Estrato 1 y 2:** Concentran una tasa de deserción superior al **45.2%**, mientras que en estratos 5 y 6 la tasa desciende al **14.8%**.
* **Condición Laboral:** Los estudiantes que trabajan mientras estudian duplican la tasa de deserción frente a quienes se dedican exclusivamente a su carrera (**48.1% vs 24.3%**).
* **Apoyo Financiero:** Estudiantes sin beca ni crédito ICETEX presentan la mayor vulnerabilidad (**44.5% de deserción**), mientras que beneficiarios de Beca + ICETEX reducen su riesgo al **11.2%**.

---

### 2.4 Relevancia No Lineal de Variables (Información Mutua y Random Forest)

| Variable / Característica | Información Mutua ($\text{MI}$) | Importancia Random Forest (Gini) | Rol en el Modelo |
| :--- | :---: | :---: | :--- |
| **`promedio_academico`** | **0.2915** | **0.4032** | Determinante primario temprano |
| **`materias_reprobadas`** | **0.2488** | **0.3663** | Alerta académica crítica |
| **`puntaje_icfes_percentil`** | **0.1308** | **0.1025** | Competencia básica de ingreso |
| **`trabaja_mientras_estudia`** | **0.0382** | **0.0245** | Carga de tiempo extracurricular |
| **`estrato_socioeconomico`** | **0.0274** | **0.0198** | Factor de vulnerabilidad económica |
| *Variables Categóricas (OHE)* | $0.0000 - 0.0096$ | $< 0.0015$ | Ajustes locales contextuales |

---

## 3. Catálogo de Técnicas de Normalización Evaluadas

Siguiendo la Sección III y la *Tabla I* del paper base (*Boteju et al., IEEE 2024*):

1. **Standard Scaler:** Estandarización a media cero y varianza unitaria: $z = \frac{x - \mu}{\sigma}$.
2. **Min Max Scaler:** Transformación lineal al intervalo $[0, 1]$: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$.
3. **Max Absolute Scaler:** Escalado preservando el signo y los ceros en $[-1, 1]$: $x_{scaled} = \frac{x}{\max(|x|)}$.
4. **Scale (Robust Scaler):** Centrado en la mediana y escalado por rango intercuartil: $x_{scaled} = \frac{x - Q_2}{Q_3 - Q_1}$.
5. **Quantile Transformer:** Mapeo cuantil no paramétrico a una distribución uniforme acotada en $[0, 1]$.
6. **Power Transformer (Yeo-Johnson):** Estabilización de varianza y aproximación a normalidad mediante $\psi(\lambda, x)$.
7. **One Hot Scaler:** Codificación dummy de variables categóricas acoplada con escalado MinMax de variables continuas.
8. **Ordinal Encoder:** Codificación entera ordinal para todas las columnas categóricas, escalada uniformemente.
9. **Label Encoder:** Mapeo secuencial entero directo.
10. **No Normalization (Control Baseline):** Datos en su escala original sin preprocesamiento.

---

## 4. Resultados Experimentales (50 Repeticiones Monte Carlo)

La siguiente tabla resume los resultados consolidados de las **50 ejecuciones independientes** ($N = 50$, $70\%$ train / $30\%$ test):

| Posición | Método de Normalización | F1-Score Promedio | Desviación Estándar | Ganancia vs Baseline ($\Delta F1$) | Estadístico $t$ (T-Test) | $p$-valor | Significancia ($p < 0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Ordinal Encoder** | **0.9100** | $\pm 0.0045$ | **+0.0523** | **27.314** | **$0.0000$** | **Altamente Significativa** |
| 🥈 **2** | **Scale (Robust Scaler)** | **0.9093** | $\pm 0.0047$ | **+0.0516** | **26.802** | **$0.0000$** | **Altamente Significativa** |
| 🥉 **3** | **Quantile Transformer** | **0.8983** | $\pm 0.0046$ | **+0.0406** | **21.136** | **$0.0000$** | **Altamente Significativa** |
| **4** | **One Hot Scaler** | **0.8915** | $\pm 0.0055$ | **+0.0338** | **17.161** | **$0.0000$** | **Altamente Significativa** |
| **5** | **Min Max Scaler** | **0.8915** | $\pm 0.0055$ | **+0.0338** | **17.161** | **$0.0000$** | **Altamente Significativa** |
| **6** | **Max Absolute Scaler** | **0.8903** | $\pm 0.0050$ | **+0.0326** | **16.833** | **$0.0000$** | **Altamente Significativa** |
| **7** | **Label Encoder** | **0.8755** | $\pm 0.0050$ | **+0.0178** | **9.200** | **$0.0000$** | **Altamente Significativa** |
| **8** | **No Normalization (Baseline)** | **0.8577** | $\pm 0.0126$ | $0.0000$ | $0.000$ | $1.0000$ | Baseline de Control |
| **9** | **Standard Scaler** | **0.8556** | $\pm 0.0065$ | $-0.0021$ | $-1.015$ | $0.3135$ | No significativa |
| **10** | **Power Transformer** | **0.8511** | $\pm 0.0056$ | $-0.0066$ | $-3.335$ | $0.0014$ | Degradación |

---

## 5. Evaluación Detallada del Mejor Modelo (*Robust Scaler / Ordinal*)

Evaluando el modelo óptimo en una partición de prueba independiente de **3,450 estudiantes**:

### 5.1 Matriz de Confusión

| | Predicción: No Desertor (0) | Predicción: Desertor (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Real: No Desertor (0)** | **2,210** (Verdaderos Negativos) | **81** (Falsos Positivos) | 2,291 |
| **Real: Desertor (1)** | **236** (Falsos Negativos) | **923** (Verdaderos Positivos) | 1,159 |
| **Total Predicho** | 2,446 | 1,004 | **3,450** |

### 5.2 Reporte de Clasificación Detallado

| Clase | Precisión (*Precision*) | Sensibilidad (*Recall*) | F1-Score | Soporte (*Support*) |
| :--- | :---: | :---: | :---: | :---: |
| **No Desertor (0)** | **90.35%** | **96.46%** | **0.9331** | 2,291 |
| **Desertor (1)** | **91.93%** | **79.64%** | **0.8534** | 1,159 |
| **Exactitud Global (*Accuracy*)** | — | — | **90.81%** | 3,450 |
| **Promedio Ponderado (*Weighted Avg*)** | **90.88%** | **90.81%** | **0.9063** | 3,450 |

* **Área Bajo la Curva ROC (ROC-AUC):** **$0.9746$ ($97.46\%$)**, demostrando una capacidad de discriminación probabilística sobresaliente frente al azar ($0.50$).
* **Vectores de Soporte:** $2,769$ de $8,050$ muestras de entrenamiento ($34.4\%$).

---

## 6. Interpretabilidad del Hiperplano SVM

### 6.1 Permutation Feature Importance (Post-Hoc SVM)
Al medir la caída en F1-Score cuando se permutan los valores de cada columna en el conjunto de prueba:
1. **`promedio_academico`:** Provoca una caída de **$\Delta F1 = -0.184$** (es el vector director principal del hiperplano).
2. **`materias_reprobadas`:** Provoca una caída de **$\Delta F1 = -0.142$**.
3. **`puntaje_icfes_percentil`:** Caída de **$\Delta F1 = -0.045$**.
4. **`trabaja_mientras_estudia`:** Caída de **$\Delta F1 = -0.018$**.
5. **`estrato_socioeconomico`:** Caída de **$\Delta F1 = -0.012$**.

### 6.2 Visualización del Espacio Latente 2D (PCA)
La proyección bidimensional de los vectores de soporte sobre los dos primeros componentes principales demuestra que el SVM con kernel polinómico genera una frontera de decisión suave y bien orientada, logrando aislar a los estudiantes en riesgo sin sobreajustar en regiones dispersas.

---

## 7. Comparación con los Hallazgos del Paper Original (*IEEE BigData 2024*)

| Criterio | Paper Original (*Boteju et al., IEEE 2024*) | Nuestra Adaptación Colombiana (SPADIES 3.0) |
| :--- | :--- | :--- |
| **Mejor Técnica** | One Hot Scaler ($F1 = 0.7785$) | Ordinal Encoder / Robust Scaler ($F1 = 0.9100$) |
| **Ganancia sobre Baseline** | $\Delta F1 \approx +0.278$ (el baseline sin normalizar colapsaba a $0.500$) | $\Delta F1 \approx +0.052$ (el baseline sin normalizar alcanzó $0.8577$ con alta varianza) |
| **Técnicas de Rango Medio** | MinMax ($0.764$), MaxAbs ($0.764$), Quantile ($0.754$) | Quantile ($0.898$), OneHot ($0.891$), MinMax ($0.891$) |
| **Consistencia Metodológica** | 50 trials Monte Carlo + T-Test | 50 trials Monte Carlo + T-Test |
| **Conclusión Compartida** | **El preprocesamiento de normalización es el factor determinante primario para la convergencia y precisión del SVM.** | **Confirmado:** Las técnicas acotadas y robustas a outliers maximizan el margen de separación del hiperplano polinomial. |

---

## 8. Conclusiones y Recomendaciones de Aplicación

1. **Relevancia del Preprocesamiento Robusto:** En presencia de distribuciones con colas largas o asimetría severa ($Skew > 5.0$), métodos como **RobustScaler** o transformaciones cuantiles deben ser la elección estándar para pipelines de clasificación con SVM.
2. **Impacto en Sistemas de Alerta Temprana (SPADIES):** El modelo entrenado ofrece una **precisión del 91.93%** al emitir alertas de deserción, lo que minimiza falsas alarmas y permite a las universidades colombianas focalizar programas de tutoría y apoyos económicos de forma costo-efectiva.
3. **Reproducibilidad:** Todos los experimentos fueron implementados en Python 3.11 con semillas aleatorias fijas, asegurando total reproducibilidad de los resultados y gráficos presentados.
