# Resultados de la Fase 2
*Modo: RÁPIDO (prueba) · semilla 42*

## Datos
Registros: 11500. Objetivo simulado. Proporciones: Graduado 30.7%, En curso 43.5%, Desertor 25.8%.

| variable | nulos | fuera_de_rango |
|---|---|---|
| promedio_academico | 0 | 0 |
| puntaje_icfes_percentil | 0 | 0 |
| estrato_socioeconomico | 0 | 0 |
| edad_ingreso | 0 | 0 |
| materias_reprobadas | 0 | 0 |
| distancia_hogar_ies_km | 0 | 0 |
| semestre_cursado | 0 | 0 |
| trabaja_mientras_estudia | 0 | 0 |
| beneficiario_icetex | 0 | 0 |
| beneficiario_beca | 0 | 0 |

## Reproducción del artículo
**A) SVM polinómica fiel**

| normalización | f1_ponderado | sd | f1_desertor | recall_desertor |
|---|---|---|---|---|
| Sin normalización | 0.647 | 0.024 | 0.558 | 0.633 |
| StandardScaler | 0.706 | 0.024 | 0.508 | 0.424 |
| MinMaxScaler | 0.677 | 0.011 | 0.454 | 0.405 |
| MaxAbsScaler | 0.681 | 0.012 | 0.465 | 0.413 |
| RobustScaler | 0.712 | 0.030 | 0.528 | 0.462 |
| Quantile (uniforme) | 0.689 | 0.021 | 0.491 | 0.470 |
| Quantile (normal) | 0.601 | 0.020 | 0.403 | 0.299 |
| PowerTransformer | 0.709 | 0.023 | 0.528 | 0.447 |
| Normalizer (L2) | 0.543 | 0.023 | 0.360 | 0.269 |

**B) SVM polinómica con class_weight**

| normalización | f1_ponderado | sd | f1_desertor | recall_desertor |
|---|---|---|---|---|
| Sin normalización | 0.598 | 0.011 | 0.545 | 0.720 |
| StandardScaler | 0.695 | 0.010 | 0.532 | 0.527 |
| MinMaxScaler | 0.670 | 0.009 | 0.484 | 0.504 |
| MaxAbsScaler | 0.667 | 0.005 | 0.488 | 0.515 |
| RobustScaler | 0.693 | 0.012 | 0.532 | 0.572 |
| Quantile (uniforme) | 0.671 | 0.010 | 0.516 | 0.583 |
| Quantile (normal) | 0.592 | 0.025 | 0.507 | 0.716 |
| PowerTransformer | 0.692 | 0.018 | 0.535 | 0.527 |
| Normalizer (L2) | 0.528 | 0.012 | 0.396 | 0.379 |

## Comparación de modelos (validación anidada, media ± sd)
| index | f1_weighted | f1_macro | f1_desertor | recall_desertor | balanced_accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.721 ± 0.014 | 0.699 ± 0.016 | 0.527 ± 0.030 | 0.469 ± 0.057 | 0.699 ± 0.016 |
| Gradient Boosting | 0.720 ± 0.012 | 0.700 ± 0.011 | 0.537 ± 0.011 | 0.528 ± 0.020 | 0.702 ± 0.011 |
| XGBoost | 0.713 ± 0.012 | 0.689 ± 0.013 | 0.495 ± 0.019 | 0.446 ± 0.027 | 0.690 ± 0.013 |
| SVM lineal | 0.712 ± 0.014 | 0.691 ± 0.016 | 0.518 ± 0.028 | 0.508 ± 0.037 | 0.693 ± 0.016 |
| LogReg ajustada | 0.706 ± 0.001 | 0.691 ± 0.003 | 0.564 ± 0.019 | 0.606 ± 0.044 | 0.696 ± 0.003 |
| SVM polinómica | 0.703 ± 0.004 | 0.690 ± 0.005 | 0.580 ± 0.032 | 0.594 ± 0.063 | 0.695 ± 0.006 |
| Baseline_LogReg | 0.703 ± 0.016 | 0.680 ± 0.017 | 0.496 ± 0.009 | 0.472 ± 0.015 | 0.680 ± 0.018 |
| SVM RBF | 0.686 ± 0.005 | 0.676 ± 0.009 | 0.582 ± 0.029 | 0.660 ± 0.053 | 0.685 ± 0.010 |
| Baseline_Mayoria | 0.263 ± 0.001 | 0.202 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.333 ± 0.000 |

## Pruebas contra la línea base
| index | metrica | modelo | mejora_media | p_bruto | p_bonferroni |
|---|---|---|---|---|---|
| 0 | f1_weighted | LogReg ajustada | 0.003 | 1.000 | 1.000 |
| 1 | f1_weighted | SVM lineal | 0.009 | 0.500 | 1.000 |
| 2 | f1_weighted | SVM polinómica | 0.000 | 1.000 | 1.000 |
| 3 | f1_weighted | SVM RBF | -0.017 | 0.250 | 1.000 |
| 4 | f1_weighted | Random Forest | 0.018 | 0.250 | 1.000 |
| 5 | f1_weighted | Gradient Boosting | 0.017 | 0.500 | 1.000 |
| 6 | f1_weighted | XGBoost | 0.010 | 0.250 | 1.000 |
| 7 | f1_desertor | LogReg ajustada | 0.068 | 0.250 | 1.000 |
| 8 | f1_desertor | SVM lineal | 0.022 | 0.500 | 1.000 |
| 9 | f1_desertor | SVM polinómica | 0.084 | 0.250 | 1.000 |
| 10 | f1_desertor | SVM RBF | 0.086 | 0.250 | 1.000 |
| 11 | f1_desertor | Random Forest | 0.030 | 0.250 | 1.000 |
| 12 | f1_desertor | Gradient Boosting | 0.040 | 0.250 | 1.000 |
| 13 | f1_desertor | XGBoost | -0.001 | 0.750 | 1.000 |

## Lectura
- Normalización: con la SVM polinómica, la mejor fue StandardScaler (F1 ponderado 0.695) y la peor Normalizer (L2) (0.528); sin normalizar dio 0.598. Friedman p = 0.00553.
- Mejor modelo: Random Forest, con F1 ponderado 0.721 ± 0.014 en validación anidada y 0.724 en el conjunto de prueba (F1 Desertor 0.516, recall Desertor 0.429).
- Frente a la regresión logística básica, la diferencia media en F1 ponderado es +0.018 (NO estadísticamente significativa; p con Bonferroni = 1).
- Variables más importantes: promedio_academico, puntaje_icfes_percentil, materias_reprobadas, estrato_socioeconomico, nivel_educativo_madre_num.
- Errores: se perdieron 44 desertores del conjunto de prueba; hay que revisar su perfil y las brechas por subgrupo.
- Limitaciones: la etiqueta es simulada (los resultados prueban la metodología, no la predicción real); esta corrida es RÁPIDA y no es reportable; verifique la lista de 9 normalizaciones contra el artículo; la base tiene menos predictores que los 36 previstos.