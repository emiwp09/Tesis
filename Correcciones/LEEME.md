# Correcciones: puntos i (modelos comparables), ii (Random Forest) y iii (umbral fuera de prueba)

Los códigos buscan los datos primero **en esta misma carpeta** y, si no están, en `../CSVTESIS/`.
Basta con poner aquí estos 4 CSV (sirven igual en .csv o en el .zip del repositorio):
`metadata.csv`, `cst_valencia_w_taxons.csv`, `taxonomy_relabd.species.csv`, `phylotype_nreads.1e_1.csv`.

Orden para correr (abriendo Jupyter o la terminal **dentro de `Correcciones/`**):

1. `python C0_pliegues_por_paciente.py` → crea `pliegues_por_paciente.csv` (ya viene incluido; correrlo de nuevo da el mismo archivo).
2. `A1_RegresionLogistica_corregido.ipynb`
3. `A2_RandomForest_corregido.ipynb` (al final, punto v: Random Forest con la abundancia media por paciente → `metricas_A2_media.csv`, `pred_A2_media.csv`)
4. `C2_RF_robustez.ipynb` (necesita los resultados de A2: matriz de confusión del RF y chequeo de regularización)
5. `A3_Red_Hibrida_CNN-LSTM_corregido.ipynb` (4 variantes x 25 pliegues = 100 entrenamientos de hasta 500 épocas con
   EarlyStopping; en CPU, unas 4 veces lo que tarda el modelo completo solo (~75 min). Para Google Colab, ver la primera celda).
   Al final, punto v: sección de análisis que no entrena (IC del AUC, comparaciones pareadas, control de tiempo de
   muestreo y subcohorte con >= 3 muestras) → `resultados/v_*.csv`, `AUC_pliegues_v.png` y `CurvaROC_A3_paciente.png`.
   Necesita que A2 ya se haya corrido (`pred_A2.csv` y `pred_A2_media.csv`).
6. `python C1_tabla_4_1.py` → `resultados/tabla_4_1.csv`, `tabla_4_1.tex` y `diferencias_auc.txt`.

Cada notebook guarda en `resultados/` sus métricas por pliegue (`metricas_A*.csv`) y las probabilidades por paciente (`pred_A*.csv`).
Versiones usadas: pandas 2.3.3, scikit-learn 1.7.2, TensorFlow 2.20 / Keras 3 (las de requirements.txt).

Umbral de decisión (punto iii): índice de Youden, max(TPR − FPR), calculado sin usar el pliegue de prueba.
A1 y A2 lo calculan con una validación cruzada interna de 3 pliegues por paciente dentro del entrenamiento
(`comun.umbral_interno`); A3 lo calcula con su conjunto de validación (`comun.umbral_youden`).
A3 guarda además `pred_val_A3.csv`, `matriz_confusion_A3.csv`, `MatrixConfusion_A3_paciente.png`,
`Umbral_validacion_A3.png` y `LossEpocas_A3.png`.
