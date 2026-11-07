# Correcciones: punto i (modelos comparables)

Esta carpeta va **dentro de la carpeta del repositorio**, junto a `CSVTESIS/`
(los códigos leen los datos de `../CSVTESIS/`, incluidos los .zip de taxonomía y filotipos).

Orden para correr (abriendo Jupyter o la terminal **dentro de `Correcciones/`**):

1. `python C0_pliegues_por_paciente.py` → crea `pliegues_por_paciente.csv` (ya viene incluido; correrlo de nuevo da el mismo archivo).
2. `A1_RegresionLogistica_corregido.ipynb`
3. `A2_RandomForest_corregido.ipynb`
4. `A3_Red_Hibrida_CNN-LSTM_corregido.ipynb` (25 entrenamientos de 500 épocas; en CPU ~75 min)
5. `python C1_tabla_4_1.py` → `resultados/tabla_4_1.csv`, `tabla_4_1.tex` y `diferencias_auc.txt`.

Cada notebook guarda en `resultados/` sus métricas por pliegue (`metricas_A*.csv`) y las probabilidades por paciente (`pred_A*.csv`).
Versiones usadas: pandas 2.3.3, scikit-learn 1.7.2, TensorFlow 2.20 / Keras 3 (las de requirements.txt).
