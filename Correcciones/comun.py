"""Utilidades comunes a los tres modelos corregidos (corrección i).

- Los tres modelos usan la misma etiqueta: was_preterm (parto < 37 semanas).
- Los tres se evalúan sobre los mismos pliegues, definidos por paciente
  (pliegues_por_paciente.csv, generado por C0_pliegues_por_paciente.py).
- La unidad de evaluación es la paciente. Los modelos que predicen por muestra
  (regresión logística y Random Forest) promedian la probabilidad de las
  muestras de cada paciente.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)

CORRECCIONES = Path(__file__).resolve().parent
# Los datos (CSV) se buscan primero en la misma carpeta que este archivo y, si no
# están ahí, en ../CSVTESIS (estructura del repositorio).
DATOS = CORRECCIONES
if not (DATOS / 'metadata.csv').exists():
    DATOS = CORRECCIONES.parent / 'CSVTESIS'
RESULTADOS = CORRECCIONES / 'resultados'
ARCHIVO_PLIEGUES = CORRECCIONES / 'pliegues_por_paciente.csv'

ETIQUETA = 'was_preterm'


def leer_datos(nombre):
    """Lee nombre.csv de la carpeta de datos; si no existe, nombre.zip (como en el repositorio)."""
    for ext in ('.csv', '.zip'):
        ruta = DATOS / (nombre + ext)
        if ruta.exists():
            return pd.read_csv(ruta)
    raise FileNotFoundError(f'No encuentro {nombre}.csv ni {nombre}.zip en {DATOS}')


def cargar_pliegues():
    """Tabla participant_id, was_preterm, rep, fold (una fila por paciente y repetición)."""
    return pd.read_csv(ARCHIVO_PLIEGUES)


def iterar_pliegues(pliegues):
    """Devuelve (rep, fold, pacientes_entrenamiento, pacientes_prueba) para cada pliegue."""
    for (rep, fold), _ in pliegues.groupby(['rep', 'fold']):
        rep_df = pliegues[pliegues.rep == rep]
        prueba = set(rep_df.loc[rep_df.fold == fold, 'participant_id'])
        entrenamiento = set(rep_df.loc[rep_df.fold != fold, 'participant_id'])
        assert not prueba & entrenamiento
        yield rep, fold, entrenamiento, prueba


def promedio_por_paciente(participant_id, y, prob):
    """Agrega predicciones por muestra a una probabilidad por paciente (media)."""
    df = pd.DataFrame({'participant_id': participant_id, 'y': y, 'prob': prob})
    assert (df.groupby('participant_id').y.nunique() == 1).all()
    return df.groupby('participant_id').agg(y=('y', 'first'), prob=('prob', 'mean')).reset_index()


def umbral_sens_igual_espec(y, prob):
    """Regla de umbral de la versión original (Listado A.3): punto de la curva ROC
    donde la sensibilidad es igual a la especificidad, TPR - (1 - FPR) ~ 0.

    Se aplica igual a los tres modelos para que las métricas que dependen del
    umbral sean comparables. NOTA: se calcula sobre el pliegue de prueba, como en
    la versión original; esto se corrige en el punto iii.
    """
    fpr, tpr, thr = roc_curve(y, prob)
    return thr[np.argmin(np.abs(tpr - (1 - fpr)))]


def metricas(y, prob, umbral):
    yp = (prob >= umbral).astype(int)
    return dict(auc_roc=roc_auc_score(y, prob),
                auc_pr=average_precision_score(y, prob),
                accuracy=accuracy_score(y, yp),
                precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp, zero_division=0),
                f1=f1_score(y, yp, zero_division=0),
                umbral=umbral,
                n_pacientes=len(y),
                n_positivas=int(np.sum(y)))
