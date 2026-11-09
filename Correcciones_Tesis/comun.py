"""Utilidades comunes a los tres modelos corregidos (corrección i).

- Los tres modelos usan la misma etiqueta: was_preterm (parto < 37 semanas).
- Los tres se evalúan sobre los mismos pliegues, definidos por paciente
  (pliegues_por_paciente.csv, generado por C0_pliegues_por_paciente.py).
- La unidad de evaluación es la paciente. Los modelos que predicen por muestra
  (regresión logística y Random Forest) promedian la probabilidad de las
  muestras de cada paciente.
- Umbral de decisión (corrección iii): índice de Youden calculado sin usar el
  pliegue de prueba (umbral_interno para A1 y A2; validación para A3).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
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


def umbral_youden(y, prob):
    """Umbral de Youden: el punto de la curva ROC que maximiza TPR - FPR.

    Se calcula SOLO con datos que no son de prueba (validación o predicciones
    fuera de pliegue dentro del entrenamiento). El umbral obtenido se congela y
    se aplica una sola vez al pliegue de prueba.
    """
    fpr, tpr, thr = roc_curve(y, prob)
    thr = np.minimum(thr, 1.0)          # roc_curve pone +inf en el primer punto
    return float(thr[np.argmax(tpr - fpr)])


def umbral_interno(ajustar_predecir, X, y, pacientes, semilla, n_pliegues=3):
    """Umbral de Youden con una validación cruzada interna por paciente.

    Para los modelos que predicen por muestra (regresión logística y Random
    Forest). Dentro de las pacientes de ENTRENAMIENTO de un pliegue externo se
    hace una validación cruzada de n_pliegues, agrupada por paciente: cada
    paciente de entrenamiento recibe una probabilidad de un modelo que no la vio.
    Las probabilidades se promedian por paciente y con ellas se calcula el umbral
    de Youden. El pliegue de prueba no interviene.

    ajustar_predecir(X_tr, y_tr, X_va) -> probabilidades de la clase 1 para X_va.
    """
    interna = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    prob = np.full(len(y), np.nan)
    for i_tr, i_va in interna.split(X, y, groups=pacientes):
        prob[i_va] = ajustar_predecir(X[i_tr], y[i_tr], X[i_va])
    assert not np.isnan(prob).any()
    pp = promedio_por_paciente(pacientes, y, prob)
    return umbral_youden(pp.y, pp.prob)


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
