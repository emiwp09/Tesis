"""Crea, una sola vez, los pliegues de evaluación comunes a los tres modelos.

Cada paciente (participant_id) cae en un solo pliegue por repetición, de modo
que ninguna paciente aparece a la vez en entrenamiento y prueba. Los pliegues
se estratifican por la etiqueta was_preterm. 5 pliegues x 5 repeticiones.

Uso: python Correcciones/C0_pliegues_por_paciente.py
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from comun import DATOS, ARCHIVO_PLIEGUES, ETIQUETA

N_PLIEGUES, N_REPETICIONES = 5, 5

metadata = pd.read_csv(DATOS / 'metadata.csv')
# La etiqueta es constante dentro de cada paciente; se verifica.
assert (metadata.groupby('participant_id')[ETIQUETA].nunique() == 1).all()
pacientes = (metadata.groupby('participant_id')[ETIQUETA].first()
             .astype(int).reset_index().sort_values('participant_id'))

filas = []
for rep in range(N_REPETICIONES):
    skf = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=rep)
    for fold, (_, prueba) in enumerate(skf.split(pacientes, pacientes[ETIQUETA])):
        bloque = pacientes.iloc[prueba].copy()
        bloque['rep'], bloque['fold'] = rep, fold
        filas.append(bloque)

pliegues = pd.concat(filas, ignore_index=True)
pliegues.to_csv(ARCHIVO_PLIEGUES, index=False)

print(f'{len(pacientes)} pacientes, {pacientes[ETIQUETA].sum()} pretérmino '
      f'({pacientes[ETIQUETA].mean():.1%})')
print(pliegues.groupby(['rep', 'fold'])[ETIQUETA].agg(['size', 'sum']).unstack('fold'))
