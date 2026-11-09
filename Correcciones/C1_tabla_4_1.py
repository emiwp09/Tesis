"""Arma la nueva Tabla 4.1 a partir de las métricas por pliegue de A1, A2 y A3 corregidos.

A1 trae dos regresiones logísticas (punto iv): una con la raza y otra con el microbioma.
Todos los modelos se evalúan con la misma etiqueta (was_preterm), la misma unidad
(paciente) y los mismos 25 pliegues por paciente. Se reporta media ± desviación
estándar entre pliegues, y la diferencia pareada de AUC-ROC entre modelos con su
valor p (prueba t pareada corregida de Nadeau y Bengio, 2003, para validación
cruzada repetida: los pliegues comparten pacientes de entrenamiento y no son
independientes). Si existe resultados/aditivo_C2.csv (C2, punto iv), también
compara los modelos con y sin interacciones entre especies.

Uso: python Correcciones/C1_tabla_4_1.py
Salida: resultados/tabla_4_1.csv y resultados/tabla_4_1.tex
"""
import numpy as np
import pandas as pd
from scipy import stats

from comun import RESULTADOS

ORDEN = ['Regresión logística (raza)', 'Regresión logística (microbioma)', 'Random Forest', 'CNN-LSTM']
METRICAS = [('auc_roc', 'AUC-ROC'), ('auc_pr', 'AUC-PR'), ('accuracy', 'Exactitud'),
            ('precision', 'Precisión'), ('recall', 'Recall'), ('f1', 'F1')]

res = pd.concat([pd.read_csv(RESULTADOS / f'metricas_{m}.csv') for m in ('A1', 'A2', 'A3')])
assert res.groupby('modelo').size().eq(25).all(), 'faltan pliegues'

resumen = res.groupby('modelo')[[m for m, _ in METRICAS]].agg(['mean', 'std']).reindex(ORDEN)
resumen.round(3).to_csv(RESULTADOS / 'tabla_4_1.csv')
print(resumen.round(3).to_string())

# Referencia: prevalencia de pretérmino en prueba (AUC-PR de un clasificador al azar)
prev = (res.n_positivas / res.n_pacientes).mean()
print(f'\nPrevalencia media en prueba (AUC-PR del azar): {prev:.3f}')

# Diferencias pareadas de AUC-ROC (mismos pliegues)
def t_corregida(d, prop_prueba=1 / 4):
    """Valor p de la t pareada corregida de Nadeau y Bengio (5 pliegues: prueba/entrenamiento = 1/4)."""
    d = np.asarray(d)
    n = len(d)
    t = d.mean() / np.sqrt((1 / n + prop_prueba) * d.var(ddof=1))
    return 2 * stats.t.sf(abs(t), n - 1)


def comparar(ancho, a, b):
    d = ancho[a] - ancho[b]
    return (f'AUC-ROC {a} - {b}: {d.mean():+.3f} ± {d.std():.3f}; '
            f'{a} mejor en {(d > 0).sum()}/{len(d)} pliegues; p = {t_corregida(d):.3f}')


ancho = res.pivot_table(index=['rep', 'fold'], columns='modelo', values='auc_roc')
pares = [('Random Forest', 'CNN-LSTM'),
         ('Random Forest', 'Regresión logística (microbioma)'),   # punto iv: lineal vs no lineal
         ('Regresión logística (microbioma)', 'Regresión logística (raza)'),
         ('Random Forest', 'Regresión logística (raza)'),
         ('CNN-LSTM', 'Regresión logística (raza)')]

# Punto iv: control sin interacciones (C2)
if (RESULTADOS / 'aditivo_C2.csv').exists():
    aditivo = pd.read_csv(RESULTADOS / 'aditivo_C2.csv')
    ancho = ancho.join(aditivo.pivot_table(index=['rep', 'fold'], columns='modelo', values='auc_roc'))
    print('\nControl de C2 (AUC-ROC por paciente):')
    print(aditivo.groupby('modelo').auc_roc.agg(['mean', 'std']).round(3).to_string())
    pares += [('Boosting profundidad 3 (con interacciones)', 'Boosting profundidad 1 (sin interacciones)'),
              ('Random Forest', 'Boosting profundidad 1 (sin interacciones)'),
              ('Boosting profundidad 1 (sin interacciones)', 'Regresión logística (microbioma)')]

lineas = [comparar(ancho, a, b) for a, b in pares]
print('\n' + '\n'.join(lineas))
(RESULTADOS / 'diferencias_auc.txt').write_text('\n'.join(lineas) + '\n', encoding='utf-8')

# Tabla LaTeX
filas = []
for modelo in ORDEN:
    celdas = [f'{resumen.loc[modelo, (m, "mean")]:.2f} $\\pm$ {resumen.loc[modelo, (m, "std")]:.2f}'
              for m, _ in METRICAS]
    filas.append(f'        {modelo} & ' + ' & '.join(celdas) + r' \\')
tex = '\n'.join([
    r'\begin{tabular}{l' + 'c' * len(METRICAS) + '}',
    r'        \toprule',
    r'        \textbf{Modelo} & ' + ' & '.join(f'\\textbf{{{n}}}' for _, n in METRICAS) + r' \\',
    r'        \midrule',
    *filas,
    r'        \bottomrule',
    r'    \end{tabular}',
])
(RESULTADOS / 'tabla_4_1.tex').write_text(tex + '\n', encoding='utf-8')
print('\n' + tex)
