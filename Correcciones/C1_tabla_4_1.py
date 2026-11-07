"""Arma la nueva Tabla 4.1 a partir de las métricas por pliegue de A1, A2 y A3 corregidos.

Los tres modelos se evalúan con la misma etiqueta (was_preterm), la misma unidad
(paciente) y los mismos 25 pliegues por paciente. Se reporta media ± desviación
estándar entre pliegues, y la diferencia pareada de AUC-ROC entre modelos.

Uso: python Correcciones/C1_tabla_4_1.py
Salida: resultados/tabla_4_1.csv y resultados/tabla_4_1.tex
"""
import pandas as pd

from comun import RESULTADOS

ORDEN = ['Regresión logística', 'Random Forest', 'CNN-LSTM']
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
ancho = res.pivot_table(index=['rep', 'fold'], columns='modelo', values='auc_roc')
lineas = []
for a, b in [('Random Forest', 'CNN-LSTM'), ('Random Forest', 'Regresión logística'),
             ('CNN-LSTM', 'Regresión logística')]:
    d = ancho[a] - ancho[b]
    linea = (f'AUC-ROC {a} - {b}: {d.mean():+.3f} ± {d.std():.3f}; '
             f'{a} mejor en {(d > 0).sum()}/{len(d)} pliegues')
    lineas.append(linea)
    print(linea)
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
