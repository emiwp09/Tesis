"""Re-análisis de la comparación de modelos (correcciones i, ii, iii y iv).

Los modelos usan la misma etiqueta (was_preterm), la misma unidad (muestra) y
los mismos pliegues agrupados por paciente (StratifiedGroupKFold), de modo que
ninguna paciente aparece a la vez en entrenamiento y prueba. El umbral de
decisión se elige con validación cruzada interna sobre el entrenamiento
(índice de Youden), nunca sobre prueba.

Uso: python B1_comparacion_por_paciente.py <carpeta con metadata.csv y taxonomy_relabd.species.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)

N_SPLITS, N_REPEATS = 5, 5
data_dir = sys.argv[1]
meta = pd.read_csv(f'{data_dir}/metadata.csv')
taxa = pd.read_csv(f'{data_dir}/taxonomy_relabd.species.csv')
full = pd.merge(meta[['specimen', 'participant_id', 'was_preterm', 'NIH Racial Category']], taxa, on='specimen')

y = full['was_preterm'].astype(int).values
groups = full['participant_id'].values
X_taxa = full.drop(columns=['specimen', 'participant_id', 'was_preterm', 'NIH Racial Category']).values
X_race = full[['NIH Racial Category']].fillna('Unknown').values


def log_rel(X):
    return np.log10(X + 1e-5)


models = {
    'LR raza (one-hot)': (X_race, lambda: make_pipeline(
        OneHotEncoder(handle_unknown='ignore'),
        LogisticRegression(class_weight='balanced', max_iter=1000))),
    'LR microbioma (log, L2)': (X_taxa, lambda: make_pipeline(
        FunctionTransformer(log_rel), StandardScaler(),
        LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000))),
    'Random Forest microbioma': (X_taxa, lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)),
}


def youden_threshold(make, X, y, g, seed):
    """Umbral de Youden con predicciones fuera de pliegue del conjunto de entrenamiento."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g):
        oof[va] = make().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    fpr, tpr, thr = roc_curve(y, oof)
    return thr[np.argmax(tpr - fpr)]


rows = []
for rep in range(N_REPEATS):
    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=rep)
    for fold, (tr, te) in enumerate(cv.split(X_taxa, y, groups)):
        assert not set(groups[tr]) & set(groups[te])
        for name, (X, make) in models.items():
            thr = youden_threshold(make, X[tr], y[tr], groups[tr], rep)
            p = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            yp = (p >= thr).astype(int)
            rows.append(dict(modelo=name, rep=rep, fold=fold, umbral=thr,
                             auc_roc=roc_auc_score(y[te], p), auc_pr=average_precision_score(y[te], p),
                             accuracy=accuracy_score(y[te], yp), precision=precision_score(y[te], yp, zero_division=0),
                             recall=recall_score(y[te], yp), f1=f1_score(y[te], yp)))
        print(f'rep {rep} fold {fold} listo', flush=True)

res = pd.DataFrame(rows)
res.to_csv('B1_resultados_por_pliegue.csv', index=False)
print(f'\nMuestras {len(y)}, pacientes {len(set(groups))}, prevalencia {y.mean():.3f}')
print(f'{N_SPLITS} pliegues x {N_REPEATS} repeticiones, agrupados por paciente\n')
summary = res.groupby('modelo')[['auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1', 'umbral']].agg(['mean', 'std'])
print(summary.round(3).to_string())

# Diferencia pareada LR microbioma vs RF (mismos pliegues)
a = res[res.modelo == 'Random Forest microbioma'].auc_roc.values
b = res[res.modelo == 'LR microbioma (log, L2)'].auc_roc.values
d = a - b
print(f'\nAUC RF - LR microbioma: {d.mean():.3f} ± {d.std():.3f}; RF mejor en {(d > 0).sum()}/{len(d)} pliegues')
