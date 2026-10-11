from pathlib import Path
import re, json, zipfile
import nbformat
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score

root = Path(__file__).resolve().parent
nb = nbformat.read(root / 'TabPFN_3_5_Comparison.ipynb', as_version=4)
nbformat.validate(nb)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'notebook cell {i}', 'exec')
        assert not any(o.output_type == 'error' for o in cell.outputs)
results = pd.read_csv(root / 'results' / 'per_split_results.csv')
assert len(results) == 40
assert results.groupby(['dataset','seed']).model.nunique().eq(4).all()
environment = json.loads((root / 'results' / 'environment.json').read_text())
assert len(environment['checkpoint_sha256']) == 64
assert environment['checkpoint'] == 'tabpfn-v3.5-20260909.safetensors'
indices = json.loads((root / 'results' / 'split_indices.json').read_text())
for row in results.to_dict('records'):
    stem = f"{row['dataset'].lower().replace(' ', '_')}_{row['seed']}_{row['model'].lower().replace(' ', '_')}"
    preds = pd.read_csv(root / 'results' / f'predictions_{stem}.csv')
    probs = preds.filter(like='probability_class_').to_numpy()
    actual, labels = preds.actual, preds.prediction
    assert preds.row_index.tolist() == indices[f"{row['dataset']}_{row['seed']}"]['test']
    assert not set(preds.row_index) & set(indices[f"{row['dataset']}_{row['seed']}"]['train'])
    computed = {'accuracy':accuracy_score(actual,labels), 'balanced_accuracy':balanced_accuracy_score(actual,labels), 'macro_f1':f1_score(actual,labels,average='macro'), 'log_loss':log_loss(actual,probs), 'roc_auc':roc_auc_score(actual, probs[:,1]) if probs.shape[1]==2 else roc_auc_score(actual,probs,multi_class='ovr',average='macro')}
    for key,value in computed.items():
        assert np.isclose(value, row[key]), (stem,key,value,row[key])
    assert np.isclose(row['construct_seconds']+row['fit_seconds']+row['predict_seconds'], row['total_seconds'])
summary = pd.read_csv(root / 'results' / 'summary.csv').set_index(['dataset','model'])
grouped = results.groupby(['dataset','model'])
for metric in ['accuracy','balanced_accuracy','macro_f1','roc_auc','log_loss']:
    for statistic in ['mean','std']:
        computed = grouped[metric].agg(statistic).reindex(summary.index)
        assert np.allclose(summary[metric+'_'+statistic], computed)
for metric in ['construct_seconds','fit_seconds','predict_seconds','total_seconds']:
    computed = grouped[metric].median().reindex(summary.index)
    assert np.allclose(summary[metric+'_median'], computed)

print('Verified 40 prediction files, all score summaries, split membership, and timing totals.')
