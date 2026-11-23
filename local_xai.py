"""Read precomputed per-recommendation SHAP and LIME from batch SQLite shards."""
import json
import sqlite3
from contextlib import closing
from contextlib import closing
from pathlib import Path

import pandas as pd


def lookup_local_explanations(data_dir, learner, course):
    folder = Path(data_dir) / 'local_xai'
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob('xai_batch_*.sqlite')):
        # Database is read-only on Streamlit Cloud. Don't copy all cases into memory.
        with closing(sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)) as conn:
            row = conn.execute('SELECT score, baseline, shap_json, lime_json, lime_intercept, lime_fidelity, samples FROM explanations WHERE learner=? AND course=?', (str(learner), str(course))).fetchone()
        if row is None:
            continue
        score, baseline, shap_data, lime_data, intercept, fidelity, samples = row
        shap_df = pd.DataFrame(json.loads(shap_data), columns=['Feature','SHAP_Value'])
        shap_df['Abs_SHAP'] = shap_df['SHAP_Value'].abs()
        lime_df = pd.DataFrame(json.loads(lime_data), columns=['Feature', 'LIME_Weight'])
        lime_df['Abs_LIME'] = lime_df['LIME_Weight'].abs()
        return dict(score=score, baseline=baseline, shap=shap_df, lime=lime_df,
                    lime_intercept=intercept, lime_fidelity=fidelity, samples=samples)
    return None
