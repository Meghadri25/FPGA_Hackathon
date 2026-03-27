import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

out = pd.read_csv('fixed_point_predictions.csv')
out['target_h1'] = out['mt_196'].shift(-1)
out['target_h4'] = out['mt_196'].shift(-4)
for h in ['h1','h4']:
    valid = out[f'target_{h}'].notna()
    y_true = out.loc[valid, f'target_{h}'].values
    y_pred = out.loc[valid, f'pred_{h}'].values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'Fixed-point aligned {h} RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}')

# Compare with original
orig = pd.read_csv('manual_predictions.csv')
print('\nComparison:')

# Merge on timestamp to align the dataframes
merged = pd.merge(orig, out[['timestamp', 'pred_h1', 'pred_h4']], on='timestamp', how='inner', suffixes=('_orig', '_fixed'))

for h in ['h1','h4']:
    orig_pred = merged[f'pred_{h}_orig'].values
    fixed_pred = merged[f'pred_{h}_fixed'].values
    diff = np.abs(orig_pred - fixed_pred)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f'{h} max diff: {max_diff:.2f}, mean diff: {mean_diff:.2f}')