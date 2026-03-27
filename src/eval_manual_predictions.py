import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

out = pd.read_csv('manual_predictions.csv')
import numpy as np
for h in ['h1','h4']:
    y_true = out['mt_196'].values
    y_pred = out[f'pred_{h}'].values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'{h} RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}')

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
    print(f'aligned {h} RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}')