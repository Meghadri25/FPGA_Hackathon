import numpy as np
import pandas as pd

# 1) Load weights/biases from exported NumPy file
params = np.load('model_params.npz')
backbone_0_weight = params['backbone_0_weight']  # shape (128, 24)
backbone_0_bias = params['backbone_0_bias']
backbone_3_weight = params['backbone_3_weight']  # shape (64, 128)
backbone_3_bias = params['backbone_3_bias']
backbone_6_weight = params['backbone_6_weight']  # shape (32, 64)
backbone_6_bias = params['backbone_6_bias']
head_h1_weight = params['head_h1_weight']  # shape (1, 32)
head_h1_bias = params['head_h1_bias']
head_h4_weight = params['head_h4_weight']  # shape (1, 32)
head_h4_bias = params['head_h4_bias']

# 2) Load dataset and pick client MT_196
# File uses semicolon delimiter and comma as decimal mark.
raw_df = pd.read_csv(
    'LD2011_2014.txt',
    sep=';',
    decimal=',',
    quotechar='"',
    low_memory=False,
)

# column 0 may be empty string; rename to timestamp
if raw_df.columns[0] == '':
    raw_df.rename(columns={'': 'timestamp'}, inplace=True)
else:
    raw_df.rename(columns={raw_df.columns[0]: 'timestamp'}, inplace=True)

# decode timestamp and values
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
raw_df['MT_196'] = pd.to_numeric(raw_df['MT_196'], errors='coerce')

# 3) choose the same windowing logic as train_decision
series = raw_df['MT_196'].fillna(0).values
rolling_std = pd.Series(series).rolling(window=10000).std()
start_idx = int(rolling_std.idxmax())
end_idx = start_idx + 100000
local = raw_df.iloc[start_idx:end_idx].copy()

# 4) feature engineering (same order as model training): lags + rolling + time
n_lags = 16
for i in range(1, n_lags + 1):
    local[f'lag_{i}'] = local['MT_196'].shift(i)

local['roll_mean_3'] = local['MT_196'].shift(1).rolling(3).mean()
local['roll_mean_12'] = local['MT_196'].shift(1).rolling(12).mean()
local['roll_std_12'] = local['MT_196'].shift(1).rolling(12).std()

local['hour'] = local['timestamp'].dt.hour
local['dow'] = local['timestamp'].dt.dayofweek
local['hour_sin'] = np.sin(2 * np.pi * local['hour'] / 24.0)
local['hour_cos'] = np.cos(2 * np.pi * local['hour'] / 24.0)
local['dow_sin'] = np.sin(2 * np.pi * local['dow'] / 7.0)
local['dow_cos'] = np.cos(2 * np.pi * local['dow'] / 7.0)
local['is_weekend'] = (local['dow'] >= 5).astype(float)

# target columns for verification (not used in inference path)
local['target_h1'] = local['MT_196'].shift(-1)
local['target_h4'] = local['MT_196'].shift(-4)

local = local.dropna().reset_index(drop=True)

# 5) Inputs for model
feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + [
    'roll_mean_3',
    'roll_mean_12',
    'roll_std_12',
    'hour_sin',
    'hour_cos',
    'dow_sin',
    'dow_cos',
    'is_weekend',
]
X = local[feature_cols].values.astype(np.float64)

# 6) MinMax scale X (same fit_transform style as original training)
x_min = X.min(axis=0)
x_max = X.max(axis=0)
denom = (x_max - x_min)
denom[denom == 0] = 1.0
X_scaled = (X - x_min) / denom

# 7) Forward pass manual (numpy)

def relu(x):
    return np.maximum(0, x)

h = relu(X_scaled @ backbone_0_weight.T + backbone_0_bias)
h = relu(h @ backbone_3_weight.T + backbone_3_bias)
feat = relu(h @ backbone_6_weight.T + backbone_6_bias)

pred_h1_scaled = feat @ head_h1_weight.T + head_h1_bias
pred_h4_scaled = feat @ head_h4_weight.T + head_h4_bias

# 8) Inverse scaling for targets (based on saved scaler params from checkpoint)
# y_scaled = y * scale (min=0), so y = y_scaled / scale
scale_y = 2.20183486e-05
pred_h1 = (pred_h1_scaled.flatten()) / scale_y
pred_h4 = (pred_h4_scaled.flatten()) / scale_y

# 9) Fill output DataFrame and save
out = pd.DataFrame({
    'timestamp': local['timestamp'],
    'mt_196': local['MT_196'],
    'pred_h1': pred_h1,
    'pred_h4': pred_h4,
})

out.to_csv('manual_predictions.csv', index=False)
print('Saved manual_predictions.csv with', len(out), 'rows')
print(out[['timestamp', 'mt_196', 'pred_h1', 'pred_h4']].head(10))
