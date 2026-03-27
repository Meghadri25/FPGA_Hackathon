import numpy as np
import pandas as pd

print("Loading quantized params...")
# Load quantized params
qparams = np.load('quantized_params.npz')
print("Loaded qparams keys:", list(qparams.keys()))

# Load original params for scales
params = np.load('model_params.npz')
print("Loaded params")

# Function to do fixed-point linear
def fixed_linear(x, weight_key, bias_key):
    weight = qparams[weight_key]
    bias = qparams[bias_key]
    w_scale = qparams[weight_key + '_scale']
    b_scale = qparams[bias_key + '_scale']
    # Assume input x is already scaled appropriately
    # For simplicity, do float mul but with quantized weights
    out = x @ (weight.T.astype(np.float32) * w_scale) + (bias.astype(np.float32) * b_scale)
    return out

# Load dataset and prepare as before
raw_df = pd.read_csv(
    'LD2011_2014.txt',
    sep=';',
    decimal=',',
    quotechar='"',
    low_memory=False,
)

if raw_df.columns[0] == '':
    raw_df.rename(columns={'': 'timestamp'}, inplace=True)
else:
    raw_df.rename(columns={raw_df.columns[0]: 'timestamp'}, inplace=True)

raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
raw_df['MT_196'] = pd.to_numeric(raw_df['MT_196'], errors='coerce')

series = raw_df['MT_196'].fillna(0).values
rolling_std = pd.Series(series).rolling(window=10000).std()
start_idx = int(rolling_std.idxmax())
end_idx = start_idx + 100000
local = raw_df.iloc[start_idx:end_idx].copy()

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

# Drop rows with NaN values
local = local.dropna()

X = local[feature_cols].values.astype(np.float64)

x_min = X.min(axis=0)
x_max = X.max(axis=0)
denom = (x_max - x_min)
denom[denom == 0] = 1.0
X_scaled = (X - x_min) / denom

# Quantize X_scaled to 16-bit signed
X_quant = np.round(X_scaled * 32767).astype(np.int16)

# Forward pass with fixed-point (approximate)
def relu(x):
    return np.maximum(0, x)

print("Starting forward pass...")
# Layer 1
h1 = relu(fixed_linear(X_quant.astype(np.float32) / 32767, 'backbone_0_weight', 'backbone_0_bias'))
print("Layer 1 done")
# Layer 2
h2 = relu(fixed_linear(h1, 'backbone_3_weight', 'backbone_3_bias'))
print("Layer 2 done")
# Layer 3
h3 = relu(fixed_linear(h2, 'backbone_6_weight', 'backbone_6_bias'))
print("Layer 3 done")

# Heads
pred_h1_scaled = fixed_linear(h3, 'head_h1_weight', 'head_h1_bias')
pred_h4_scaled = fixed_linear(h3, 'head_h4_weight', 'head_h4_bias')
print("Heads done")

# Inverse scale
scale_y = 2.20183486e-05
pred_h1 = pred_h1_scaled.flatten() / scale_y
pred_h4 = pred_h4_scaled.flatten() / scale_y

out = pd.DataFrame({
    'timestamp': local['timestamp'],
    'mt_196': local['MT_196'],
    'pred_h1': pred_h1,
    'pred_h4': pred_h4,
})

out.to_csv('fixed_point_predictions.csv', index=False)
print('Saved fixed_point_predictions.csv with', len(out), 'rows')
print(out[['timestamp', 'mt_196', 'pred_h1', 'pred_h4']].head(10))