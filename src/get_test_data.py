import numpy as np
import pandas as pd

print("Loading fixed params...")
# Load fixed params for client 196
params = np.load('fixed_params_196.npz')

print("Loading dataset...")
# Load dataset and prepare as in fixed_point_model.py
raw_df = pd.read_csv(
    'LD2011_2014.txt',
    sep=';',
    decimal=',',
    quotechar='"',
    low_memory=False,
    nrows=1000  # Load only first 1000 rows for speed
)

if raw_df.columns[0] == '':
    raw_df.rename(columns={'': 'timestamp'}, inplace=True)
else:
    raw_df.rename(columns={raw_df.columns[0]: 'timestamp'}, inplace=True)

raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
raw_df['MT_196'] = pd.to_numeric(raw_df['MT_196'], errors='coerce')

series = raw_df['MT_196'].fillna(0).values
rolling_std = pd.Series(series).rolling(window=100).std()
start_idx = int(rolling_std.idxmax())
end_idx = start_idx + 1000
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

local = local.dropna()
X = local[feature_cols].values.astype(np.float64)

x_min = X.min(axis=0)
x_max = X.max(axis=0)
denom = (x_max - x_min)
denom[denom == 0] = 1.0
X_scaled = (X - x_min) / denom

# Quantize X_scaled to 16-bit signed
X_quant = np.round(X_scaled * 32767).astype(np.int16)

# Use first sample
features = X_quant[0]

print("Features (16-bit signed):")
for i, f in enumerate(features):
    print(f"features[{i}] = 16'd{f};")

# Now compute expected h1 and h4 using fixed-point
scale = 256.0

def relu(x):
    return np.maximum(0, x)

# Layer 1
h1 = np.zeros(128, dtype=np.int32)
for r in range(128):
    acc = params['backbone_0_bias'][r]
    for c in range(24):
        acc += params['backbone_0_weight'][r, c] * features[c]
    h1[r] = max(0, acc)

# Layer 2
h2 = np.zeros(64, dtype=np.int32)
for r in range(64):
    acc = params['backbone_3_bias'][r]
    for c in range(128):
        acc += params['backbone_3_weight'][r, c] * (h1[c] >> 8)
    h2[r] = max(0, acc)

# Layer 3
h3 = np.zeros(32, dtype=np.int32)
for r in range(32):
    acc = params['backbone_6_bias'][r]
    for c in range(64):
        acc += params['backbone_6_weight'][r, c] * (h2[c] >> 8)
    h3[r] = max(0, acc)

# Head h1
h1_out_acc = params['head_h1_bias'][0]
for c in range(32):
    h1_out_acc += params['head_h1_weight'][0, c] * (h3[c] >> 8)
h1_out = h1_out_acc >> 8  # Since Q16.16 to Q8.8

# Head h4
h4_out_acc = params['head_h4_bias'][0]
for c in range(32):
    h4_out_acc += params['head_h4_weight'][0, c] * (h3[c] >> 8)
h4_out = h4_out_acc >> 8

print(f"Expected h1_out: {h1_out}")
print(f"Expected h4_out: {h4_out}")