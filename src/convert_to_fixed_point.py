import numpy as np
import os

# List of client IDs
clients = ['196', '279', '362', '364', '370']

# Fixed-point parameters
total_bits = 16
frac_bits = 8
scale = 2 ** frac_bits  # 256
max_val = 2 ** (total_bits - 1) - 1  # 32767

for client in clients:
    params_file = f'model_params_{client}.npz'
    if not os.path.exists(params_file):
        print(f"File {params_file} not found, skipping.")
        continue

    params = np.load(params_file)
    fixed_params = {}

    for key, param in params.items():
        # Quantize to fixed-point
        quantized = np.round(param * scale).astype(np.int16)
        # Clip to range
        quantized = np.clip(quantized, -max_val, max_val)
        fixed_params[key] = quantized
        fixed_params[key + '_scale'] = scale

    np.savez(f'fixed_params_{client}.npz', **fixed_params)
    print(f"Saved fixed_params_{client}.npz")

print("All fixed-point conversions done.")