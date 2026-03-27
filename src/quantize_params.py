import numpy as np

# Load original params
params = np.load('model_params.npz')

# Quantize weights to 8-bit signed, biases to 16-bit signed
quantized_params = {}

for key, param in params.items():
    if 'weight' in key:
        # 8-bit signed: -128 to 127
        scale = np.max(np.abs(param)) / 127.0
        quantized = np.round(param / scale).astype(np.int8)
        quantized_params[key] = quantized
        quantized_params[key + '_scale'] = scale
    elif 'bias' in key:
        # 16-bit signed: -32768 to 32767
        scale = np.max(np.abs(param)) / 32767.0
        quantized = np.round(param / scale).astype(np.int16)
        quantized_params[key] = quantized
        quantized_params[key + '_scale'] = scale

# Save quantized params
np.savez('quantized_params.npz', **quantized_params)
print('Saved quantized_params.npz')