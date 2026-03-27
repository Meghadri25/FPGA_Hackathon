import torch
import numpy as np
import os

# List of model files
model_files = [
    'model_MT_196.pt',
    'model_MT_279.pt',
    'model_MT_362.pt',
    'model_MT_364.pt',
    'model_MT_370.pt'
]

# Function to extract state dict
def get_state_dict(model_path):
    state = torch.load(model_path, weights_only=False)
    if 'model_state_dict' in state:
        return state['model_state_dict']
    else:
        return state

# Process each model
for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"Model {model_file} not found, skipping.")
        continue

    client_id = model_file.split('_')[2].split('.')[0]  # e.g., '196'
    print(f"Processing client {client_id}")

    state_dict = get_state_dict(model_file)

    # Save to npz
    params = {}
    for key, param in state_dict.items():
        params[key.replace('.', '_')] = param.cpu().numpy()

    np.savez(f'model_params_{client_id}.npz', **params)
    print(f"Saved model_params_{client_id}.npz")

print("All models processed.")