import torch

import torch

# Load the PyTorch model
model_path = 'model_MT_196.pt'
model = torch.load(model_path, weights_only=False)

print(f"Loaded model type: {type(model)}")
print(f"Model keys or attributes: {dir(model) if not isinstance(model, dict) else list(model.keys())}")

# Check if it's a state_dict or the whole model
if isinstance(model, dict):
    if 'model_state_dict' in model:
        state_dict = model['model_state_dict']
        print("Using model_state_dict")
    else:
        state_dict = model
        print("It's a state_dict dict")
else:
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        print("It's a model with state_dict")
    else:
        print("Unknown format")
        exit()

print(f"State dict keys: {list(state_dict.keys())}")

# Open the text file for writing
with open('weights_biases.txt', 'w') as f:
    for name, param in state_dict.items():
        f.write(f'Layer: {name}\n')
        f.write(f'Shape: {param.shape}\n')
        f.write(f'Values:\n{param.numpy()}\n\n')

print("Weights and biases have been written to weights_biases.txt")