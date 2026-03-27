import numpy as np
import os

# List of clients
clients = ['196', '279', '362', '364', '370']

# Layer order and shapes
layers = [
    ('backbone_0_weight', (128, 24)),
    ('backbone_0_bias', (128,)),
    ('backbone_3_weight', (64, 128)),
    ('backbone_3_bias', (64,)),
    ('backbone_6_weight', (32, 64)),
    ('backbone_6_bias', (32,)),
    ('head_h1_weight', (1, 32)),
    ('head_h1_bias', (1,)),
    ('head_h4_weight', (1, 32)),
    ('head_h4_bias', (1,)),
]

for client in clients:
    fixed_file = f'fixed_params_{client}.npz'
    if not os.path.exists(fixed_file):
        print(f"File {fixed_file} not found, skipping.")
        continue

    params = np.load(fixed_file)

    # Flatten all parameters
    flattened = []
    for layer, shape in layers:
        param = params[layer]
        if len(shape) == 2:
            # Flatten row-major
            flattened.extend(param.flatten().tolist())
        else:
            flattened.extend(param.tolist())

    # Convert to unsigned 16-bit for .coe (since signed, but BRAM is unsigned)
    # In .coe, values are interpreted as signed if needed
    # But to make it simple, store as is, since np.int16 is two's complement

    # Write .coe file
    with open(f'weights_{client}.coe', 'w') as f:
        f.write('memory_initialization_radix=10;\n')
        f.write('memory_initialization_vector=\n')
        for i, val in enumerate(flattened):
            f.write(f'{val}')
            if i < len(flattened) - 1:
                f.write(',\n')
            else:
                f.write(';\n')

    print(f"Generated weights_{client}.coe with {len(flattened)} values")

print("All .coe files generated.")