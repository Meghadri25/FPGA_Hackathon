import os

# Convert .coe to .hex for $readmemh

clients = ['196', '279', '362', '364', '370']

for client in clients:
    coe_file = f'weights_{client}.coe'
    hex_file = f'weights_{client}.hex'

    if not os.path.exists(coe_file):
        print(f"{coe_file} not found")
        continue

    with open(coe_file, 'r') as f:
        lines = f.readlines()

    # Find the vector line
    data_start = False
    data = []
    for line in lines:
        line = line.strip()
        if 'memory_initialization_vector=' in line:
            data_start = True
            continue
        if data_start:
            if line.endswith(';'):
                line = line[:-1]
            if line:
                vals = line.split(',')
                for val in vals:
                    val = val.strip()
                    if val:
                        # Convert to int, then to hex
                        num = int(val)
                        # To 16-bit hex, two's complement
                        if num < 0:
                            num = (1 << 16) + num
                        hex_val = f"{num:04X}"
                        data.append(hex_val)

    # Write to hex file
    with open(hex_file, 'w') as f:
        for val in data:
            f.write(f"{val}\n")

    print(f"Converted {coe_file} to {hex_file} with {len(data)} values")