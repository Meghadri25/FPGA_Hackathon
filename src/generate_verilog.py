import numpy as np
import os

# List of clients
clients = ['196', '279', '362', '364', '370']

# Fixed-point scale
scale = 256.0

for client in clients:
    fixed_file = f'fixed_params_{client}.npz'
    if not os.path.exists(fixed_file):
        print(f"File {fixed_file} not found, skipping.")
        continue

    params = np.load(fixed_file)

    # Generate Verilog module
    verilog_code = f"""// Neural Network for Client {client}
// Fixed-point Q8.8 (16-bit signed)

module nn_client_{client} (
    input clk,
    input rst,
    input [15:0] features [0:23],  // 24 features, Q8.8
    output [15:0] h1_out,  // Q8.8
    output [15:0] h4_out   // Q8.8
);

// Parameters
"""

    # Define weights and biases as parameters
    layer_keys = [
        ('backbone_0', 'weight', (128, 24)),
        ('backbone_0', 'bias', (128,)),
        ('backbone_3', 'weight', (64, 128)),
        ('backbone_3', 'bias', (64,)),
        ('backbone_6', 'weight', (32, 64)),
        ('backbone_6', 'bias', (32,)),
        ('head_h1', 'weight', (1, 32)),
        ('head_h1', 'bias', (1,)),
        ('head_h4', 'weight', (1, 32)),
        ('head_h4', 'bias', (1,)),
    ]

    for layer, type_, shape in layer_keys:
        key = f'{layer}_{type_}'
        param = params[key]
        if len(shape) == 2:
            rows, cols = shape
            verilog_code += f"parameter signed [15:0] {key} [0:{rows-1}][0:{cols-1}] = '{{\n"
            for r in range(rows):
                row_vals = ', '.join(f"16'd{int(param[r,c])}" for c in range(cols))
                verilog_code += f"    {{{row_vals}}}"
                if r < rows-1:
                    verilog_code += ','
                verilog_code += '\n'
            verilog_code += "};\n\n"
        else:
            # bias
            vals = ', '.join(f"16'd{int(param[i])}" for i in range(shape[0]))
            verilog_code += f"parameter signed [15:0] {key} [0:{shape[0]-1}] = {{{vals}}};\n\n"

    # Now, implement the computation
    verilog_code += """
// Internal signals
reg signed [31:0] h1 [0:127];  // Layer 1 outputs, accumulated
reg signed [31:0] h2 [0:63];   // Layer 2
reg signed [31:0] h3 [0:31];   // Layer 3
reg signed [31:0] out_h1;      // Final h1
reg signed [31:0] out_h4;      // Final h4

// Computation
always @(posedge clk or posedge rst) begin
    if (rst) begin
        // Reset
        integer i;
        for (i = 0; i < 128; i = i + 1) h1[i] <= 0;
        for (i = 0; i < 64; i = i + 1) h2[i] <= 0;
        for (i = 0; i < 32; i = i + 1) h3[i] <= 0;
        out_h1 <= 0;
        out_h4 <= 0;
    end else begin
        // Layer 1: 128 x 24
        integer r, c;
        for (r = 0; r < 128; r = r + 1) begin
            h1[r] = backbone_0_bias[r];
            for (c = 0; c < 24; c = c + 1) begin
                h1[r] = h1[r] + backbone_0_weight[r][c] * features[c];
            end
            // ReLU
            if (h1[r] < 0) h1[r] = 0;
        end

        // Layer 2: 64 x 128
        for (r = 0; r < 64; r = r + 1) begin
            h2[r] = backbone_3_bias[r];
            for (c = 0; c < 128; c = c + 1) begin
                h2[r] = h2[r] + backbone_3_weight[r][c] * (h1[c] >>> 8);  // Divide by 256 for fixed-point
            end
            if (h2[r] < 0) h2[r] = 0;
        end

        // Layer 3: 32 x 64
        for (r = 0; r < 32; r = r + 1) begin
            h3[r] = backbone_6_bias[r];
            for (c = 0; c < 64; c = c + 1) begin
                h3[r] = h3[r] + backbone_6_weight[r][c] * (h2[c] >>> 8);
            end
            if (h3[r] < 0) h3[r] = 0;
        end

        // Head h1: 1 x 32
        out_h1 = head_h1_bias[0];
        for (c = 0; c < 32; c = c + 1) begin
            out_h1 = out_h1 + head_h1_weight[0][c] * (h3[c] >>> 8);
        end

        // Head h4: 1 x 32
        out_h4 = head_h4_bias[0];
        for (c = 0; c < 32; c = c + 1) begin
            out_h4 = out_h4 + head_h4_weight[0][c] * (h3[c] >>> 8);
        end
    end
end

// Outputs
assign h1_out = out_h1[23:8];  // Take middle 16 bits
assign h4_out = out_h4[23:8];

endmodule
"""

    # Write to file
    with open(f'nn_client_{client}.sv', 'w') as f:
        f.write(verilog_code)

    print(f"Generated nn_client_{client}.sv")

print("All Verilog files generated.")