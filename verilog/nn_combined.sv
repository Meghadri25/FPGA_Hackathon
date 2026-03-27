// Combined Neural Network FPGA Module for 5 Clients
// Uses BRAM for weights, DSP slices for multiplication
// For PYNQ Z2 (Zynq-7000)

module nn_fpga (
    input clk,
    input rst,
    input [2:0] model_select,  // 0-4 for clients 196,279,362,364,370
    input start,
    input signed [15:0] features [0:23],  // Q8.8
    output signed [15:0] h1_out,  // Q8.8
    output signed [15:0] h4_out,  // Q8.8
    output done
);

// BRAM interfaces (to be connected to Block Memory Generator IPs)
wire [15:0] bram_dout [0:4];
reg [13:0] bram_addr;
reg bram_en;

// Instantiate 5 BRAMs (in Vivado, use IP with .coe files)
bram_16x13602 bram0 (.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[0]));
defparam bram0.FILE_NAME = "weights_196.hex";
bram_16x13602 bram1 (.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[1]));
defparam bram1.FILE_NAME = "weights_279.hex";
bram_16x13602 bram2 (.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[2]));
defparam bram2.FILE_NAME = "weights_362.hex";
bram_16x13602 bram3 (.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[3]));
defparam bram3.FILE_NAME = "weights_364.hex";
bram_16x13602 bram4 (.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[4]));
defparam bram4.FILE_NAME = "weights_370.hex";

wire signed [15:0] weight = bram_dout[model_select];  // Select based on model

// Address offsets
localparam BACKBONE0_WEIGHT_START = 0;
localparam BACKBONE0_WEIGHT_SIZE = 128*24;  // 3072
localparam BACKBONE0_BIAS_START = BACKBONE0_WEIGHT_START + BACKBONE0_WEIGHT_SIZE;  // 3072
localparam BACKBONE0_BIAS_SIZE = 128;
localparam BACKBONE3_WEIGHT_START = BACKBONE0_BIAS_START + BACKBONE0_BIAS_SIZE;  // 3200
localparam BACKBONE3_WEIGHT_SIZE = 64*128;  // 8192
localparam BACKBONE3_BIAS_START = BACKBONE3_WEIGHT_START + BACKBONE3_WEIGHT_SIZE;  // 11392
localparam BACKBONE3_BIAS_SIZE = 64;
localparam BACKBONE6_WEIGHT_START = BACKBONE3_BIAS_START + BACKBONE3_BIAS_SIZE;  // 11456
localparam BACKBONE6_WEIGHT_SIZE = 32*64;  // 2048
localparam BACKBONE6_BIAS_START = BACKBONE6_WEIGHT_START + BACKBONE6_WEIGHT_SIZE;  // 13504
localparam BACKBONE6_BIAS_SIZE = 32;
localparam HEAD_H1_WEIGHT_START = BACKBONE6_BIAS_START + BACKBONE6_BIAS_SIZE;  // 13536
localparam HEAD_H1_WEIGHT_SIZE = 32;
localparam HEAD_H1_BIAS_START = HEAD_H1_WEIGHT_START + HEAD_H1_WEIGHT_SIZE;  // 13568
localparam HEAD_H1_BIAS_SIZE = 1;
localparam HEAD_H4_WEIGHT_START = HEAD_H1_BIAS_START + HEAD_H1_BIAS_SIZE;  // 13569
localparam HEAD_H4_WEIGHT_SIZE = 32;
localparam HEAD_H4_BIAS_START = HEAD_H4_WEIGHT_START + HEAD_H4_WEIGHT_SIZE;  // 13601
localparam HEAD_H4_BIAS_SIZE = 1;

// FSM states
localparam IDLE = 0;
localparam LAYER1 = 1;
localparam LAYER2 = 2;
localparam LAYER3 = 3;
localparam COMPUTE_H1 = 4;
localparam COMPUTE_H4 = 5;
localparam DONE = 6;

reg [2:0] state;
reg signed [31:0] acc;  // Accumulator for MAC
reg signed [15:0] h1_reg, h4_reg;
reg done_reg;

// Intermediate storage
reg signed [31:0] hidden1 [0:127];  // Layer 1 outputs
reg signed [31:0] hidden2 [0:63];   // Layer 2
reg signed [31:0] hidden3 [0:31];   // Layer 3

// Counters
reg [7:0] out_idx;  // 0-127 for layer1, etc.
reg [7:0] in_idx;   // 0-23 for layer1, 0-127 for layer2, etc.

// DSP inference: use (* use_dsp48 = "yes" *) for multiplies
(* use_dsp48 = "yes" *) wire signed [31:0] mult_result = weight * (state == LAYER1 ? features[in_idx] : 
                                                                 (state == LAYER2 ? (hidden1[in_idx] >>> 8) : 
                                                                  (state == LAYER3 ? (hidden2[in_idx] >>> 8) : 
                                                                   (state == COMPUTE_H1 || state == COMPUTE_H4 ? features[in_idx] : 0))));

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        acc <= 0;
        h1_reg <= 0;
        h4_reg <= 0;
        done_reg <= 0;
        bram_en <= 1;  // Always enable
        bram_addr <= 0;
        out_idx <= 0;
        in_idx <= 0;
    end else begin
        case (state)
            IDLE: begin
                if (start) begin
                    state <= LAYER1;
                    acc <= 0;
                    out_idx <= 0;
                    in_idx <= 0;
                end
                done_reg <= 0;
            end
            LAYER1: begin
                // 128 outputs, 24 inputs
                if (out_idx < 128) begin
                    if (in_idx == 0) begin
                        // Load bias
                        bram_addr <= BACKBONE0_BIAS_START + out_idx;
                        acc <= {{16{weight[15]}}, weight};  // Sign extend bias
                        in_idx <= 1;
                    end else if (in_idx <= 24) begin
                        bram_addr <= BACKBONE0_WEIGHT_START + out_idx * 24 + (in_idx - 1);
                        acc <= acc + mult_result;
                        in_idx <= in_idx + 1;
                    end else begin
                        // ReLU
                        hidden1[out_idx] <= (acc < 0) ? 0 : acc;
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                    end
                end else begin
                    state <= LAYER2;
                    out_idx <= 0;
                    in_idx <= 0;
                end
            end
            LAYER2: begin
                // 64 outputs, 128 inputs
                if (out_idx < 64) begin
                    if (in_idx == 0) begin
                        bram_addr <= BACKBONE3_BIAS_START + out_idx;
                        acc <= {{16{weight[15]}}, weight};
                        in_idx <= 1;
                    end else if (in_idx <= 128) begin
                        bram_addr <= BACKBONE3_WEIGHT_START + out_idx * 128 + (in_idx - 1);
                        acc <= acc + mult_result;
                        in_idx <= in_idx + 1;
                    end else begin
                        hidden2[out_idx] <= (acc < 0) ? 0 : acc;
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                    end
                end else begin
                    state <= LAYER3;
                    out_idx <= 0;
                    in_idx <= 0;
                end
            end
            LAYER3: begin
                // 32 outputs, 64 inputs
                if (out_idx < 32) begin
                    if (in_idx == 0) begin
                        bram_addr <= BACKBONE6_BIAS_START + out_idx;
                        acc <= {{16{weight[15]}}, weight};
                        in_idx <= 1;
                    end else if (in_idx <= 64) begin
                        bram_addr <= BACKBONE6_WEIGHT_START + out_idx * 64 + (in_idx - 1);
                        acc <= acc + mult_result;
                        in_idx <= in_idx + 1;
                    end else begin
                        hidden3[out_idx] <= (acc < 0) ? 0 : acc;
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                    end
                end else begin
                    state <= COMPUTE_H1;
                    out_idx <= 0;
                    in_idx <= 0;
                    acc <= 0;
                end
            end
            COMPUTE_H1: begin
                // 1 output, 32 inputs
                if (in_idx == 0) begin
                    bram_addr <= HEAD_H1_BIAS_START;
                    acc <= {{16{weight[15]}}, weight};
                    in_idx <= 1;
                end else if (in_idx <= 32) begin
                    bram_addr <= HEAD_H1_WEIGHT_START + (in_idx - 1);
                    acc <= acc + mult_result;  // mult_result uses features[in_idx-1]
                    in_idx <= in_idx + 1;
                end else begin
                    h1_reg <= acc[23:8];
                    state <= COMPUTE_H4;
                    in_idx <= 0;
                    acc <= 0;
                end
            end
            COMPUTE_H4: begin
                // 1 output, 32 inputs
                if (in_idx == 0) begin
                    bram_addr <= HEAD_H4_BIAS_START;
                    acc <= {{16{weight[15]}}, weight};
                    in_idx <= 1;
                end else if (in_idx <= 32) begin
                    bram_addr <= HEAD_H4_WEIGHT_START + (in_idx - 1);
                    acc <= acc + mult_result;
                    in_idx <= in_idx + 1;
                end else begin
                    h4_reg <= acc[23:8];
                    state <= DONE;
                end
            end
            DONE: begin
                done_reg <= 1;
                state <= IDLE;
            end
        endcase
    end
end

assign h1_out = h1_reg;
assign h4_out = h4_reg;
assign done = done_reg;

endmodule

// BRAM module (for simulation, loads from hex file)
module bram_16x13602 (
    input clka,
    input ena,
    input [13:0] addra,
    output reg [15:0] douta
);

parameter FILE_NAME = "weights_196.hex";

reg [15:0] mem [0:13601];

initial begin
    $readmemh(FILE_NAME, mem);
end

always @(posedge clka) begin
    if (ena) begin
        douta <= mem[addra];
    end
end

endmodule