`timescale 1ns / 1ps

module nn_combined (
    input  logic clk,
    input  logic rst,
    input  logic [2:0] model_select,            // 0..4
    input  logic start,
    input  logic signed [383:0] features_flat,   // 24 x 16-bit packed, feature[0] at [15:0]
    output logic signed [15:0] h1_out,
    output logic signed [15:0] h4_out,
    output logic done
);

    // ---------------- BRAM INTERFACE ----------------
    logic signed [15:0] bram_dout [0:4];
    logic [13:0]        bram_addr;
    logic               bram_en;

    bram_16x13602 #(.FILE_NAME("weights_196.hex")) bram0 (
        .clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[0])
    );
    bram_16x13602 #(.FILE_NAME("weights_279.hex")) bram1 (
        .clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[1])
    );
    bram_16x13602 #(.FILE_NAME("weights_362.hex")) bram2 (
        .clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[2])
    );
    bram_16x13602 #(.FILE_NAME("weights_364.hex")) bram3 (
        .clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[3])
    );
    bram_16x13602 #(.FILE_NAME("weights_370.hex")) bram4 (
        .clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout[4])
    );

    logic signed [15:0] weight;
    always_comb begin
        unique case (model_select)
            3'd0: weight = bram_dout[0];
            3'd1: weight = bram_dout[1];
            3'd2: weight = bram_dout[2];
            3'd3: weight = bram_dout[3];
            3'd4: weight = bram_dout[4];
            default: weight = '0;
        endcase
    end

    function automatic logic signed [15:0] get_feature(input int idx);
        begin
            get_feature = features_flat[idx*16 +: 16];
        end
    endfunction

    function automatic logic signed [47:0] mul_q8_8_to_q16_16(
        input logic signed [15:0] a,
        input logic signed [15:0] b
    );
        logic signed [31:0] p;
        begin
            p = a * b;                    // Q16.16
            mul_q8_8_to_q16_16 = {{16{p[31]}}, p};
        end
    endfunction

    function automatic logic signed [15:0] q8_8_from_acc(
        input logic signed [47:0] acc48
    );
        logic signed [47:0] scaled;
        begin
            scaled = acc48 >>> 8;         // back to Q8.8
            if (scaled > 48'sd32767)
                q8_8_from_acc = 16'sh7fff;
            else if (scaled < -48'sd32768)
                q8_8_from_acc = 16'sh8000;
            else
                q8_8_from_acc = scaled[15:0];
        end
    endfunction

    function automatic logic signed [15:0] relu_q8_8(
        input logic signed [47:0] acc48
    );
        logic signed [15:0] tmp;
        begin
            tmp = q8_8_from_acc(acc48);
            relu_q8_8 = (tmp < 0) ? 16'sd0 : tmp;
        end
    endfunction

    // ---------------- ADDRESS MAP ----------------
    localparam int BACKBONE0_WEIGHT_START = 0;
    localparam int BACKBONE0_WEIGHT_SIZE   = 128*24;
    localparam int BACKBONE0_BIAS_START    = BACKBONE0_WEIGHT_START + BACKBONE0_WEIGHT_SIZE; // 3072

    localparam int BACKBONE3_WEIGHT_START  = BACKBONE0_BIAS_START + 128;
    localparam int BACKBONE3_WEIGHT_SIZE   = 64*128;
    localparam int BACKBONE3_BIAS_START    = BACKBONE3_WEIGHT_START + BACKBONE3_WEIGHT_SIZE; // 11392

    localparam int BACKBONE6_WEIGHT_START  = BACKBONE3_BIAS_START + 64;
    localparam int BACKBONE6_WEIGHT_SIZE   = 32*64;
    localparam int BACKBONE6_BIAS_START    = BACKBONE6_WEIGHT_START + BACKBONE6_WEIGHT_SIZE; // 13504

    localparam int HEAD_H1_WEIGHT_START    = BACKBONE6_BIAS_START + 32;
    localparam int HEAD_H1_BIAS_START      = HEAD_H1_WEIGHT_START + 32; // 13568

    localparam int HEAD_H4_WEIGHT_START    = HEAD_H1_BIAS_START + 1;
    localparam int HEAD_H4_BIAS_START      = HEAD_H4_WEIGHT_START + 32; // 13601

    // ---------------- FSM ----------------
    typedef enum logic [3:0] {
        IDLE,
        L1_BIAS,
        L1_MAC,
        L2_BIAS,
        L2_MAC,
        L3_BIAS,
        L3_MAC,
        H1_BIAS,
        H1_MAC,
        H4_BIAS,
        H4_MAC,
        DONE
    } state_t;

    state_t state;

    logic signed [47:0] acc, sum;
    logic signed [15:0] h1_reg, h4_reg;
    logic done_reg;

    logic signed [15:0] hidden1 [0:127];
    logic signed [15:0] hidden2 [0:63];
    logic signed [15:0] hidden3 [0:31];

    logic [7:0] out_idx;
    logic [7:0] in_idx;

    assign h1_out = h1_reg;
    assign h4_out = h4_reg;
    assign done   = done_reg;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state     <= IDLE;
            acc       <= '0;
            sum       <= '0;
            h1_reg    <= '0;
            h4_reg    <= '0;
            done_reg  <= 1'b0;
            bram_en   <= 1'b1;
            bram_addr <= '0;
            out_idx   <= '0;
            in_idx    <= '0;
        end else begin
            done_reg <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        out_idx   <= 0;
                        in_idx    <= 0;
                        bram_addr <= BACKBONE0_BIAS_START;
                        state     <= L1_BIAS;
                    end
                end

                // ---------- Layer 1: 24 -> 128 ----------
                L1_BIAS: begin
                    acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
                    in_idx    <= 0;
                    bram_addr <= BACKBONE0_WEIGHT_START + (out_idx * 24);
                    state     <= L1_MAC;
                end

                L1_MAC: begin
                    sum = acc + mul_q8_8_to_q16_16(weight, get_feature(in_idx));

                    if (in_idx == 8'd23) begin
                        hidden1[out_idx] <= relu_q8_8(sum);

                        if (out_idx == 8'd127) begin
                            out_idx   <= 0;
                            in_idx    <= 0;
                            bram_addr <= BACKBONE3_BIAS_START;
                            state     <= L2_BIAS;
                        end else begin
                            out_idx   <= out_idx + 1;
                            in_idx    <= 0;
                            bram_addr <= BACKBONE0_BIAS_START + (out_idx + 1);
                            state     <= L1_BIAS;
                        end
                    end else begin
                        acc       <= sum;
                        in_idx    <= in_idx + 1;
                        bram_addr <= BACKBONE0_WEIGHT_START + (out_idx * 24) + (in_idx + 1);
                        state     <= L1_MAC;
                    end
                end

                // ---------- Layer 2: 128 -> 64 ----------
                L2_BIAS: begin
                    acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
                    in_idx    <= 0;
                    bram_addr <= BACKBONE3_WEIGHT_START + (out_idx * 128);
                    state     <= L2_MAC;
                end

                L2_MAC: begin
                    sum = acc + mul_q8_8_to_q16_16(weight, hidden1[in_idx]);

                    if (in_idx == 8'd127) begin
                        hidden2[out_idx] <= relu_q8_8(sum);

                        if (out_idx == 8'd63) begin
                            out_idx   <= 0;
                            in_idx    <= 0;
                            bram_addr <= BACKBONE6_BIAS_START;
                            state     <= L3_BIAS;
                        end else begin
                            out_idx   <= out_idx + 1;
                            in_idx    <= 0;
                            bram_addr <= BACKBONE3_BIAS_START + (out_idx + 1);
                            state     <= L2_BIAS;
                        end
                    end else begin
                        acc       <= sum;
                        in_idx    <= in_idx + 1;
                        bram_addr <= BACKBONE3_WEIGHT_START + (out_idx * 128) + (in_idx + 1);
                        state     <= L2_MAC;
                    end
                end

                // ---------- Layer 3: 64 -> 32 ----------
                L3_BIAS: begin
                    acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
                    in_idx    <= 0;
                    bram_addr <= BACKBONE6_WEIGHT_START + (out_idx * 64);
                    state     <= L3_MAC;
                end

                L3_MAC: begin
                    sum = acc + mul_q8_8_to_q16_16(weight, hidden2[in_idx]);

                    if (in_idx == 8'd63) begin
                        hidden3[out_idx] <= relu_q8_8(sum);

                        if (out_idx == 8'd31) begin
                            out_idx   <= 0;
                            in_idx    <= 0;
                            bram_addr <= HEAD_H1_BIAS_START;
                            state     <= H1_BIAS;
                        end else begin
                            out_idx   <= out_idx + 1;
                            in_idx    <= 0;
                            bram_addr <= BACKBONE6_BIAS_START + (out_idx + 1);
                            state     <= L3_BIAS;
                        end
                    end else begin
                        acc       <= sum;
                        in_idx    <= in_idx + 1;
                        bram_addr <= BACKBONE6_WEIGHT_START + (out_idx * 64) + (in_idx + 1);
                        state     <= L3_MAC;
                    end
                end

                // ---------- Head H1: 32 -> 1 ----------
                H1_BIAS: begin
                    acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
                    in_idx    <= 0;
                    bram_addr <= HEAD_H1_WEIGHT_START;
                    state     <= H1_MAC;
                end

                H1_MAC: begin
                    sum = acc + mul_q8_8_to_q16_16(weight, hidden3[in_idx]);

                    if (in_idx == 8'd31) begin
                        h1_reg    <= q8_8_from_acc(sum);
                        bram_addr <= HEAD_H4_BIAS_START;
                        state     <= H4_BIAS;
                    end else begin
                        acc       <= sum;
                        in_idx    <= in_idx + 1;
                        bram_addr <= HEAD_H1_WEIGHT_START + (in_idx + 1);
                        state     <= H1_MAC;
                    end
                end

                // ---------- Head H4: 32 -> 1 ----------
                H4_BIAS: begin
                    acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
                    in_idx    <= 0;
                    bram_addr <= HEAD_H4_WEIGHT_START;
                    state     <= H4_MAC;
                end

                H4_MAC: begin
                    sum = acc + mul_q8_8_to_q16_16(weight, hidden3[in_idx]);

                    if (in_idx == 8'd31) begin
                        h4_reg   <= q8_8_from_acc(sum);
                        state    <= DONE;
                    end else begin
                        acc       <= sum;
                        in_idx    <= in_idx + 1;
                        bram_addr <= HEAD_H4_WEIGHT_START + (in_idx + 1);
                        state     <= H4_MAC;
                    end
                end

                DONE: begin
                    done_reg <= 1'b1;
                    state    <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule


// Simulation-only BRAM
module bram_16x13602 #(
    parameter FILE_NAME = "weights_196.hex"
)(
    input  logic clka,
    input  logic ena,
    input  logic [13:0] addra,
    output logic signed [15:0] douta
);

    logic signed [15:0] mem [0:13601];

    initial begin
        $readmemh(FILE_NAME, mem);
        $display("Loaded BRAM from %s, first value: %h", FILE_NAME, mem[0]);
    end

    always_ff @(posedge clka) begin
        if (ena) begin
            douta <= mem[addra];
        end
    end

endmodule