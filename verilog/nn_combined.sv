`timescale 1ns / 1ps

// Low-I/O top module.
module nn_combined (
	input  logic clk,
	input  logic rst,
	input  logic [2:0] model_select,
	input  logic start_load,
	input  logic feature_valid,
	input  logic signed [15:0] feature_in,
	output logic feature_ready,
	output logic signed [15:0] h1_out,
	output logic signed [15:0] h4_out,
	output logic done,
	output logic [3:0] core_state_out,
	output logic [4:0] load_count_out
);

	typedef enum logic [1:0] {
		S_IDLE,
		S_LOAD,
		S_RUN
	} top_state_t;

	top_state_t top_state;

	logic signed [383:0] features_flat_reg;
	logic [4:0] load_count;
	logic core_start;
	logic core_done;

	nn_combined_core u_core (
		.clk(clk),
		.rst(rst),
		.model_select(model_select),
		.start(core_start),
		.features_flat(features_flat_reg),
		.h1_out(h1_out),
		.h4_out(h4_out),
		.done(core_done),
		.state_out(core_state_out)
	);

	assign feature_ready = (top_state == S_LOAD);
	assign load_count_out = load_count;

	always_ff @(posedge clk or posedge rst) begin
		if (rst) begin
			top_state          <= S_IDLE;
			features_flat_reg  <= '0;
			load_count         <= '0;
			core_start         <= 1'b0;
			done               <= 1'b0;
		end else begin
			core_start <= 1'b0;
			done       <= 1'b0;

			case (top_state)
				S_IDLE: begin
					load_count <= '0;
					if (start_load) begin
						features_flat_reg <= '0;
						top_state         <= S_LOAD;
					end
				end

				S_LOAD: begin
					if (feature_valid) begin
						features_flat_reg[load_count*16 +: 16] <= feature_in;

						if (load_count == 5'd23) begin
							core_start <= 1'b1;
							top_state  <= S_RUN;
						end else begin
							load_count <= load_count + 1'b1;
						end
					end
				end

				S_RUN: begin
					if (core_done) begin
						done      <= 1'b1;
						top_state <= S_IDLE;
					end
				end

				default: top_state <= S_IDLE;
			endcase
		end
	end

endmodule


// Backward-compatible alias for existing testbenches/scripts.
module nn_combined_serial_top (
	input  logic clk,
	input  logic rst,
	input  logic [2:0] model_select,
	input  logic start_load,
	input  logic feature_valid,
	input  logic signed [15:0] feature_in,
	output logic feature_ready,
	output logic signed [15:0] h1_out,
	output logic signed [15:0] h4_out,
	output logic done,
	output logic [3:0] core_state_out,
	output logic [4:0] load_count_out
);

	nn_combined u_top (
		.clk(clk),
		.rst(rst),
		.model_select(model_select),
		.start_load(start_load),
		.feature_valid(feature_valid),
		.feature_in(feature_in),
		.feature_ready(feature_ready),
		.h1_out(h1_out),
		.h4_out(h4_out),
		.done(done),
		.core_state_out(core_state_out),
		.load_count_out(load_count_out)
	);

endmodule


module nn_combined_core (
	input  logic clk,
	input  logic rst,
	input  logic [2:0] model_select,            // 0..4
	input  logic start,
	input  logic signed [383:0] features_flat,   // 24 x 16-bit packed, feature[0] at [15:0]
	output logic signed [15:0] h1_out,
	output logic signed [15:0] h4_out,
	output logic done,
	output logic [3:0] state_out
);

	// ---------------- BRAM INTERFACE ----------------
	logic signed [17:0] bram_dout18 [0:4];
	logic signed [15:0] bram_dout [0:4];
	logic [13:0]        bram_addr;
	logic               bram_en;

	assign bram_en = 1'b1;

	// Direct BRAM IP instances (Vivado blk_mem_gen outputs).
	blk_mem_gen_0 bram0 (
		.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout18[0])
	);
	blk_mem_gen_1 bram1 (
		.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout18[1])
	);
	blk_mem_gen_2 bram2 (
		.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout18[2])
	);
	blk_mem_gen_3 bram3 (
		.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout18[3])
	);
	blk_mem_gen_4 bram4 (
		.clka(clk), .ena(bram_en), .addra(bram_addr), .douta(bram_dout18[4])
	);

	genvar gi;
	generate
		for (gi = 0; gi < 5; gi = gi + 1) begin : g_weight_slice
			assign bram_dout[gi] = bram_dout18[gi][15:0];
		end
	endgenerate

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
	assign state_out = state;

	// Combinational BRAM prefetch address generation for synchronous BRAM IP.
	// This keeps weight data aligned with the next FSM cycle's math operation.
	always_comb begin
		bram_addr = '0;

		unique case (state)
			IDLE: begin
				if (start) bram_addr = BACKBONE0_BIAS_START;
			end

			L1_BIAS: bram_addr = BACKBONE0_WEIGHT_START + (out_idx * 24);
			L1_MAC: begin
				if (in_idx == 8'd23) begin
					if (out_idx == 8'd127)
						bram_addr = BACKBONE3_BIAS_START;
					else
						bram_addr = BACKBONE0_BIAS_START + (out_idx + 1);
				end else begin
					bram_addr = BACKBONE0_WEIGHT_START + (out_idx * 24) + (in_idx + 1);
				end
			end

			L2_BIAS: bram_addr = BACKBONE3_WEIGHT_START + (out_idx * 128);
			L2_MAC: begin
				if (in_idx == 8'd127) begin
					if (out_idx == 8'd63)
						bram_addr = BACKBONE6_BIAS_START;
					else
						bram_addr = BACKBONE3_BIAS_START + (out_idx + 1);
				end else begin
					bram_addr = BACKBONE3_WEIGHT_START + (out_idx * 128) + (in_idx + 1);
				end
			end

			L3_BIAS: bram_addr = BACKBONE6_WEIGHT_START + (out_idx * 64);
			L3_MAC: begin
				if (in_idx == 8'd63) begin
					if (out_idx == 8'd31)
						bram_addr = HEAD_H1_BIAS_START;
					else
						bram_addr = BACKBONE6_BIAS_START + (out_idx + 1);
				end else begin
					bram_addr = BACKBONE6_WEIGHT_START + (out_idx * 64) + (in_idx + 1);
				end
			end

			H1_BIAS: bram_addr = HEAD_H1_WEIGHT_START;
			H1_MAC: begin
				if (in_idx == 8'd31)
					bram_addr = HEAD_H4_BIAS_START;
				else
					bram_addr = HEAD_H1_WEIGHT_START + (in_idx + 1);
			end

			H4_BIAS: bram_addr = HEAD_H4_WEIGHT_START;
			H4_MAC: begin
				if (in_idx != 8'd31)
					bram_addr = HEAD_H4_WEIGHT_START + (in_idx + 1);
			end

			default: bram_addr = '0;
		endcase
	end

	always_ff @(posedge clk or posedge rst) begin
		if (rst) begin
			state     <= IDLE;
			acc       <= '0;
			sum       <= '0;
			h1_reg    <= '0;
			h4_reg    <= '0;
			done_reg  <= 1'b0;
			out_idx   <= '0;
			in_idx    <= '0;
		end else begin
			done_reg <= 1'b0;

			case (state)
				IDLE: begin
					if (start) begin
						out_idx   <= 0;
						in_idx    <= 0;
						state     <= L1_BIAS;
						$display("%0t ns: FSM started, model_select=%0d", $time, model_select);
					end
				end

				// ---------- Layer 1: 24 -> 128 ----------
				L1_BIAS: begin
					acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
					in_idx    <= 0;
					state     <= L1_MAC;
				end

				L1_MAC: begin
					sum = acc + mul_q8_8_to_q16_16(weight, get_feature(in_idx));

					if (in_idx == 8'd23) begin
						hidden1[out_idx] <= relu_q8_8(sum);

						if (out_idx == 8'd127) begin
							out_idx   <= 0;
							in_idx    <= 0;
							state     <= L2_BIAS;
						end else begin
							out_idx   <= out_idx + 1;
							in_idx    <= 0;
							state     <= L1_BIAS;
						end
					end else begin
						acc       <= sum;
						in_idx    <= in_idx + 1;
						state     <= L1_MAC;
					end
				end

				// ---------- Layer 2: 128 -> 64 ----------
				L2_BIAS: begin
					acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
					in_idx    <= 0;
					state     <= L2_MAC;
				end

				L2_MAC: begin
					sum = acc + mul_q8_8_to_q16_16(weight, hidden1[in_idx]);

					if (in_idx == 8'd127) begin
						hidden2[out_idx] <= relu_q8_8(sum);

						if (out_idx == 8'd63) begin
							out_idx   <= 0;
							in_idx    <= 0;
							state     <= L3_BIAS;
						end else begin
							out_idx   <= out_idx + 1;
							in_idx    <= 0;
							state     <= L2_BIAS;
						end
					end else begin
						acc       <= sum;
						in_idx    <= in_idx + 1;
						state     <= L2_MAC;
					end
				end

				// ---------- Layer 3: 64 -> 32 ----------
				L3_BIAS: begin
					acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
					in_idx    <= 0;
					state     <= L3_MAC;
				end

				L3_MAC: begin
					sum = acc + mul_q8_8_to_q16_16(weight, hidden2[in_idx]);

					if (in_idx == 8'd63) begin
						hidden3[out_idx] <= relu_q8_8(sum);

						if (out_idx == 8'd31) begin
							out_idx   <= 0;
							in_idx    <= 0;
							state     <= H1_BIAS;
						end else begin
							out_idx   <= out_idx + 1;
							in_idx    <= 0;
							state     <= L3_BIAS;
						end
					end else begin
						acc       <= sum;
						in_idx    <= in_idx + 1;
						state     <= L3_MAC;
					end
				end

				// ---------- Head H1: 32 -> 1 ----------
				H1_BIAS: begin
					acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
					in_idx    <= 0;
					state     <= H1_MAC;
				end

				H1_MAC: begin
					sum = acc + mul_q8_8_to_q16_16(weight, hidden3[in_idx]);

					if (in_idx == 8'd31) begin
						h1_reg    <= q8_8_from_acc(sum);
						$display("H1 output for model %0d vec current: %0d (sum=%0d, weight=%0d, hidden3[%0d]=%0d)", model_select, q8_8_from_acc(sum), sum, weight, in_idx, hidden3[in_idx]);
						state     <= H4_BIAS;
					end else begin
						acc       <= sum;
						in_idx    <= in_idx + 1;
						state     <= H1_MAC;
					end
				end

				// ---------- Head H4: 32 -> 1 ----------
				H4_BIAS: begin
					acc       <= $signed({{32{weight[15]}}, weight}) <<< 8;
					in_idx    <= 0;
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
						state     <= H4_MAC;
					end
				end

				DONE: begin
					done_reg <= 1'b1;
					$display("DONE model %0d h1=%0d h4=%0d", model_select, h1_reg, h4_reg);
					state    <= IDLE;
				end

				default: state <= IDLE;
			endcase
		end
	end

endmodule
