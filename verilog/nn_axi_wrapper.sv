`timescale 1ns / 1ps

module nn_axi_wrapper #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6,
    parameter integer FIFO_DEPTH = 512,
    parameter integer DEFAULT_FEATURE_COUNT = 24
) (
    input  wire                                  s_axi_aclk,
    input  wire                                  s_axi_aresetn,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]         s_axi_awaddr,
    input  wire [2:0]                            s_axi_awprot,
    input  wire                                  s_axi_awvalid,
    output reg                                   s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]         s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]     s_axi_wstrb,
    input  wire                                  s_axi_wvalid,
    output reg                                   s_axi_wready,
    output reg [1:0]                             s_axi_bresp,
    output reg                                   s_axi_bvalid,
    input  wire                                  s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]         s_axi_araddr,
    input  wire [2:0]                            s_axi_arprot,
    input  wire                                  s_axi_arvalid,
    output reg                                   s_axi_arready,
    output reg [C_S_AXI_DATA_WIDTH-1:0]          s_axi_rdata,
    output reg [1:0]                             s_axi_rresp,
    output reg                                   s_axi_rvalid,
    input  wire                                  s_axi_rready
);

    localparam integer ADDR_LSB = 2;
    localparam integer PTR_W = (FIFO_DEPTH <= 2) ? 1 : $clog2(FIFO_DEPTH);

    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_CONTROL      = 6'h00;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_MODEL_SELECT = 6'h04;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_FEATURE_IN   = 6'h08;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_FEAT_COUNT   = 6'h0C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_FIFO_LEVEL   = 6'h10;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_H1_OUT       = 6'h14;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_H4_OUT       = 6'h18;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_CORE_DEBUG   = 6'h1C;

    localparam [2:0] ST_IDLE      = 3'd0;
    localparam [2:0] ST_START     = 3'd1;
    localparam [2:0] ST_STREAM    = 3'd2;
    localparam [2:0] ST_WAIT_DONE = 3'd3;

    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg aw_en;

    wire slv_reg_wren;
    wire slv_reg_rden;

    reg [2:0] model_select_reg;
    reg [15:0] feature_count_reg;

    reg done_sticky;
    reg busy;
    reg overflow_err;
    reg underflow_err;
    reg invalid_start_err;

    reg signed [15:0] h1_out_reg;
    reg signed [15:0] h4_out_reg;

    reg [2:0] stream_state;
    reg [15:0] stream_remaining;

    reg signed [15:0] fifo_mem [0:FIFO_DEPTH-1];
    reg [PTR_W-1:0] fifo_wr_ptr;
    reg [PTR_W-1:0] fifo_rd_ptr;
    reg [15:0] fifo_count;

    wire fifo_full;
    wire fifo_empty;

    reg nn_start_load;
    reg nn_feature_valid;
    reg signed [15:0] nn_feature_in;
    wire nn_feature_ready;
    wire signed [15:0] nn_h1_out;
    wire signed [15:0] nn_h4_out;
    wire nn_done;
    wire [3:0] nn_core_state;
    wire [4:0] nn_load_count;

    reg [C_S_AXI_DATA_WIDTH-1:0] reg_data_out;

    assign fifo_full = (fifo_count == FIFO_DEPTH);
    assign fifo_empty = (fifo_count == 0);

    assign slv_reg_wren = s_axi_wready && s_axi_wvalid && s_axi_awready && s_axi_awvalid;
    assign slv_reg_rden = s_axi_arready && s_axi_arvalid && ~s_axi_rvalid;

    nn_combined u_nn_combined (
        .clk(s_axi_aclk),
        .rst(~s_axi_aresetn),
        .model_select(model_select_reg),
        .start_load(nn_start_load),
        .feature_valid(nn_feature_valid),
        .feature_in(nn_feature_in),
        .feature_ready(nn_feature_ready),
        .h1_out(nn_h1_out),
        .h4_out(nn_h4_out),
        .done(nn_done),
        .core_state_out(nn_core_state),
        .load_count_out(nn_load_count)
    );

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            s_axi_awready <= 1'b0;
            aw_en <= 1'b1;
        end else begin
            if (~s_axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                s_axi_awready <= 1'b1;
                aw_en <= 1'b0;
            end else if (s_axi_bready && s_axi_bvalid) begin
                aw_en <= 1'b1;
                s_axi_awready <= 1'b0;
            end else begin
                s_axi_awready <= 1'b0;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            axi_awaddr <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (~s_axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                axi_awaddr <= s_axi_awaddr;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            s_axi_wready <= 1'b0;
        end else begin
            if (~s_axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en) begin
                s_axi_wready <= 1'b1;
            end else begin
                s_axi_wready <= 1'b0;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp <= 2'b00;
        end else begin
            if (s_axi_awready && s_axi_awvalid && ~s_axi_bvalid && s_axi_wready && s_axi_wvalid) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp <= 2'b00;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            s_axi_arready <= 1'b0;
            axi_araddr <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (~s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
                axi_araddr <= s_axi_araddr;
            end else begin
                s_axi_arready <= 1'b0;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rresp <= 2'b00;
        end else begin
            if (slv_reg_rden) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp <= 2'b00;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    always @(*) begin
        case (axi_araddr)
            REG_CONTROL: begin
                reg_data_out = 32'd0;
                reg_data_out[0] = (stream_state == ST_IDLE) && (~busy);
                reg_data_out[1] = busy;
                reg_data_out[2] = done_sticky;
                reg_data_out[3] = fifo_full;
                reg_data_out[4] = fifo_empty;
                reg_data_out[5] = overflow_err;
                reg_data_out[6] = underflow_err;
                reg_data_out[7] = invalid_start_err;
                reg_data_out[10:8] = stream_state;
            end
            REG_MODEL_SELECT: reg_data_out = {29'd0, model_select_reg};
            REG_FEATURE_IN:   reg_data_out = 32'd0;
            REG_FEAT_COUNT:   reg_data_out = {16'd0, feature_count_reg};
            REG_FIFO_LEVEL:   reg_data_out = {16'd0, fifo_count};
            REG_H1_OUT:       reg_data_out = {{16{h1_out_reg[15]}}, h1_out_reg};
            REG_H4_OUT:       reg_data_out = {{16{h4_out_reg[15]}}, h4_out_reg};
            REG_CORE_DEBUG:   reg_data_out = {23'd0, nn_load_count, nn_core_state};
            default:          reg_data_out = 32'd0;
        endcase
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            s_axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}};
        end else begin
            if (slv_reg_rden) begin
                s_axi_rdata <= reg_data_out;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (~s_axi_aresetn) begin
            model_select_reg <= 3'd0;
            feature_count_reg <= DEFAULT_FEATURE_COUNT[15:0];

            done_sticky <= 1'b0;
            busy <= 1'b0;
            overflow_err <= 1'b0;
            underflow_err <= 1'b0;
            invalid_start_err <= 1'b0;

            h1_out_reg <= 16'sd0;
            h4_out_reg <= 16'sd0;

            stream_state <= ST_IDLE;
            stream_remaining <= 16'd0;

            fifo_wr_ptr <= {PTR_W{1'b0}};
            fifo_rd_ptr <= {PTR_W{1'b0}};
            fifo_count <= 16'd0;

            nn_start_load <= 1'b0;
            nn_feature_valid <= 1'b0;
            nn_feature_in <= 16'sd0;
        end else begin
            reg start_cmd;
            reg clear_done_cmd;
            reg clear_err_cmd;
            reg clear_fifo_cmd;
            reg feature_push_cmd;
            reg signed [15:0] feature_push_data;
            reg stream_xfer;

            start_cmd = 1'b0;
            clear_done_cmd = 1'b0;
            clear_err_cmd = 1'b0;
            clear_fifo_cmd = 1'b0;
            feature_push_cmd = 1'b0;
            feature_push_data = 16'sd0;
            stream_xfer = 1'b0;

            nn_start_load <= 1'b0;
            nn_feature_valid <= 1'b0;

            if (slv_reg_wren) begin
                case (axi_awaddr)
                    REG_CONTROL: begin
                        start_cmd = s_axi_wdata[0];
                        clear_done_cmd = s_axi_wdata[1];
                        clear_err_cmd = s_axi_wdata[2];
                        clear_fifo_cmd = s_axi_wdata[3];
                    end
                    REG_MODEL_SELECT: begin
                        if (s_axi_wstrb[0]) begin
                            model_select_reg <= s_axi_wdata[2:0];
                        end
                    end
                    REG_FEATURE_IN: begin
                        feature_push_cmd = 1'b1;
                        feature_push_data = s_axi_wdata[15:0];
                    end
                    REG_FEAT_COUNT: begin
                        if (s_axi_wstrb[0] || s_axi_wstrb[1]) begin
                            feature_count_reg <= s_axi_wdata[15:0];
                        end
                    end
                    default: begin
                    end
                endcase
            end

            if (clear_done_cmd) begin
                done_sticky <= 1'b0;
            end
            if (clear_err_cmd) begin
                overflow_err <= 1'b0;
                underflow_err <= 1'b0;
                invalid_start_err <= 1'b0;
            end
            if (clear_fifo_cmd && ~busy) begin
                fifo_wr_ptr <= {PTR_W{1'b0}};
                fifo_rd_ptr <= {PTR_W{1'b0}};
                fifo_count <= 16'd0;
            end

            if (feature_push_cmd) begin
                if (~fifo_full) begin
                    fifo_mem[fifo_wr_ptr] <= feature_push_data;
                    fifo_wr_ptr <= fifo_wr_ptr + 1'b1;
                    fifo_count <= fifo_count + 1'b1;
                end else begin
                    overflow_err <= 1'b1;
                end
            end

            case (stream_state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start_cmd) begin
                        if ((feature_count_reg != 16'd0) && (fifo_count >= feature_count_reg)) begin
                            busy <= 1'b1;
                            done_sticky <= 1'b0;
                            stream_remaining <= feature_count_reg;
                            stream_state <= ST_START;
                        end else begin
                            invalid_start_err <= 1'b1;
                        end
                    end
                end

                ST_START: begin
                    nn_start_load <= 1'b1;
                    stream_state <= ST_STREAM;
                end

                ST_STREAM: begin
                    if (stream_remaining != 16'd0) begin
                        if (nn_feature_ready) begin
                            if (~fifo_empty) begin
                                nn_feature_valid <= 1'b1;
                                nn_feature_in <= fifo_mem[fifo_rd_ptr];
                                stream_xfer = 1'b1;
                            end else begin
                                underflow_err <= 1'b1;
                                busy <= 1'b0;
                                stream_state <= ST_IDLE;
                            end
                        end
                    end else begin
                        stream_state <= ST_WAIT_DONE;
                    end

                    if (stream_xfer) begin
                        fifo_rd_ptr <= fifo_rd_ptr + 1'b1;
                        fifo_count <= fifo_count - 1'b1;

                        if (stream_remaining == 16'd1) begin
                            stream_remaining <= 16'd0;
                            stream_state <= ST_WAIT_DONE;
                        end else begin
                            stream_remaining <= stream_remaining - 1'b1;
                        end
                    end
                end

                ST_WAIT_DONE: begin
                    if (nn_done) begin
                        h1_out_reg <= nn_h1_out;
                        h4_out_reg <= nn_h4_out;
                        done_sticky <= 1'b1;
                        busy <= 1'b0;
                        stream_state <= ST_IDLE;
                    end
                end

                default: begin
                    stream_state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
