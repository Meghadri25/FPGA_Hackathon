`timescale 1ns / 1ps

module nn_axi_wrapper_tb;

    localparam integer C_S_AXI_DATA_WIDTH = 32;
    localparam integer C_S_AXI_ADDR_WIDTH = 6;

    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_CONTROL      = 6'h00;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_MODEL_SELECT = 6'h04;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_FEATURE_IN   = 6'h08;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_FEAT_COUNT   = 6'h0C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_FIFO_LEVEL   = 6'h10;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_H1_OUT       = 6'h14;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] REG_H4_OUT       = 6'h18;

    logic s_axi_aclk;
    logic s_axi_aresetn;

    logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr;
    logic [2:0]                    s_axi_awprot;
    logic                          s_axi_awvalid;
    logic                          s_axi_awready;
    logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata;
    logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb;
    logic                          s_axi_wvalid;
    logic                          s_axi_wready;
    logic [1:0]                    s_axi_bresp;
    logic                          s_axi_bvalid;
    logic                          s_axi_bready;
    logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr;
    logic [2:0]                    s_axi_arprot;
    logic                          s_axi_arvalid;
    logic                          s_axi_arready;
    logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata;
    logic [1:0]                    s_axi_rresp;
    logic                          s_axi_rvalid;
    logic                          s_axi_rready;

    integer i;
    integer timeout_cycles;
    reg [31:0] rd;

    nn_axi_wrapper #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .FIFO_DEPTH(64),
        .DEFAULT_FEATURE_COUNT(24)
    ) dut (
        .s_axi_aclk(s_axi_aclk),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready)
    );

    always #5 s_axi_aclk = ~s_axi_aclk;

    task automatic axi_write(input [C_S_AXI_ADDR_WIDTH-1:0] addr, input [31:0] data);
    begin
        @(posedge s_axi_aclk);
        s_axi_awaddr  <= addr;
        s_axi_awprot  <= 3'b000;
        s_axi_awvalid <= 1'b1;
        s_axi_wdata   <= data;
        s_axi_wstrb   <= 4'hF;
        s_axi_wvalid  <= 1'b1;
        s_axi_bready  <= 1'b1;

        while (!(s_axi_awready && s_axi_wready)) begin
            @(posedge s_axi_aclk);
        end

        @(posedge s_axi_aclk);
        s_axi_awvalid <= 1'b0;
        s_axi_wvalid  <= 1'b0;

        while (!s_axi_bvalid) begin
            @(posedge s_axi_aclk);
        end

        @(posedge s_axi_aclk);
        s_axi_bready <= 1'b0;
    end
    endtask

    task automatic axi_read(input [C_S_AXI_ADDR_WIDTH-1:0] addr, output [31:0] data);
    begin
        @(posedge s_axi_aclk);
        s_axi_araddr  <= addr;
        s_axi_arprot  <= 3'b000;
        s_axi_arvalid <= 1'b1;
        s_axi_rready  <= 1'b1;

        while (!s_axi_arready) begin
            @(posedge s_axi_aclk);
        end

        @(posedge s_axi_aclk);
        s_axi_arvalid <= 1'b0;

        while (!s_axi_rvalid) begin
            @(posedge s_axi_aclk);
        end

        data = s_axi_rdata;

        @(posedge s_axi_aclk);
        s_axi_rready <= 1'b0;
    end
    endtask

    initial begin
        s_axi_aclk    = 1'b0;
        s_axi_aresetn = 1'b0;

        s_axi_awaddr  = '0;
        s_axi_awprot  = 3'b000;
        s_axi_awvalid = 1'b0;
        s_axi_wdata   = '0;
        s_axi_wstrb   = 4'hF;
        s_axi_wvalid  = 1'b0;
        s_axi_bready  = 1'b0;
        s_axi_araddr  = '0;
        s_axi_arprot  = 3'b000;
        s_axi_arvalid = 1'b0;
        s_axi_rready  = 1'b0;

        repeat (10) @(posedge s_axi_aclk);
        s_axi_aresetn = 1'b1;
        repeat (5) @(posedge s_axi_aclk);

        // Clear done/errors/fifo
        axi_write(REG_CONTROL, 32'h0000_000E);

        // Configure model and feature count
        axi_write(REG_MODEL_SELECT, 32'd0);
        axi_write(REG_FEAT_COUNT, 32'd24);

        // Push 24 test features
        for (i = 0; i < 24; i = i + 1) begin
            axi_write(REG_FEATURE_IN, i + 32'd1000);
        end

        // Check FIFO level
        axi_read(REG_FIFO_LEVEL, rd);
        $display("FIFO level before start = %0d", rd[15:0]);
        if (rd[15:0] < 16'd24) begin
            $fatal(1, "FIFO level too low before start");
        end

        // Start computation
        axi_write(REG_CONTROL, 32'h0000_0001);

        // Poll done_sticky (control bit 2)
        timeout_cycles = 0;
        while (1'b1) begin
            axi_read(REG_CONTROL, rd);
            if (rd[2]) begin
                $display("Done detected. STATUS=0x%08h", rd);
                break;
            end
            timeout_cycles = timeout_cycles + 1;
            if (timeout_cycles > 10000) begin
                $fatal(1, "Timeout waiting for done");
            end
        end

        // Error bits should be clear
        if (rd[7] || rd[6] || rd[5]) begin
            $fatal(1, "Wrapper error bits set. STATUS=0x%08h", rd);
        end

        // Read outputs
        axi_read(REG_H1_OUT, rd);
        $display("H1 raw=0x%08h signed16=%0d", rd, $signed(rd[15:0]));
        axi_read(REG_H4_OUT, rd);
        $display("H4 raw=0x%08h signed16=%0d", rd, $signed(rd[15:0]));

        // Clear done sticky
        axi_write(REG_CONTROL, 32'h0000_0002);
        axi_read(REG_CONTROL, rd);
        if (rd[2] != 1'b0) begin
            $fatal(1, "done_sticky did not clear");
        end

        // Negative test: start without enough data should set invalid_start_err (bit 7)
        axi_write(REG_CONTROL, 32'h0000_000E); // clear done/errors/fifo
        axi_write(REG_FEAT_COUNT, 32'd24);
        axi_write(REG_CONTROL, 32'h0000_0001); // start
        axi_read(REG_CONTROL, rd);
        if (rd[7] != 1'b1) begin
            $fatal(1, "Expected invalid_start_err to be set");
        end

        $display("nn_axi_wrapper_tb PASSED");
        $finish;
    end

endmodule
