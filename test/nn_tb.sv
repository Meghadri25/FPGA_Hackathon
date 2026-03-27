`timescale 1ns / 1ps

module nn_tb;

// Test bench for nn_combined module

reg clk;
reg rst;
reg [2:0] model_select;
reg start;
reg signed [15:0] features [0:23];
wire signed [15:0] h1_out;
wire signed [15:0] h4_out;
wire done;

// Instantiate the module
nn_combined dut (
    .clk(clk),
    .rst(rst),
    .model_select(model_select),
    .start(start),
    .features(features),
    .h1_out(h1_out),
    .h4_out(h4_out),
    .done(done)
);

// Clock generation
always #5 clk = ~clk;  // 100MHz clock

// Several test input vectors (Q8.8 fixed-point) for broader coverage
reg signed [15:0] test_vectors [0:3][0:23];
reg signed [15:0] out_h1 [0:4][0:3];
reg signed [15:0] out_h4 [0:4][0:3];

initial begin
    // vector 0: sample from data
    test_vectors[0][0]  = 16'd17278; test_vectors[0][1]  = 16'd17222; test_vectors[0][2]  = 16'd18900; test_vectors[0][3]  = 16'd21472;
    test_vectors[0][4]  = 16'd20689; test_vectors[0][5]  = 16'd21863; test_vectors[0][6]  = 16'd22255; test_vectors[0][7]  = 16'd21025;
    test_vectors[0][8]  = 16'd21640; test_vectors[0][9]  = 16'd22926; test_vectors[0][10] = 16'd21528; test_vectors[0][11] = 16'd24044;
    test_vectors[0][12] = 16'd23317; test_vectors[0][13] = 16'd21472; test_vectors[0][14] = 16'd23205; test_vectors[0][15] = 16'd25442;
    test_vectors[0][16] = 16'd18668; test_vectors[0][17] = 16'd22761; test_vectors[0][18] = 16'd7306;  test_vectors[0][19] = 16'd12143;
    test_vectors[0][20] = 16'd32209; test_vectors[0][21] = 16'd3245;  test_vectors[0][22] = 16'd26277; test_vectors[0][23] = 16'd32767;

    // vector 1: relatively low values
    test_vectors[1][0]  = 16'd1000;  test_vectors[1][1]  = 16'd1200;  test_vectors[1][2]  = 16'd1100;  test_vectors[1][3]  = 16'd900;
    test_vectors[1][4]  = 16'd1400;  test_vectors[1][5]  = 16'd1300;  test_vectors[1][6]  = 16'd1250;  test_vectors[1][7]  = 16'd1350;
    test_vectors[1][8]  = 16'd1450;  test_vectors[1][9]  = 16'd1550;  test_vectors[1][10] = 16'd1600;  test_vectors[1][11] = 16'd1650;
    test_vectors[1][12] = 16'd1700;  test_vectors[1][13] = 16'd1800;  test_vectors[1][14] = 16'd1900;  test_vectors[1][15] = 16'd2000;
    test_vectors[1][16] = 16'd2100;  test_vectors[1][17] = 16'd2200;  test_vectors[1][18] = 16'd2300;  test_vectors[1][19] = 16'd2400;
    test_vectors[1][20] = 16'd2500;  test_vectors[1][21] = 16'd2600;  test_vectors[1][22] = 16'd2700;  test_vectors[1][23] = 16'd2800;

    // vector 2: mid-range constant value
    integer ii;
    for (ii = 0; ii < 24; ii = ii + 1) begin
        test_vectors[2][ii] = 16'd16000;
    end

    // vector 3: high values near upper tied to Q8.8 range
    for (ii = 0; ii < 24; ii = ii + 1) begin
        test_vectors[3][ii] = 16'd30000;
    end
end

initial begin
    // Initialize
    clk = 0;
    rst = 1;
    start = 0;

    integer model_idx;
    integer vec_idx;
    reg [31:0] cycle_count;
    reg tests_ok;

    tests_ok = 1;
    #10 rst = 0;

    for (model_idx = 0; model_idx < 5; model_idx = model_idx + 1) begin
        model_select = model_idx;

        for (vec_idx = 0; vec_idx < 4; vec_idx = vec_idx + 1) begin
            // Load vector
            for (ii = 0; ii < 24; ii = ii + 1) begin
                features[ii] = test_vectors[vec_idx][ii];
            end

            // Start inference
            start = 1;
            #10 start = 0;

            // Wait for done with timeout
            cycle_count = 0;
            while (!done && cycle_count < 5000) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end

            if (!done) begin
                $display("ERROR: model %0d vector %0d did not finish (timeout)", model_idx, vec_idx);
                tests_ok = 0;
                disable all_done;
            end

            // Save outputs
            out_h1[model_idx][vec_idx] = h1_out;
            out_h4[model_idx][vec_idx] = h4_out;

            // Basic bounds checks
            if (h1_out < -32768 || h1_out > 32767 || h4_out < -32768 || h4_out > 32767) begin
                $display("ERROR: output out-of-range model %0d vector %0d h1=%0d h4=%0d", model_idx, vec_idx, h1_out, h4_out);
                tests_ok = 0;
            end

            $display("Model %0d Vector %0d : h1=%0d h4=%0d", model_idx, vec_idx, h1_out, h4_out);

            // Ensure output varies with input patterns
            if (vec_idx > 0 && h1_out == out_h1[model_idx][0] && h4_out == out_h4[model_idx][0]) begin
                $display("WARNING: model %0d output unchanged for vector %0d", model_idx, vec_idx);
            end

            // small pause before next run
            #10;
        end
    end

    // compare across models for vector 0
    if (out_h1[0][0] == out_h1[1][0] && out_h1[1][0] == out_h1[2][0] && out_h1[2][0] == out_h1[3][0] && out_h1[3][0] == out_h1[4][0]) begin
        $display("WARNING: h1 outputs identical for all models on vector 0");
    end

    if (out_h4[0][0] == out_h4[1][0] && out_h4[1][0] == out_h4[2][0] && out_h4[2][0] == out_h4[3][0] && out_h4[3][0] == out_h4[4][0]) begin
        $display("WARNING: h4 outputs identical for all models on vector 0");
    end

    if (tests_ok) begin
        $display("ALL TESTS PASSED");
    end else begin
        $display("SOME TESTS FAILED");
    end

    ::all_done:;
    $finish;
end

endmodule