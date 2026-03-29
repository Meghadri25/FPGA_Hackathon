`timescale 1ns / 1ps

`include "blk_mem_gen_models.sv"

module nn_tb;

    logic clk;
    logic rst;
    logic [2:0] model_select;
    logic start_load;
    logic feature_valid;
    logic signed [15:0] feature_in;
    logic feature_ready;

    logic signed [15:0] features [0:23];

    logic signed [15:0] h1_out;
    logic signed [15:0] h4_out;
    logic done;

    logic [3:0] core_state_out;
    logic [4:0] load_count_out;

    nn_combined dut (
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

    always #5 clk = ~clk;

    logic signed [15:0] test_vectors [0:3][0:23];
    logic signed [15:0] out_h1 [0:4][0:3];
    logic signed [15:0] out_h4 [0:4][0:3];

    logic signed [15:0] expected_h1 [0:4][0:3];
    logic signed [15:0] expected_h4 [0:4][0:3];

    int ii;
    int model_idx;
    int vec_idx;
    int cycle_count;
    bit tests_ok;

    task automatic set_features(input int vec);
        begin
            for (ii = 0; ii < 24; ii++) begin
                features[ii] = test_vectors[vec][ii];
            end
        end
    endtask

    task automatic push_features_serial;
        begin
            for (ii = 0; ii < 24; ii++) begin
                @(posedge clk);
                feature_valid <= 1'b1;
                feature_in    <= features[ii];

                while (feature_ready !== 1'b1) begin
                    @(posedge clk);
                end
            end

            @(posedge clk);
            feature_valid <= 1'b0;
            feature_in    <= '0;
        end
    endtask

    initial begin
        clk = 0;
        rst = 1;
        start_load = 0;
        feature_valid = 0;
        feature_in = '0;
        model_select = 0;
        tests_ok = 1;

        for (ii = 0; ii < 24; ii++) begin
            features[ii] = '0;
        end

        // Expected outputs
        expected_h1[0] = '{16'd16749, 16'd1279, 16'd14194, 16'd25969};
        expected_h1[1] = '{16'd13687, 16'd1043, 16'd11300, 16'd18407};
        expected_h1[2] = '{16'd13774, 16'd905,  16'd11747, 16'd20742};
        expected_h1[3] = '{16'd3972,  16'd37,   16'd3991,  16'd7287};
        expected_h1[4] = '{16'd17391, 16'd1492, 16'd12697, 16'd21160};

        expected_h4[0] = '{16'd14145, 16'd1131, 16'd12383, 16'd21965};
        expected_h4[1] = '{16'd13697, 16'd1006, 16'd11162, 16'd19315};
        expected_h4[2] = '{16'd13995, 16'd963,  16'd12052, 16'd21064};
        expected_h4[3] = '{16'd6177,  16'd353,  16'd6001,  16'd10974};
        expected_h4[4] = '{16'd17360, 16'd1467, 16'd12762, 16'd21787};

        // vector 0
        test_vectors[0] = '{
            16'd17278, 16'd17222, 16'd18900, 16'd21472,
            16'd20689, 16'd21863, 16'd22255, 16'd21025,
            16'd21640, 16'd22926, 16'd21528, 16'd24044,
            16'd23317, 16'd21472, 16'd23205, 16'd25442,
            16'd18668, 16'd22761, 16'd7306,  16'd12143,
            16'd32209, 16'd3245,  16'd26277, 16'd32767
        };

        // vector 1
        test_vectors[1] = '{
            16'd1000, 16'd1200, 16'd1100, 16'd900,
            16'd1400, 16'd1300, 16'd1250, 16'd1350,
            16'd1450, 16'd1550, 16'd1600, 16'd1650,
            16'd1700, 16'd1800, 16'd1900, 16'd2000,
            16'd2100, 16'd2200, 16'd2300, 16'd2400,
            16'd2500, 16'd2600, 16'd2700, 16'd2800
        };

        // vector 2
        for (ii = 0; ii < 24; ii++) begin
            test_vectors[2][ii] = 16'd16000;
        end

        // vector 3
        for (ii = 0; ii < 24; ii++) begin
            test_vectors[3][ii] = 16'd30000;
        end

        repeat (5) @(posedge clk);
        rst = 0;
    end

    initial begin
        wait(rst == 0);
        @(posedge clk);

        for (model_idx = 0; model_idx < 5; model_idx++) begin
            model_select = model_idx[2:0];

            for (vec_idx = 0; vec_idx < 4; vec_idx++) begin
                set_features(vec_idx);

                @(posedge clk);
                start_load = 1'b1;
                @(posedge clk);
                start_load = 1'b0;

                push_features_serial();

                cycle_count = 0;
                @(posedge clk); // allow state machine to enter run
                $display("model=%0d vec=%0d load complete, core_state=%0d", model_idx, vec_idx, core_state_out);

                while ((done !== 1'b1) && (cycle_count < 20000)) begin
                    @(posedge clk);
                    cycle_count++;
                    if (cycle_count % 5000 == 0) begin
                        $display("still running model=%0d vec=%0d cycle=%0d core_state=%0d", model_idx, vec_idx, cycle_count, core_state_out);
                    end
                end

                if (done !== 1'b1) begin
                    $display("ERROR: model %0d vector %0d timeout", model_idx, vec_idx);
                    tests_ok = 0;
                end else begin
                    out_h1[model_idx][vec_idx] = h1_out;
                    out_h4[model_idx][vec_idx] = h4_out;

                    if (h1_out !== expected_h1[model_idx][vec_idx] ||
                        h4_out !== expected_h4[model_idx][vec_idx]) begin
                        $display("ERROR: model %0d vec %0d mismatch | exp h1=%0d h4=%0d | got h1=%0d h4=%0d",
                            model_idx, vec_idx,
                            expected_h1[model_idx][vec_idx],
                            expected_h4[model_idx][vec_idx],
                            h1_out, h4_out);
                        tests_ok = 0;
                    end

                    $display("Model %0d Vector %0d : h1=%0d h4=%0d",
                        model_idx, vec_idx, h1_out, h4_out);
                end

                wait(done == 1'b0);
                #10;
            end
        end

        if (tests_ok)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;
    end

endmodule