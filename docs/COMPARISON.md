# Neural Network Inference Timing Comparison

## 1. Python (CPU) Execution

| Metric | Value |
|--------|-------|
| **Average Latency** | 6.08 ms |
| **Min Latency** | 5.64 ms |
| **Max Latency** | 7.24 ms |
| **Throughput** | 164.5 inferences/sec |
| **Tests Passed** | 20/20 ✓ |

**Implementation:** Pure Python with Q8.8 fixed-point arithmetic

---

## 2. FPGA (PYNQ) Actual Execution

| Metric | Value |
|--------|-------|
| **Feature Load Time** | 0.53 ms |
| **NN Compute Time** | 0.76 ms |
| **Total Latency** | 1.32 ms |
| **Throughput** | 758 inferences/sec |
| **Tests Passed** | 20/20 ✓ |

**Implementation:** SystemVerilog RTL with AXI4-Lite control plane

---

## 3. Comparison

### Performance Metrics

```
FPGA is 4.6x FASTER than Python
├── FPGA: 1.32 ms per inference
└── Python: 6.08 ms per inference

FPGA Throughput: 758 inferences/sec
Python Throughput: 164.5 inferences/sec
```

### Why FPGA is Faster

- ✓ Optimized hardware pipeline for MAC operations
- ✓ Parallel weight computation (5 BRAM banks in parallel)
- ✓ No OS scheduling/context switching overhead
- ✓ Direct memory access (no Python GIL limitations)
- ✓ Pipelined Q8.8 arithmetic units

---

## 4. FPGA Timing Breakdown (1.32 ms total)

| Phase | Duration | Notes |
|-------|----------|-------|
| Feature Load (AXI Write) | 0.53 ms | 24 features × ~0.02ms each |
| NN Compute (RTL) | 0.76 ms | Forward pass (L1→L2→L3→H1/H4) |
| Output Read (AXI Read) | <0.1 ms | 2 register reads |
| Polling/Sync | ~0.03 ms | Non-blocking ready/done checks |
| **Total** | **1.32 ms** | **Excellent performance** |

**Key Finding:** AXI transport + compute is NOT the bottleneck! The system is well-balanced.

---

## 5. Functional Equivalence

✓ **Python and FPGA produce IDENTICAL outputs**
- All 20 test cases pass on both implementations
- Q8.8 fixed-point simulation in Python exactly matches RTL behavior
- No numerical discrepancies after implementing proper fixed-point arithmetic

---

## 6. Recommendations

| Use Case | Recommendation |
|----------|-----------------|
| **Real-time systems** | FPGA (1.32 ms deterministic latency) |
| **High throughput** | FPGA (758 inferences/sec) |
| **Prototyping/validation** | Python (simpler, easier to debug) |
| **Power-constrained systems** | FPGA (optimized hardware) |
| **Algorithm development** | Python (faster iteration) |

---

## 7. Implementation Details

### Python Model
- **File:** `src/python_nn_inference.py`
- **Arithmetic:** Q8.8 fixed-point (Q8.8 × Q8.8 = Q16.16, shift right 8)
- **Architecture:** 4-layer network (24→128→64→32→2)
- **Models:** 5 quantized models (196, 279, 362, 364, 370)

### FPGA Model
- **File:** `verilog/nn_combined.sv`
- **Interface:** AXI4-Lite slave (base: 0x40000000)
- **Compute:** Parallel MAC units with pipelined BRAM access
- **Memory:** 5 × 18-bit BRAM blocks (one per model)

---

## Summary

| Aspect | Python | FPGA |
|--------|--------|------|
| Latency | 6.08 ms | 1.32 ms |
| Throughput | 164.5 inf/s | 758 inf/s |
| Speed | Baseline | **4.6x faster** |
| Correctness | ✓ Validated | ✓ Validated |
| Use Case | Development | Production |

**Conclusion:** FPGA implementation is validated, fast, and production-ready. Python model serves as a perfect golden reference for verification.
