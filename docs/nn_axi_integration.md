# nn_axi_wrapper Integration Guide (PYNQ-Z2)

This guide shows how to control `nn_combined` from the PS using AXI4-Lite through `nn_axi_wrapper`.

## 1) Register Map

Base address: assigned by Vivado Address Editor.

- `0x00` CONTROL_STATUS (RW)
- `0x04` MODEL_SELECT (RW)
- `0x08` FEATURE_IN (W)
- `0x0C` FEATURE_COUNT (RW)
- `0x10` FIFO_LEVEL (R)
- `0x14` H1_OUT (R)
- `0x18` H4_OUT (R)
- `0x1C` CORE_DEBUG (R)

### CONTROL_STATUS (`0x00`)

Read bits:
- bit 0: `ready_to_start` (1 when wrapper is idle)
- bit 1: `busy`
- bit 2: `done_sticky`
- bit 3: `fifo_full`
- bit 4: `fifo_empty`
- bit 5: `overflow_err`
- bit 6: `underflow_err`
- bit 7: `invalid_start_err`
- bits 10:8: wrapper FSM state

Write bits (write-1 command):
- bit 0: `start`
- bit 1: `clear_done`
- bit 2: `clear_errors`
- bit 3: `clear_fifo` (only effective when not busy)

### MODEL_SELECT (`0x04`)
- bits 2:0: model ID 0..4

### FEATURE_IN (`0x08`)
- write signed 16-bit feature in bits 15:0
- each write pushes one item into wrapper FIFO

### FEATURE_COUNT (`0x0C`)
- number of features to stream to `nn_combined` on start
- default set in RTL (`DEFAULT_FEATURE_COUNT`, default 24)

### FIFO_LEVEL (`0x10`)
- bits 15:0: current FIFO count

### H1_OUT / H4_OUT (`0x14`, `0x18`)
- signed 16-bit result in bits 15:0 (sign-extended to 32-bit)

### CORE_DEBUG (`0x1C`)
- bits 8:5 : `nn_combined.load_count_out`
- bits 3:0 : `nn_combined.core_state_out`

## 2) Vivado Block Design (Simple)

1. Create block design.
2. Add Zynq7 Processing System and run Block Automation.
3. Add your packaged custom IP containing `nn_axi_wrapper`.
4. Connect:
- `S_AXI` of wrapper to `M_AXI_GP0` of PS via AXI interconnect (or SmartConnect)
- `s_axi_aclk` to `FCLK_CLK0`
- `s_axi_aresetn` to `peripheral_aresetn` from Processor System Reset
5. Assign address in Address Editor.
6. Generate output products, create HDL wrapper, synth/impl/bitstream.

Notes:
- `nn_axi_wrapper` internally instantiates `nn_combined` and drives its valid/ready streaming signals.
- No AXI DMA is required for this first version.

## 3) PYNQ Python Example

```python
from pynq import Overlay, MMIO
import time

# Load overlay
ol = Overlay("design_1.bit")

# Use your actual address map values from ol.ip_dict
base = ol.ip_dict["nn_axi_wrapper_0"]["phys_addr"]
range_bytes = ol.ip_dict["nn_axi_wrapper_0"]["addr_range"]
mmio = MMIO(base, range_bytes)

REG_CONTROL      = 0x00
REG_MODEL_SELECT = 0x04
REG_FEATURE_IN   = 0x08
REG_FEATURE_COUNT= 0x0C
REG_FIFO_LEVEL   = 0x10
REG_H1_OUT       = 0x14
REG_H4_OUT       = 0x18


def to_u16(x):
    return x & 0xFFFF


def from_s16(x):
    x = x & 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def write_features(features):
    for f in features:
        mmio.write(REG_FEATURE_IN, to_u16(int(f)))


def run_nn(features, model_select=0, timeout_s=1.0):
    # Optional maintenance
    mmio.write(REG_CONTROL, (1 << 1) | (1 << 2) | (1 << 3))  # clear_done, clear_errors, clear_fifo

    mmio.write(REG_MODEL_SELECT, model_select & 0x7)
    mmio.write(REG_FEATURE_COUNT, len(features) & 0xFFFF)

    write_features(features)

    # Start
    mmio.write(REG_CONTROL, 1 << 0)

    # Poll done_sticky (bit 2)
    t0 = time.time()
    while True:
        status = mmio.read(REG_CONTROL)
        done = (status >> 2) & 0x1
        if done:
            break
        if (time.time() - t0) > timeout_s:
            raise TimeoutError("NN inference timeout")

    h1_raw = mmio.read(REG_H1_OUT)
    h4_raw = mmio.read(REG_H4_OUT)

    h1 = from_s16(h1_raw)
    h4 = from_s16(h4_raw)

    # Clear done for next run
    mmio.write(REG_CONTROL, 1 << 1)

    return h1, h4


# Example
features = [1000] * 24
h1, h4 = run_nn(features, model_select=0)
print("h1:", h1, "h4:", h4)
```

## 4) Practical Notes

- Keep `FEATURE_COUNT` equal to how many features you write into FIFO before start.
- If `invalid_start_err` is set, start was requested without enough FIFO data.
- If `overflow_err` is set, software wrote more features than FIFO could hold.
- If `underflow_err` is set, stream ran out of data during transfer.
