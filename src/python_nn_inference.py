"""
Run the same neural network models in Python with the same test vectors
and timing, matching the PYNQ FPGA test structure.
"""

import numpy as np
import time

# =========================
# Load all 5 quantized models
# =========================
model_ids = [196, 279, 362, 364, 370]
models = {}

for mid in model_ids:
    path = f'weights/fixed_params_{mid}.npz'
    params = np.load(path)
    models[mid] = params
    print(f"Loaded model {mid}")

print(f"Loaded {len(models)} models: {model_ids}")

# =========================
# Test vectors (same as pynq_nn_test.py)
# =========================
test_vectors = [
    [
        17278, 17222, 18900, 21472,
        20689, 21863, 22255, 21025,
        21640, 22926, 21528, 24044,
        23317, 21472, 23205, 25442,
        18668, 22761, 7306,  12143,
        32209, 3245,  26277, 32767
    ],
    [
        1000, 1200, 1100, 900,
        1400, 1300, 1250, 1350,
        1450, 1550, 1600, 1650,
        1700, 1800, 1900, 2000,
        2100, 2200, 2300, 2400,
        2500, 2600, 2700, 2800
    ],
    [16000] * 24,
    [30000] * 24,
]

expected_h1 = [
    [16749, 1279, 14194, 25969],
    [13687, 1043, 11300, 18407],
    [13774,  905, 11747, 20742],
    [ 3972,   37,  3991,  7287],
    [17391, 1492, 12697, 21160],
]

expected_h4 = [
    [14145, 1131, 12383, 21965],
    [13697, 1006, 11162, 19315],
    [13995,  963, 12052, 21064],
    [ 6177,  353,  6001, 10974],
    [17360, 1467, 12762, 21787],
]

# =========================
# Fixed-point forward pass
# =========================
def sign_extend_16(val):
    """Convert 16-bit unsigned to signed"""
    if val & 0x8000:
        return val - 0x10000
    return val

def clamp_16(val):
    """Clamp to 16-bit signed range"""
    if val > 32767:
        return 32767
    if val < -32768:
        return -32768
    return int(val)

def relu_fixed(x):
    """ReLU for fixed-point arrays"""
    return np.maximum(0, x)

def fixed_linear(x, weight, bias, w_scale, b_scale):
    """
    Fixed-point Q8.8 linear operation: y = x @ W.T + b
    Mimics RTL behavior: 
    - Q8.8 * Q8.8 = Q16.16 (stored in 32-bit)
    - Accumulate in 48-bit
    - Shift right by 8 bits to get back to Q8.8
    
    x: input array (int16, Q8.8 format)
    weight: int16 weight matrix (Q8.8 format)
    bias: int16 bias (Q8.8 format)
    w_scale, b_scale: unused (kept for compatibility)
    """
    
    batch_size = x.shape[0]
    out_features = weight.shape[0]
    
    out = np.zeros((batch_size, out_features), dtype=np.int32)
    
    for b in range(batch_size):
        for out_idx in range(out_features):
            # Accumulator: 48 bits
            acc = 0
            
            # MAC: Multiply-Accumulate
            # Q8.8 * Q8.8 = Q16.16
            for in_idx in range(weight.shape[1]):
                a = int(x[b, in_idx])  # Q8.8
                w = int(weight[out_idx, in_idx])  # Q8.8
                
                # Multiply: Q8.8 * Q8.8 = Q16.16
                product = a * w  # This is Q16.16 in a 32-bit signed value
                
                # Accumulate in 48-bit (keep full precision)
                acc += product
            
            # Add bias (Q8.8)
            bias_val = int(bias[out_idx])
            # Bias is Q8.8, need to shift it up to match Q16.16 scale
            acc += (bias_val << 8)
            
            # Convert back from Q16.16 to Q8.8 by shifting right by 8
            result = acc >> 8
            
            # Saturate to int16 range
            out[b, out_idx] = clamp_16(result)
    
    return out

def forward_model(features, model_params):
    """
    Run forward pass for a single model
    features: list of 24 int16 values
    model_params: quantized weights from .npz file
    returns: (h1, h4) as int16 values
    """
    # Convert input to array, ensure signed int16
    x = np.array(features, dtype=np.int16).reshape(1, -1)
    
    # Layer 1: 24 -> 128
    h = fixed_linear(
        x,
        model_params['backbone_0_weight'],
        model_params['backbone_0_bias'],
        float(model_params['backbone_0_weight_scale']),
        float(model_params['backbone_0_bias_scale'])
    )
    h = relu_fixed(h)
    
    # Layer 2: 128 -> 64
    h = fixed_linear(
        h,
        model_params['backbone_3_weight'],
        model_params['backbone_3_bias'],
        float(model_params['backbone_3_weight_scale']),
        float(model_params['backbone_3_bias_scale'])
    )
    h = relu_fixed(h)
    
    # Layer 3: 64 -> 32
    feat = fixed_linear(
        h,
        model_params['backbone_6_weight'],
        model_params['backbone_6_bias'],
        float(model_params['backbone_6_weight_scale']),
        float(model_params['backbone_6_bias_scale'])
    )
    feat = relu_fixed(feat)
    
    # Head h1: 32 -> 1
    h1_out = fixed_linear(
        feat,
        model_params['head_h1_weight'],
        model_params['head_h1_bias'],
        float(model_params['head_h1_weight_scale']),
        float(model_params['head_h1_bias_scale'])
    )
    h1_val = clamp_16(int(h1_out.flatten()[0]))
    
    # Head h4: 32 -> 1
    h4_out = fixed_linear(
        feat,
        model_params['head_h4_weight'],
        model_params['head_h4_bias'],
        float(model_params['head_h4_weight_scale']),
        float(model_params['head_h4_bias_scale'])
    )
    h4_val = clamp_16(int(h4_out.flatten()[0]))
    
    return h1_val, h4_val

# =========================
# Run all tests with timing
# =========================
print("\n" + "="*80)
print("PYTHON MODEL INFERENCE WITH TIMING")
print("="*80)

total = 0
passes = 0
timings_ms = []
timings_per_model = {mid: [] for mid in model_ids}

for m_idx, mid in enumerate(model_ids):
    model_params = models[mid]
    
    for v in range(4):
        total += 1
        features = test_vectors[v]
        
        # Time the inference
        t_start = time.perf_counter()
        h1, h4 = forward_model(features, model_params)
        t_end = time.perf_counter()
        
        elapsed_ms = (t_end - t_start) * 1000.0
        timings_ms.append(elapsed_ms)
        timings_per_model[mid].append(elapsed_ms)
        
        # Compare with expected
        eh1 = expected_h1[m_idx][v]
        eh4 = expected_h4[m_idx][v]
        
        match = (h1 == eh1) and (h4 == eh4)
        if match:
            passes += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"[{status}] mid={mid} v={v} | h1={h1:6d} (exp {eh1:6d}) | "
              f"h4={h4:6d} (exp {eh4:6d}) | time={elapsed_ms:.4f}ms")

# =========================
# Summary statistics
# =========================
print("\n" + "="*80)
print(f"SUMMARY: {passes}/{total} tests passed")
print("="*80)

print(f"\nOverall Timing:")
print(f"  Average: {np.mean(timings_ms):.4f} ms")
print(f"  Min:     {np.min(timings_ms):.4f} ms")
print(f"  Max:     {np.max(timings_ms):.4f} ms")
print(f"  Median:  {np.median(timings_ms):.4f} ms")
print(f"  Std Dev: {np.std(timings_ms):.4f} ms")

print(f"\nTiming by Model:")
for mid in model_ids:
    times = timings_per_model[mid]
    print(f"  Model {mid}: avg={np.mean(times):.4f}ms, "
          f"min={np.min(times):.4f}ms, max={np.max(times):.4f}ms")

throughput = 1000.0 / np.mean(timings_ms)  # inferences per second
print(f"\nThroughput: {throughput:.1f} inferences/sec")
