import numpy as np

def relu(x):
    return np.maximum(0, x)

def fixed_linear(x, weight, bias):
    # x: (n,) int16 Q8.8
    # weight: (m, n) int16 Q8.8
    # bias: (m,) int16 Q8.8
    # out: (m,) int16 Q8.8
    out = np.zeros(weight.shape[0], dtype=np.int32)
    for j in range(weight.shape[0]):
        sum = 0
        for i in range(weight.shape[1]):
            prod = np.int32(x[i]) * np.int32(weight[j, i])
            sum += prod
        # Shift right 8 for Q16.16 to Q8.8
        sum >>= 8
        sum += bias[j]
        # Clip to 16-bit signed
        if sum > 32767:
            sum = 32767
        elif sum < -32768:
            sum = -32768
        out[j] = sum
    return out.astype(np.int16)

def run_model(features, params):
    # features: (24,) int16 Q8.8
    # Layer 1
    h1 = relu(fixed_linear(features, params['backbone_0_weight'], params['backbone_0_bias']))
    # Layer 2
    h2 = relu(fixed_linear(h1, params['backbone_3_weight'], params['backbone_3_bias']))
    # Layer 3
    h3 = relu(fixed_linear(h2, params['backbone_6_weight'], params['backbone_6_bias']))
    # Head h1
    h1_out = fixed_linear(h3, params['head_h1_weight'], params['head_h1_bias'])[0]
    # Head h4
    h4_out = fixed_linear(h3, params['head_h4_weight'], params['head_h4_bias'])[0]
    return h1_out, h4_out

# Test vectors from testbench
test_vectors = [
    # vector 0
    np.array([17278, 17222, 18900, 21472, 20689, 21863, 22255, 21025, 21640, 22926, 21528, 24044, 23317, 21472, 23205, 25442, 18668, 22761, 7306, 12143, 32209, 3245, 26277, 32767], dtype=np.int16),
    # vector 1
    np.array([1000, 1200, 1100, 900, 1400, 1300, 1250, 1350, 1450, 1550, 1600, 1650, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800], dtype=np.int16),
    # vector 2
    np.array([16000] * 24, dtype=np.int16),
    # vector 3
    np.array([30000] * 24, dtype=np.int16),
]

clients = ['196', '279', '362', '364', '370']

for client_idx, client in enumerate(clients):
    params = np.load(f'weights/fixed_params_{client}.npz')
    print(f"\nClient {client} (model {client_idx}):")
    for vec_idx, features in enumerate(test_vectors):
        h1_out, h4_out = run_model(features, params)
        print(f"  Vector {vec_idx}: h1={h1_out}, h4={h4_out}")