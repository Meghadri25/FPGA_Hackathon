from pynq import Overlay, MMIO
import time

# =========================
# Load bitstream
# =========================
ol = Overlay("design_1.bit")  
ip_base = ol.nn_axi_wrapper_0.mmio.base_addr
ip_range = ol.nn_axi_wrapper_0.mmio.length

mmio = MMIO(ip_base, ip_range)

# =========================
# Registers
# =========================
REG_CONTROL       = 0x00
REG_MODEL_SELECT  = 0x04
REG_FEATURE_IN    = 0x08
REG_FEAT_COUNT    = 0x0C
REG_FIFO_LEVEL    = 0x10
REG_H1_OUT        = 0x14
REG_H4_OUT        = 0x18

# =========================
# Helpers
# =========================
def to_u16(x):
    return x & 0xFFFF

def from_s16(x):
    x &= 0xFFFF
    return x - 0x10000 if (x & 0x8000) else x

def status():
    s = mmio.read(REG_CONTROL)
    return {
        "ready":   (s >> 0) & 1,
        "busy":    (s >> 1) & 1,
        "done":    (s >> 2) & 1,
        "full":    (s >> 3) & 1,
        "empty":   (s >> 4) & 1,
        "ovf_err": (s >> 5) & 1,
        "udf_err": (s >> 6) & 1,
        "inv_err": (s >> 7) & 1,
        "state":   (s >> 8) & 0x7,
        "raw": s
    }

def clear_all():
    mmio.write(REG_CONTROL, (1 << 1) | (1 << 2) | (1 << 3))

# =========================
# Run once with timing
# =========================
def run_once(features, model_select, timeout_s=2.0):
    clear_all()

    mmio.write(REG_MODEL_SELECT, model_select & 0x7)
    mmio.write(REG_FEAT_COUNT, len(features) & 0xFFFF)

    # -------- Load timing --------
    t_load_start = time.perf_counter()

    for f in features:
        mmio.write(REG_FEATURE_IN, to_u16(int(f)))

    t_load_end = time.perf_counter()

    lvl = mmio.read(REG_FIFO_LEVEL) & 0xFFFF
    if lvl < len(features):
        raise RuntimeError(f"FIFO level too low: {lvl} < {len(features)}")

    # -------- Compute timing --------
    t_start = time.perf_counter()

    mmio.write(REG_CONTROL, 1 << 0)  # start

    while True:
        st = status()
        if st["done"]:
            break
        if time.perf_counter() - t_start > timeout_s:
            raise TimeoutError(f"Timeout. status={st}")
        time.sleep(0.0005)

    t_done = time.perf_counter()

    # -------- Read outputs --------
    h1 = from_s16(mmio.read(REG_H1_OUT))
    h4 = from_s16(mmio.read(REG_H4_OUT))

    # clear done
    mmio.write(REG_CONTROL, 1 << 1)

    return {
        "h1": h1,
        "h4": h4,
        "status": st,
        "t_load": t_load_end - t_load_start,
        "t_compute": t_done - t_start,
        "t_total": t_done - t_load_start
    }

# =========================
# Test vectors
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
    [30000] * 24
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
# Run all tests
# =========================
total = 0
fails = 0

t_total_all = []
t_compute_all = []
t_load_all = []

for m in range(5):
    for v in range(4):
        total += 1

        res = run_once(test_vectors[v], m)

        h1 = res["h1"]
        h4 = res["h4"]
        st = res["status"]

        eh1 = expected_h1[m][v]
        eh4 = expected_h4[m][v]

        ok = (h1 == eh1) and (h4 == eh4) and \
             (st["ovf_err"] == 0) and (st["udf_err"] == 0) and (st["inv_err"] == 0)

        t_total_all.append(res["t_total"])
        t_compute_all.append(res["t_compute"])
        t_load_all.append(res["t_load"])

        if ok:
            print(f"[PASS] m={m} v={v} h1={h1} h4={h4} | "
                  f"load={res['t_load']*1e3:.2f}ms compute={res['t_compute']*1e3:.2f}ms total={res['t_total']*1e3:.2f}ms")
        else:
            fails += 1
            print(f"[FAIL] m={m} v={v} got(h1={h1},h4={h4}) exp(h1={eh1},h4={eh4}) status={st}")

print(f"\nSummary: {total-fails}/{total} passed")

# =========================
# Timing summary
# =========================
def summarize(arr, name):
    return (f"{name}: avg={sum(arr)/len(arr)*1e3:.2f}ms | "
            f"min={min(arr)*1e3:.2f}ms | max={max(arr)*1e3:.2f}ms")

print("\nTiming Summary:")
print(summarize(t_load_all, "Load"))
print(summarize(t_compute_all, "Compute"))
print(summarize(t_total_all, "Total"))