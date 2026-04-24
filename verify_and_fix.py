"""
verify_and_fix_dataset.py
Run this FIRST before training CNN or RL.
It checks whether fault_dataset_13.mat signals are in PU or raw Volts/Amps,
and fixes them in-place if needed.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ── LOAD ──────────────────────────────────────────────────────────
data    = sio.loadmat('fault_dataset_13.mat', squeeze_me=True, struct_as_record=False)
dataset = data['dataset'].flatten()

V_BASE = float(np.array(data.get('V_BASE', [1.0])).squeeze())
I_BASE = float(np.array(data.get('I_BASE', [1.0])).squeeze())

print(f"Stored V_BASE = {V_BASE:.4f}")
print(f"Stored I_BASE = {I_BASE:.4f}")
print(f"Total segments: {len(dataset)}")

# ── CHECK WHETHER SIGNALS ARE ALREADY IN PU ───────────────────────
# Sample 10 segments and check their peak voltage
peaks = []
for i in range(min(10, len(dataset))):
    sig = np.array(dataset[i].signal, dtype=np.float32)
    peaks.append(np.max(np.abs(sig[:, :3])))   # voltage columns only

median_peak = float(np.median(peaks))
print(f"\nMedian peak voltage in dataset: {median_peak:.4f}")

if median_peak > 2.0:
    print(f"\n*** SIGNALS ARE IN RAW VOLTS (peak={median_peak:.1f}) ***")
    print(f"    V_BASE={V_BASE:.2f} will be used to convert to PU.")
    already_pu = False
else:
    print(f"\n*** SIGNALS ALREADY IN PU (peak={median_peak:.4f}) ***")
    already_pu = True

# ── PLOT FIRST SEGMENT BEFORE FIX ─────────────────────────────────
sig0 = np.array(dataset[0].signal, dtype=np.float32)
lbl0 = int(np.array(dataset[0].label).squeeze())
fs   = 20000
t0   = np.arange(sig0.shape[0]) / fs

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(t0, sig0[:, 0], label='Va')
axes[0].plot(t0, sig0[:, 1], label='Vb')
axes[0].plot(t0, sig0[:, 2], label='Vc')
axes[0].set_title(f"Voltages BEFORE fix — label={lbl0}  peak={median_peak:.2f}")
axes[0].set_ylabel("Value"); axes[0].legend(); axes[0].grid()

axes[1].plot(t0, sig0[:, 3], label='Ia')
axes[1].plot(t0, sig0[:, 4], label='Ib')
axes[1].plot(t0, sig0[:, 5], label='Ic')
axes[1].set_title("Currents BEFORE fix")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Value")
axes[1].legend(); axes[1].grid()
plt.tight_layout()
plt.savefig("dataset_before_fix.png", dpi=100)
plt.show()
print("Saved dataset_before_fix.png")

# ── FIX: CONVERT RAW → PU ─────────────────────────────────────────
if not already_pu:
    print(f"\nConverting all segments to PU ...")
    print(f"  dividing voltages by {V_BASE:.4f}")
    print(f"  dividing currents by {I_BASE:.4f}")

    for i, item in enumerate(dataset):
        sig = np.array(item.signal, dtype=np.float64)
        sig[:, :3] /= V_BASE   # Va Vb Vc → PU
        sig[:, 3:] /= I_BASE   # Ia Ib Ic → PU
        dataset[i].signal = sig.astype(np.float32)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(dataset)}")

    print("Done converting.")

    # Verify
    peaks_after = []
    for i in range(min(10, len(dataset))):
        sig = np.array(dataset[i].signal, dtype=np.float32)
        peaks_after.append(np.max(np.abs(sig[:, :3])))
    print(f"Peak voltage after fix: {np.median(peaks_after):.4f} pu (should be ~1.0)")

    # Plot after
    sig1 = np.array(dataset[0].signal, dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t0[:len(sig1)], sig1[:, 0], label='Va')
    axes[0].plot(t0[:len(sig1)], sig1[:, 1], label='Vb')
    axes[0].plot(t0[:len(sig1)], sig1[:, 2], label='Vc')
    axes[0].set_title(f"Voltages AFTER PU conversion — should be ±1.0")
    axes[0].set_ylabel("PU"); axes[0].legend(); axes[0].grid()
    axes[1].plot(t0[:len(sig1)], sig1[:, 3], label='Ia')
    axes[1].plot(t0[:len(sig1)], sig1[:, 4], label='Ib')
    axes[1].plot(t0[:len(sig1)], sig1[:, 5], label='Ic')
    axes[1].set_title("Currents AFTER PU conversion")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("PU")
    axes[1].legend(); axes[1].grid()
    plt.tight_layout()
    plt.savefig("dataset_after_fix.png", dpi=100)
    plt.show()
    print("Saved dataset_after_fix.png")

    # ── SAVE FIXED DATASET ────────────────────────────────────────
    print("\nSaving fixed dataset to fault_dataset_13_pu.mat ...")
    # We can't easily re-save the struct array, so save as arrays instead
    # which is much faster to load in Python training

    max_len = max(np.array(item.signal).shape[0] for item in dataset)
    n       = len(dataset)

    signals = np.zeros((n, max_len, 6), dtype=np.float32)
    labels  = np.zeros(n, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)

    for i, item in enumerate(dataset):
        sig = np.array(item.signal, dtype=np.float32)
        L   = sig.shape[0]
        signals[i, :L, :] = sig
        labels[i]  = int(np.array(item.label).squeeze())
        lengths[i] = L

    sio.savemat('fault_dataset_13_pu.mat', {
        'signals': signals,
        'labels':  labels,
        'lengths': lengths,
        'V_BASE':  V_BASE,
        'I_BASE':  I_BASE,
        'already_pu': 1
    })
    print("Saved fault_dataset_13_pu.mat")
    print("\n*** Use fault_dataset_13_pu.mat for CNN and RL training ***")

else:
    print("\nDataset is already in PU — no fix needed.")
    print("You can use fault_dataset_13.mat directly for training.")

# ── SUMMARY STATISTICS ────────────────────────────────────────────
print("\n=== DATASET SUMMARY ===")
labels_all = np.array([int(np.array(item.label).squeeze()) for item in dataset])
fault_names = ['NO','AG','BG','CG','AB','BC','CA','ABG','BCG','CAG','ABC','ABCG','HIF']
for c in range(13):
    n = np.sum(labels_all == c)
    if n > 0:
        print(f"  Class {c:2d} {fault_names[c]:<6s}: {n} segments")