import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
try:
    data = sio.loadmat('fault_dataset_13_pu.mat')
    print("Loaded pre-processed 'fault_dataset_13_pu.mat'")
    signals = data['signals']
    labels  = data['labels'].flatten()
    lengths = data['lengths'].flatten()
    
    # Reconstruct the dataset structure for plotting
    dataset = []
    for i in range(len(labels)):
        sig = signals[i, :lengths[i], :]
        # Create a simple object to mimic the old structure
        sample = type('sample', (object,), {'signal': sig, 'label': labels[i]})()
        dataset.append(sample)

except FileNotFoundError:
    print("Could not find 'fault_dataset_13_pu.mat', trying 'fault_dataset_13.mat'.")
    data = sio.loadmat('fault_dataset_13.mat', squeeze_me=True, struct_as_record=False)
    dataset = data['dataset']

print("Total samples:", len(dataset))

# =========================
# PICK ONE SAMPLE
# =========================
sample_idx = 0   # change this to view different samples
sig = dataset[sample_idx].signal   # shape [N,6]
label = dataset[sample_idx].label

print("Sample label:", label)
print("Signal shape:", sig.shape)

# =========================
# TIME AXIS
# =========================
fs = 20000   # sampling rate (adjust if needed)
t = np.arange(sig.shape[0]) / fs

# =========================
# PLOT VOLTAGES
# =========================
plt.figure(figsize=(10,6))

plt.plot(t, sig[:,0], label='Va')
plt.plot(t, sig[:,1], label='Vb')
plt.plot(t, sig[:,2], label='Vc')

plt.title(f"Voltage (Sample {sample_idx}, Label={label})")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (PU)")
plt.legend()
plt.grid()

# =========================
# PLOT CURRENTS
# =========================
plt.figure(figsize=(10,6))

plt.plot(t, sig[:,3], label='Ia')
plt.plot(t, sig[:,4], label='Ib')
plt.plot(t, sig[:,5], label='Ic')

plt.title(f"Current (Sample {sample_idx}, Label={label})")
plt.xlabel("Time (s)")
plt.ylabel("Current (PU)")
plt.legend()
plt.grid()

plt.show()

# =========================
# QUICK NUMERIC CHECK
# =========================
print("\nMax Voltage:", np.max(np.abs(sig[:,:3])))
print("Max Current:", np.max(np.abs(sig[:,3:])))