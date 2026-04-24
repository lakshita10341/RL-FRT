"""
dataset_loader.py
Loads fault dataset and converts to per-unit.
"""
import numpy as np
import scipy.io as sio

def load_dataset():
    """
    Loads fault_dataset_13_pu.mat or fault_dataset_13.mat,
    and returns signals converted to per-unit.

    Returns:
        samples (list): list of (signal, label) tuples
        V_BASE (float): base voltage
        I_BASE (float): base current
    """
    V_BASE = 20000.0
    I_BASE = 200.0

    try:
        data = sio.loadmat('fault_dataset_13_pu.mat')
        print("Loaded pre-processed 'fault_dataset_13_pu.mat'")
        signals = data['signals']
        labels  = data['labels'].flatten()
        lengths = data['lengths'].flatten()
        samples = []
        for i in range(len(labels)):
            sig = signals[i, :lengths[i], :]
            samples.append((sig, labels[i]))
        return samples, V_BASE, I_BASE

    except FileNotFoundError:
        print("Could not find 'fault_dataset_13_pu.mat', loading 'fault_dataset_13.mat' instead.")
        data = sio.loadmat('fault_dataset_13.mat', squeeze_me=True, struct_as_record=False)
        dataset = data['dataset'].flatten()

        samples = []
        for item in dataset:
            sig = np.array(item.signal, dtype=np.float64)
            # Convert to PU
            sig[:, :3] /= V_BASE
            sig[:, 3:] /= I_BASE
            samples.append((sig.astype(np.float32), int(item.label)))

        return samples, V_BASE, I_BASE
