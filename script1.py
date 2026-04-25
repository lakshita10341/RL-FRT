"""
train_cnn.py  —  13-class fault CNN, trains on fault_dataset_13.mat
Exports to ONNX so MATLAB can load it via onnxruntime or Deep Learning Toolbox.

FIXES vs original:
[1] RMS computed correctly — original used cumsum which gives wrong RMS for the first WIN samples. Fixed with a proper sliding window.
[2] Input shape: original stacked raw+rms → [12×WIN]. Correct but the Conv2d kernel (12,5) at the end must span all 12 rows. Fixed.
[3] Dataset struct parsing — MATLAB cell arrays of structs load differently depending on squeeze_me. Added robust parser.
[4] Added ONNX export with correct dummy input shape for MATLAB inference.
[5] Added per-class accuracy report so you know which fault types are weak.
[6] Saves V_BASE, I_BASE, WIN into a separate .mat for buffer_block.
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────
WIN         = 333       # must match what MATLAB trained with originally
                        # if retraining fresh: set WIN = round(sample_rate/grid_freq)
                        # e.g. 400 for 20kHz/50Hz, 333 for 20kHz/60Hz
STRIDE      = 10        # match original MATLAB STRIDE
NUM_CLASSES = 13
BATCH_SIZE  = 256
EPOCHS      = 25
LR          = 1e-3
NUM_WORKERS = 6         # CPU workers for dataloading

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True
print(f"Device: {DEVICE}")

FAULT_NAMES = ['NO','AG','BG','CG','AB','BC','CA','ABG','BCG','CAG','ABC','ABCG','HIF']

# ── DATASET ───────────────────────────────────────────────────────
class FaultDS(Dataset):
    def __init__(self, X, Y):
        # Add channel dim: [N, 1, 12, WIN]
        self.X = torch.tensor(X).unsqueeze(1)
        self.Y = torch.tensor(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ── CNN ───────────────────────────────────────────────────────────
class FaultCNN(nn.Module):
    """
    Input:  [B, 1, 12, WIN]  — 1 channel, 12 signal rows, WIN time steps
    Output: [B, 13]          — softmax logits

    Architecture:
    - [1×15] convs slide along TIME axis, extract waveform features
    - [12×5] conv spans ALL signal rows — learns cross-phase patterns
    - GlobalAvgPool → FC → 13 classes
    """
    def __init__(self, win=WIN):
        super().__init__()
        self.features = nn.Sequential(
            # Time-axis feature extraction
            nn.Conv2d(1,  32, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32), nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),   # WIN → WIN/2

            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),   # WIN/2 → WIN/4

            # Cross-channel: span all 12 rows
            nn.Conv2d(128, 256, kernel_size=(12, 3), padding=(0, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))        # → [B, 256, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

if __name__ == '__main__':
    # ── LOAD DATASET (auto PU conversion) ────────────────────────────
    from dataset_loader import load_dataset
    samples_all, V_BASE, I_BASE = load_dataset()
    samples = [(sig, lbl) for sig, lbl in samples_all if sig.shape[0] >= WIN]
    print(f"Valid segments (len>={WIN}): {len(samples)}")

    # ── WINDOWING ─────────────────────────────────────────────────────
    def sliding_rms(sig, win):
        """
        Correct sliding-window RMS.
        Returns array same shape as sig.
        Values at indices < win are set to the RMS of first full window.
        """
        sq      = sig ** 2
        # Use cumsum trick but only for valid windows
        cs      = np.cumsum(sq, axis=0)
        # Full windows from index win onwards
        rms     = np.zeros_like(sig)
        rms[win:] = np.sqrt((cs[win:] - cs[:-win]) / win)
        # Fill first WIN rows with first valid RMS
        if len(rms) > win:
            rms[:win] = rms[win]
        return rms

    print("Creating windows ...")
    X_list, Y_list = [], []

    for sig, lbl in tqdm(samples):
        n = sig.shape[0]
        if n < WIN:
            continue

        rms = sliding_rms(sig, WIN)  # [N×6]

        for i in range(0, n - WIN, STRIDE):
            raw_w = sig[i:i+WIN].T   # [6×WIN]
            rms_w = rms[i:i+WIN].T   # [6×WIN]
            combined = np.vstack([raw_w, rms_w])  # [12×WIN]
            X_list.append(combined)
            Y_list.append(lbl)

    X = np.array(X_list, dtype=np.float32)  # [N, 12, WIN]
    Y = np.array(Y_list, dtype=np.int64)
    print(f"Total windows: {X.shape}  labels: {np.unique(Y)}")

    # ── CLASS BALANCE ─────────────────────────────────────────────────
    # Oversample rare classes to 2× the median count
    counts = np.bincount(Y, minlength=NUM_CLASSES)
    print("Class counts before balancing:", counts)

    target_count = int(np.median(counts[counts > 0]) * 2)
    idx_balanced = []
    for c in range(NUM_CLASSES):
        ci = np.where(Y == c)[0]
        if len(ci) == 0:
            continue
        if len(ci) < target_count:
            ci = np.random.choice(ci, target_count, replace=True)
        else:
            ci = np.random.choice(ci, target_count, replace=False)
        idx_balanced.append(ci)

    idx_balanced = np.concatenate(idx_balanced)
    np.random.shuffle(idx_balanced)
    X = X[idx_balanced]
    Y = Y[idx_balanced]
    print(f"Balanced dataset: {X.shape}")

    # ── SPLIT ─────────────────────────────────────────────────────────
    X_tr, X_va, y_tr, y_va = train_test_split(X, Y, test_size=0.2, stratify=Y)
    print(f"Train: {X_tr.shape}  Val: {X_va.shape}")

    # Windows optimization: persistent_workers=True prevents worker recreation overhead
    train_loader = DataLoader(FaultDS(X_tr, y_tr), BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(FaultDS(X_va, y_va), BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = FaultCNN().to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── TRAINING ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (DEVICE == 'cuda')
    # Modern PyTorch 2.x AMP syntax
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            # Modern PyTorch 2.x AMP syntax
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(x), y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        per_class_correct = np.zeros(NUM_CLASSES)
        per_class_total   = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
                for c in range(NUM_CLASSES):
                    mask = (y == c)
                    per_class_total[c]   += mask.sum().item()
                    per_class_correct[c] += (pred[mask] == y[mask]).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'cnn_best.pt')
            print(f"  ✓ Saved best model ({acc:.2f}%)")

    # Per-class report
    print("\nPer-class accuracy:")
    model.load_state_dict(torch.load('cnn_best.pt'))
    model.eval()
    
    # Recompute
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total   = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            for c in range(NUM_CLASSES):
                mask = (y == c)
                per_class_total[c]   += mask.sum().item()
                per_class_correct[c] += (pred[mask] == y[mask]).sum().item()

    for c in range(NUM_CLASSES):
        if per_class_total[c] > 0:
            print(f"  {c:2d} {FAULT_NAMES[c]:<6s}: {100*per_class_correct[c]/per_class_total[c]:.1f}%  ({int(per_class_total[c])} samples)")

    # ── ONNX EXPORT ───────────────────────────────────────────────────
    # MATLAB Deep Learning Toolbox can import ONNX models directly.
    # In MATLAB: net = importONNXNetwork('cnn_fault.onnx', 'OutputLayerType','classification')
    dummy = torch.zeros(1, 1, 12, WIN).to(DEVICE)
    torch.onnx.export(
        model, dummy, 'cnn_fault.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )
    print("✓ Exported cnn_fault.onnx")

    # Also save metadata so buffer_block knows WIN, V_BASE, I_BASE
    sio.savemat('cnn_metadata.mat', {
        'V_BASE': V_BASE,
        'I_BASE': I_BASE,
        'WIN':    WIN,
        'channels': 12,
        'best_val_acc': best_acc
    })
    print(f"✓ Saved cnn_metadata.mat  (WIN={WIN}, best_acc={best_acc:.2f}%)")
    print("\nDone. Copy cnn_fault.onnx and cnn_metadata.mat to MATLAB working directory.")