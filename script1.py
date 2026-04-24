import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# =========================
# CONFIG
# =========================
WIN = 333          # 1 cycle @ 60 Hz
STRIDE = 50
NUM_CLASSES = 13
BATCH_SIZE = 512
EPOCHS = 12

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

print("Device:", DEVICE)

# =========================
# LOAD DATASET (ROBUST)
# =========================
data = sio.loadmat('fault_dataset_13.mat', squeeze_me=True, struct_as_record=False)

dataset = data['dataset']

# SAFE BASE HANDLING
def get_scalar(mat, name, default=1.0):
    if name in mat:
        val = mat[name]
        try:
            return float(np.array(val).squeeze())
        except:
            return default
    return default

V_BASE = get_scalar(data, 'V_BASE', 1.0)
I_BASE = get_scalar(data, 'I_BASE', 1.0)

print("V_BASE:", V_BASE, "I_BASE:", I_BASE)

# =========================
# RMS FUNCTION (FAST)
# =========================
def compute_rms(sig, win):
    sq = sig**2
    cumsum = np.cumsum(sq, axis=0)

    rms = np.zeros_like(sig)

    for i in range(win, len(sig)):
        rms[i] = np.sqrt((cumsum[i] - cumsum[i-win]) / win)

    return rms

# =========================
# WINDOW CREATION (PARALLEL)
# =========================
def process_item(item):
    sig = item.signal   # already PU
    lbl = int(item.label)

    X_local, Y_local = [], []

    if sig.shape[0] < WIN:
        return X_local, Y_local

    rms_sig = compute_rms(sig, WIN)

    for i in range(0, sig.shape[0] - WIN, STRIDE):
        raw_w = sig[i:i+WIN].T
        rms_w = rms_sig[i:i+WIN].T

        combined = np.vstack([raw_w, rms_w])  # 12×WIN

        X_local.append(combined)
        Y_local.append(lbl)

    return X_local, Y_local

print("Creating windows...")

results = Parallel(n_jobs=-1)(
    delayed(process_item)(item) for item in dataset
)

X, Y = [], []
for x_part, y_part in results:
    X.extend(x_part)
    Y.extend(y_part)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

print("Total windows:", X.shape)

# =========================
# BALANCE DATASET
# =========================
fault_idx = np.where(Y != 0)[0]
no_idx = np.where(Y == 0)[0]

np.random.shuffle(no_idx)
no_idx = no_idx[:len(fault_idx)]

keep = np.concatenate([fault_idx, no_idx])

X = X[keep]
Y = Y[keep]

print("Balanced:", X.shape)

# =========================
# TRAIN / VAL SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, stratify=Y
)

# =========================
# DATASET
# =========================
class FaultDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(
    FaultDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    FaultDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)

# =========================
# CNN MODEL (12-CHANNEL)
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (1,15), padding=(0,7)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, (1,15), padding=(0,7)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d((1,4)),

            nn.Conv2d(64, 128, (12,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d((1,2)),

            nn.Conv2d(128, 256, (1,3), padding=(0,1)),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.unsqueeze(1)   # [B,1,12,WIN]
        x = self.net(x)
        return self.fc(x)

model = CNN().to(DEVICE)

# =========================
# TRAINING
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

# =========================
# EVALUATION
# =========================
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader)

    for x, y in loop:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    acc = evaluate()
    print(f"Epoch {epoch+1}: Val Accuracy = {acc:.2f}%")

# =========================
# SAVE FOR MATLAB
# =========================
weights = {}
for name, param in model.state_dict().items():
    weights[name] = param.cpu().numpy()

sio.savemat('cnn_model_final.mat', {
    'weights': weights,
    'V_BASE': V_BASE,
    'I_BASE': I_BASE,
    'WIN': WIN,
    'channels': 12
})

print("✅ Saved cnn_model_final.mat")