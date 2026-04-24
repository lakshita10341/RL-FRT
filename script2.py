import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from joblib import Parallel, delayed
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

# =========================
# LOAD DATASET (ROBUST)
# =========================
data = sio.loadmat('fault_dataset_13.mat', squeeze_me=True, struct_as_record=False)

dataset = data['dataset']

# 🔥 SAFE BASE HANDLING
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

print("V_BASE:", V_BASE)
print("I_BASE:", I_BASE)

# =========================
# PARAMETERS
# =========================
STATE_DIM = 12
ACTION_DIM = 2
T_STEPS = 800
DT = 5e-5

# =========================
# LVRT CURVE
# =========================
def lvrt_floor(t):
    if t <= 0.15: return 0.0
    elif t <= 0.625: return (t-0.15)*0.2/(0.625-0.15)
    elif t <= 1.3: return 0.2+(t-0.625)*0.3/(1.3-0.625)
    elif t <= 5.0: return 0.5+(t-1.3)*0.4/(5.0-1.3)
    else: return 0.9

# =========================
# SIMPLE GRID MODEL
# =========================
def simulate(V, Id, Iq, fault):
    disturbance = 0.4 if fault else 0.0
    dV = 0.5*Iq + 0.1*Id - disturbance
    Vn = V + dV * 0.02
    return np.clip(Vn, 0, 1.5)

# =========================
# DATA PREP (PARALLEL)
# =========================
def build_sample(item):
    sig = item.signal  # already PU
    lbl = int(item.label)

    if sig.shape[0] < T_STEPS:
        return None

    return sig[:T_STEPS].T, lbl

samples = Parallel(n_jobs=-1)(
    delayed(build_sample)(d) for d in dataset
)

samples = [s for s in samples if s is not None]

print("Samples:", len(samples))

# =========================
# REPLAY BUFFER
# =========================
class ReplayBuffer:
    def __init__(self, size=200000):
        self.buffer = deque(maxlen=size)

    def add(self, s,a,r,s2,d):
        self.buffer.append((s,a,r,s2,d))

    def sample(self, batch):
        batch = random.sample(self.buffer, batch)
        s,a,r,s2,d = zip(*batch)

        return (
            torch.tensor(s, dtype=torch.float32).to(DEVICE),
            torch.tensor(a, dtype=torch.float32).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(s2, dtype=torch.float32).to(DEVICE),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )

    def size(self):
        return len(self.buffer)

buffer = ReplayBuffer()

# =========================
# NETWORKS
# =========================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128,ACTION_DIM), nn.Tanh()
        )
    def forward(self,x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM+ACTION_DIM,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self,s,a):
        return self.net(torch.cat([s,a],dim=1))

actor = Actor().to(DEVICE)
critic = Critic().to(DEVICE)

target_actor = Actor().to(DEVICE)
target_critic = Critic().to(DEVICE)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

opt_a = optim.Adam(actor.parameters(), lr=1e-4)
opt_c = optim.Adam(critic.parameters(), lr=5e-4)

# =========================
# TRAINING LOOP
# =========================
GAMMA = 0.99
TAU = 0.005

for episode in range(500):

    seg, lbl = random.choice(samples)

    V = 1.0
    t_fault = 0

    for t in range(T_STEPS):

        raw = seg[:,t]

        obs = np.concatenate([
            raw,
            [V, 0, t_fault, lvrt_floor(t_fault), lbl/12, 0]
        ])

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # action + noise
        a = actor(s).detach().cpu().numpy()[0]
        a += np.random.normal(0, 0.2, 2)
        a = np.clip(a, -1, 1)

        Id, Iq = a

        Vn = simulate(V, Id, Iq, lbl > 0)
        Vmin = lvrt_floor(t_fault)

        # =========================
        # REWARD
        # =========================
        reward = -20 * abs(Vn - 1)

        if Vn < Vmin:
            reward -= 100 * (Vmin - Vn)

        reward -= 2 * (Id**2 + Iq**2)

        done = Vn < 0.2

        next_obs = np.concatenate([
            raw,
            [Vn, Vn - V, t_fault, Vmin, lbl/12, 0]
        ])

        buffer.add(obs, a, reward, next_obs, done)

        V = Vn
        t_fault += DT

        # =========================
        # TRAIN
        # =========================
        if buffer.size() > 2000:

            s_b, a_b, r_b, s2_b, d_b = buffer.sample(256)

            with torch.no_grad():
                a2 = target_actor(s2_b)
                q2 = target_critic(s2_b, a2)
                y = r_b + GAMMA * (1 - d_b) * q2

            q = critic(s_b, a_b)
            loss_c = ((q - y) ** 2).mean()

            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            loss_a = -critic(s_b, actor(s_b)).mean()

            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()

            # soft update
            for tp, p in zip(target_actor.parameters(), actor.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

            for tp, p in zip(target_critic.parameters(), critic.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Buffer {buffer.size()}")

# =========================
# SAVE FOR MATLAB
# =========================
weights = {}
for name, param in actor.state_dict().items():
    weights[name] = param.cpu().numpy()

sio.savemat("rl_agent_python.mat", {
    "actor_weights": weights,
    "V_BASE": V_BASE,
    "I_BASE": I_BASE
})

print("✅ Saved rl_agent_python.mat")
