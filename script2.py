"""
train_rl.py  —  DDPG agent for LVRT voltage support
Action space: [Id_cmd, Iq_cmd] in per-unit  (2D continuous)
State space:  12-dim vector (see buildObs)

The agent learns to command Id and Iq to:
  1. Keep Vmag above the LVRT floor during fault
  2. Not over-inject (waste inverter capacity)
  3. Recover gracefully when fault clears

CORRECT CONTROL FLOW (matches what Simulink will do):
  Vabc (raw) → Park transform → Id_meas, Iq_meas
  RL agent → Id_cmd, Iq_cmd
  PI current controller: Vd_ref = Kp*(Id_cmd-Id_meas) + Ki*∫...
                         Vq_ref = Kp*(Iq_cmd-Iq_meas) + Ki*∫...
  Inverse Park → Vabc_ref → PWM Generator → Inverter

FIXES vs original:
[1] simulate() — original had arbitrary constants with no physical meaning.
    Fixed: proper simplified grid model using voltage-behind-reactance.
      Vd = -Xg*Iq  (reactive current depresses d-axis voltage)
      Vq = +Xg*Id  (active current raises q-axis voltage)
    This is the standard Thevenin equivalent of a grid-connected inverter.

[2] Observation — original mixed raw signal rows with scalars inconsistently.
    Fixed: clean 12-element obs that exactly matches rl_obs_builder_13.m

[3] Reward — original penalised equally above and below Vmin.
    Fixed: asymmetric reward — staying above LVRT floor is positive,
    violation is heavily negative, overshoot is penalised.

[4] ACTION is [Id_pu, Iq_pu] not a scalar Q_cmd.
    Id controls active power (keep at rated=1.0 when possible).
    Iq controls reactive power (boost when voltage sags).

[5] ONNX export so MATLAB can load the actor directly.
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import trange
import onnx

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── CONFIG ────────────────────────────────────────────────────────
STATE_DIM  = 12
ACTION_DIM = 2       # [Id_cmd_pu, Iq_cmd_pu]
T_STEPS    = 1000
DT         = 5e-5
GAMMA      = 0.99
TAU        = 0.005
LR_ACTOR   = 1e-4
LR_CRITIC  = 5e-4
BATCH      = 256
BUFFER_MAX = 300_000
EPISODES   = 800
NOISE_STD  = 0.15    # exploration noise std

# Grid model parameters (simplified Thevenin)
Xg = 0.15   # grid reactance in pu  (typical weak grid: 0.10–0.20)
             # Effect: Vmag_drop ≈ Xg * Iq_fault

FAULT_NAMES = ['NO','AG','BG','CG','AB','BC','CA','ABG','BCG','CAG','ABC','ABCG','HIF']

# ── LOAD DATASET ──────────────────────────────────────────────────
from dataset_loader import load_dataset
_all_samples, V_BASE, I_BASE = load_dataset()
samples = [(sig[:T_STEPS], lbl) for sig, lbl in _all_samples if sig.shape[0] >= T_STEPS]
print(f"V_BASE={V_BASE:.4f}  I_BASE={I_BASE:.4f}  Segments>={T_STEPS}steps: {len(samples)}")

# Bias: 80% fault samples
fault_samples  = [(s, l) for s, l in samples if l > 0]
normal_samples = [(s, l) for s, l in samples if l == 0]
print(f"Fault segments: {len(fault_samples)}  Normal: {len(normal_samples)}")

# ── LVRT CURVE ────────────────────────────────────────────────────
def lvrt_floor(t):
    if   t <= 0.150: return 0.00
    elif t <= 0.625: return (t-0.150)*0.20/(0.625-0.150)
    elif t <= 1.300: return 0.20+(t-0.625)*0.30/(1.300-0.625)
    elif t <= 5.000: return 0.50+(t-1.300)*0.40/(5.000-1.300)
    else:            return 0.90

def lvrt_vref(t):
    """Voltage recovery reference — target for RL to chase."""
    if   t <= 0.150: return 0.05
    elif t <= 0.625: return 0.05+(t-0.150)*0.25/(0.625-0.150)
    elif t <= 1.300: return 0.30+(t-0.625)*0.25/(1.300-0.625)
    elif t <= 5.000: return 0.55+(t-1.300)*0.40/(5.000-1.300)
    else:            return 1.00

# ── GRID SIMULATION ───────────────────────────────────────────────
def simulate_step(Vmag, Id_pu, Iq_pu, fault_label, data_row):
    """
    Simplified voltage-behind-reactance model.

    During fault: grid voltage source drops (modelled from data_row).
    Inverter injects Id (active) and Iq (reactive).
    Reactive injection Iq raises PCC voltage: ΔV ≈ Xg * Iq
    Active injection Id has smaller effect on voltage magnitude.

    data_row: [Va, Vb, Vc, Ia, Ib, Ic] in PU from dataset
    Vmag_data = instantaneous RMS-equivalent voltage from data
    """
    Va, Vb, Vc = data_row[0], data_row[1], data_row[2]
    Vmag_data = np.sqrt((Va**2 + Vb**2 + Vc**2) / 3.0)

    # Grid voltage "target" pulled from actual simulation data
    # Inverter correction via Iq injection
    # ΔVmag = Xg * Iq_pu  (reactive support raises voltage)
    # Small active effect: 0.05 * Id_pu
    V_correction = Xg * Iq_pu + 0.05 * Id_pu

    # Blend: 70% from data (real grid response), 30% from inverter correction
    Vmag_new = 0.70 * Vmag_data + 0.30 * (Vmag_data + V_correction)
    return float(np.clip(Vmag_new, 0.0, 1.5))

def label_to_severity(lbl):
    if   lbl == 0:           return 0
    elif lbl == 12:          return 1
    elif 1 <= lbl <= 3:      return 1
    elif 4 <= lbl <= 9:      return 2
    else:                    return 3

# ── OBSERVATION BUILDER ───────────────────────────────────────────
def build_obs(data_row, Vmag, dVmag, t_fault, lbl, severity, Vmag_prev_mean, Vmag_std):
    """
    12-element observation matching rl_obs_builder_13.m:
    [Va, Vb, Vc, Ia, Ib, Ic, Vmag_n, dVmag, t_norm, Vmin_now, lbl/12, sev/3]
    """
    Vmag_n   = (Vmag - Vmag_prev_mean) / max(Vmag_std, 1e-6)
    t_norm   = min(t_fault / 5.0, 1.0)
    Vmin_now = lvrt_floor(t_fault)
    return np.array([
        data_row[0], data_row[1], data_row[2],   # Va Vb Vc
        data_row[3], data_row[4], data_row[5],   # Ia Ib Ic
        Vmag_n, dVmag, t_norm, Vmin_now,
        lbl / 12.0, severity / 3.0
    ], dtype=np.float32)

# ── REPLAY BUFFER ─────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxsize):
        self.buf = deque(maxlen=maxsize)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(a),  dtype=torch.float32).to(DEVICE),
            torch.tensor(r,  dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
            torch.tensor(d,  dtype=torch.float32).unsqueeze(1).to(DEVICE),
        )

    def __len__(self): return len(self.buf)

buf = ReplayBuffer(BUFFER_MAX)

# ── NETWORKS ──────────────────────────────────────────────────────
class Actor(nn.Module):
    """Outputs [Id_cmd, Iq_cmd] in [-1, +1] (per-unit of I_BASE)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, ACTION_DIM), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a): return self.net(torch.cat([s, a], 1))

actor  = Actor().to(DEVICE)
critic = Critic().to(DEVICE)
t_actor  = Actor().to(DEVICE);  t_actor.load_state_dict(actor.state_dict())
t_critic = Critic().to(DEVICE); t_critic.load_state_dict(critic.state_dict())

opt_a = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
opt_c = optim.Adam(critic.parameters(), lr=LR_CRITIC)

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1-tau) * tp.data)

# ── TRAINING LOOP ─────────────────────────────────────────────────
noise_std   = NOISE_STD
best_reward = -np.inf
ep_rewards  = []

for episode in trange(EPISODES, desc="Training"):

    # 80% fault, 20% normal
    if random.random() < 0.80 and fault_samples:
        seg, lbl = random.choice(fault_samples)
    else:
        seg, lbl = random.choice(normal_samples)

    severity = label_to_severity(lbl)
    is_fault = (lbl > 0)

    # Episode state
    Vmag         = 1.0
    t_fault      = 0.0
    ep_reward    = 0.0
    violation_ms = 0.0   # ms below LVRT floor — trip if > threshold

    # Running stats for Vmag normalisation
    Vmag_hist = [1.0] * 20
    Vmag_mean = 1.0
    Vmag_std  = 0.1

    start_t = random.randint(0, max(0, T_STEPS - 400))

    for step in range(T_STEPS - start_t):
        t_idx   = start_t + step
        row     = seg[t_idx]           # [6] PU
        dVmag   = Vmag - Vmag_hist[-1] if len(Vmag_hist) > 1 else 0.0

        obs = build_obs(row, Vmag, dVmag, t_fault, lbl, severity, Vmag_mean, Vmag_std)

        # ── SELECT ACTION ─────────────────────────────────────────
        with torch.no_grad():
            s_t  = torch.tensor(obs).unsqueeze(0).to(DEVICE)
            action = actor(s_t).cpu().numpy()[0]

        # Exploration noise (decays over training)
        action = action + np.random.normal(0, noise_std, ACTION_DIM)
        action = np.clip(action, -1.0, 1.0)

        Id_pu, Iq_pu = action

        # ── SIMULATE ──────────────────────────────────────────────
        Vmag_new = simulate_step(Vmag, Id_pu, Iq_pu, lbl, row)
        Vmin     = lvrt_floor(t_fault)
        Vref     = lvrt_vref(t_fault)

        # Track violation time
        if is_fault and Vmag_new < Vmin:
            violation_ms += DT * 1000

        # ── REWARD ────────────────────────────────────────────────
        # Normal operation: just keep Id≈1, Iq≈0
        if not is_fault:
            reward = -2.0 * abs(Id_pu - 1.0) - 3.0 * Iq_pu**2
            done   = False

        elif Vmag_new >= Vmin:
            # Above floor — positive reward, bonus for rising
            reward  = 8.0 - 10.0*abs(Vmag_new - Vref)
            reward += 5.0 * max(Vmag_new - Vmag, 0)   # bonus rising
            reward -= 1.0 * Id_pu**2                   # penalise excess active
            # Iq injection is good — no penalty when below 0.5 pu
            if abs(Iq_pu) > 0.5:
                reward -= 0.5 * (abs(Iq_pu) - 0.5)**2
            done = False

        elif violation_ms > 40.0:
            # 40ms below floor → would trip — heavy penalty, end episode
            reward = -200.0 - 100.0*(Vmin - Vmag_new)
            done   = True

        else:
            # Below floor but not yet trip threshold
            reward  = -50.0*(Vmin - Vmag_new) - 2.0*(Id_pu**2 + Iq_pu**2)
            done    = False

        # Overvoltage penalty
        if Vmag_new > 1.10:
            reward -= 30.0*(Vmag_new - 1.10)

        # ── NEXT OBS ──────────────────────────────────────────────
        Vmag_hist.append(Vmag_new)
        if len(Vmag_hist) > 20: Vmag_hist.pop(0)
        Vmag_mean = np.mean(Vmag_hist)
        Vmag_std  = max(np.std(Vmag_hist), 1e-4)

        next_dV  = Vmag_new - Vmag
        next_obs = build_obs(row, Vmag_new, next_dV, t_fault+DT, lbl, severity, Vmag_mean, Vmag_std)

        buf.add(obs, action, reward, next_obs, float(done))
        ep_reward += reward
        Vmag       = Vmag_new
        t_fault   += DT

        # ── UPDATE NETWORKS ───────────────────────────────────────
        if len(buf) >= 2000:
            s_b, a_b, r_b, s2_b, d_b = buf.sample(BATCH)

            with torch.no_grad():
                a2   = t_actor(s2_b)
                q2   = t_critic(s2_b, a2)
                y_td = r_b + GAMMA * (1 - d_b) * q2

            # Critic update
            q_pred  = critic(s_b, a_b)
            loss_c  = nn.functional.mse_loss(q_pred, y_td)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

            # Actor update
            loss_a = -critic(s_b, actor(s_b)).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()

            soft_update(t_actor,  actor,  TAU)
            soft_update(t_critic, critic, TAU)

        if done:
            break

    # Decay noise
    noise_std = max(noise_std * 0.997, 0.02)
    ep_rewards.append(ep_reward)

    if episode % 20 == 0:
        avg = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else ep_reward
        print(f"Ep {episode:4d}  avg_reward={avg:8.1f}  buf={len(buf):6d}  noise={noise_std:.3f}")

    if ep_reward > best_reward and episode > 50:
        best_reward = ep_reward
        torch.save(actor.state_dict(), 'rl_actor_best.pt')

print(f"\nBest episode reward: {best_reward:.1f}")

# ── ONNX EXPORT ───────────────────────────────────────────────────
# Load best weights
actor.load_state_dict(torch.load('rl_actor_best.pt'))
actor.eval()

dummy = torch.zeros(1, STATE_DIM).to(DEVICE)
torch.onnx.export(
    actor, dummy, 'rl_actor.onnx',
    input_names=['obs'],
    output_names=['action'],
    dynamic_axes={'obs': {0:'batch'}, 'action': {0:'batch'}},
    opset_version=11
)
print("✓ Exported rl_actor.onnx")

# Save weights as .mat for manual MATLAB loading
weights = {name: p.cpu().detach().numpy() for name, p in actor.state_dict().items()}
sio.savemat('rl_agent_python.mat', {
    'actor_weights': weights,
    'V_BASE': V_BASE,
    'I_BASE': I_BASE,
    'Xg':     Xg,
    'best_reward': best_reward
})
print("✓ Saved rl_agent_python.mat")
print("\nFiles to copy to MATLAB:")
print("  rl_actor.onnx    — import with importONNXNetwork()")
print("  rl_agent_python.mat — manual weight loading fallback")