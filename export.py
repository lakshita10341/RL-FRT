import torch
import torch.nn as nn
import scipy.io as sio

STATE_DIM = 12
ACTION_DIM = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Recreate Actor (MUST match training)
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, ACTION_DIM), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

# Load model
actor = Actor().to(DEVICE)
actor.load_state_dict(torch.load('rl_actor_best.pt', map_location=DEVICE))
actor.eval()

# Export ONNX
dummy = torch.zeros(1, STATE_DIM).to(DEVICE)
torch.onnx.export(
    actor, dummy, 'rl_actor.onnx',
    input_names=['obs'],
    output_names=['action'],
    dynamic_axes={'obs': {0:'batch'}, 'action': {0:'batch'}},
    opset_version=11
)

print("✓ ONNX export done")

# Save MAT file (important for MATLAB fallback)
weights = {name: p.cpu().detach().numpy() for name, p in actor.state_dict().items()}
sio.savemat('rl_agent_python.mat', {
    'actor_weights': weights
})

print("✓ MAT file saved")