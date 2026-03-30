# ============================================================
# BLOCK 1: Import libraries
# ============================================================
import torch
import torch.nn as nn


# ============================================================
# BLOCK 2: Define DQN model
# ============================================================
class SumoDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Neural network for approximating Q(s, a)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure float type
        x = x.float()

        # Add batch dimension for single state
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Return raw Q-values
        return self.net(x)