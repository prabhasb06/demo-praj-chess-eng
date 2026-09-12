"""

MODULE 2: neural_net.py
Implements the deep neural network backbone (ResNet with Squeeze-and-Excitation blocks)
and the multi-head outputs (Policy, Value, and Moves-Left).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from board_encoder import POLICY_SIZE

# Single source of truth for the network architecture. Training pipeline,
# self-play and evaluation must all construct ChessNet with these values so
# checkpoints always load. Input is the 18-plane encoding from board_encoder.
DEFAULT_ARCH = {"num_blocks": 10, "filters": 128, "in_channels": 18}


def load_checkpoint(model, path, device='cpu'):
    """Load weights, migrating the old 4096-action policy head in memory.

    New underpromotion rows start from the matching queen-promotion row.
    The original checkpoint is never overwritten; further training is needed
    to distinguish promotion choices in a migrated model.
    """
    from board_encoder import decode_move
    import chess
    state = torch.load(path, map_location=device, weights_only=True)
    weight = state.get('fc_policy.weight')
    if weight is not None and weight.shape[0] == 4096:
        indices = list(range(4096))
        for action in range(4096, POLICY_SIZE):
            move = decode_move(action, chess.Board())
            indices.append(move.from_square * 64 + move.to_square)
        index = torch.tensor(indices, device=weight.device)
        state['fc_policy.weight'] = weight[index].clone()
        state['fc_policy.bias'] = state['fc_policy.bias'][index].clone()
    model.load_state_dict(state)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature maps dynamically by modeling channel interdependencies.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze: global average pooling (B, C, 1, 1)
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid() # Excitation: channel-wise weights in [0, 1]
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale

class ResidualBlockSE(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Contains two 3x3 convolutions, Batch Normalization, ReLU, and an SE Block.
    """
    def __init__(self, channels):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ChessNet(nn.Module):
    """
    The main Chess Neural Network architecture.
    Input shape: (Batch, 18, 8, 8)
    Outputs:
      - Policy Logits: (Batch, 4096)
      - Value: (Batch, 1) in [-1, 1] (tanh)
      - Moves Left: (Batch, 1) (relu)
    """
    def __init__(self, num_blocks=10, filters=128, in_channels=18):
        super(ChessNet, self).__init__()
        self.num_blocks = num_blocks
        self.filters = filters

        # Convolutional Block (Entry)
        self.conv_entry = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_entry = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

        # Residual Tower
        self.res_tower = nn.ModuleList([ResidualBlockSE(filters) for _ in range(num_blocks)])

        # Policy Head
        self.conv_policy = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Value Head (tanh)
        self.conv_value = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

        # Moves Left Head (regression)
        self.conv_moves = nn.Conv2d(filters, 16, kernel_size=1, bias=False)
        self.bn_moves = nn.BatchNorm2d(16)
        self.fc_moves1 = nn.Linear(16 * 8 * 8, 128)
        self.fc_moves2 = nn.Linear(128, 1)

    def forward(self, x):
        # Entry block
        out = self.conv_entry(x)
        out = self.bn_entry(out)
        out = self.relu(out)

        # Residual blocks
        for res_block in self.res_tower:
            out = res_block(out)

        # Policy Head
        p = self.conv_policy(out)
        p = self.bn_policy(p)
        p = self.relu(p)
        p = torch.flatten(p, start_dim=1)
        policy_logits = self.fc_policy(p)

        # Value Head
        v = self.conv_value(out)
        v = self.bn_value(v)
        v = self.relu(v)
        v = torch.flatten(v, start_dim=1)
        v = self.fc_value1(v)
        v = self.relu(v)
        value = torch.tanh(self.fc_value2(v))

        # Moves Left Head
        m = self.conv_moves(out)
        m = self.bn_moves(m)
        m = self.relu(m)
        m = torch.flatten(m, start_dim=1)
        m = self.fc_moves1(m)
        m = self.relu(m)
        moves_left = F.softplus(self.fc_moves2(m))  # non-negative; softplus avoids dead gradients unlike ReLU

        return policy_logits, value, moves_left

def compute_loss(policy_logits, value, moves_left, target_policy, target_value, target_moves_left):
    """
    Computes the combined loss:
    Loss = CrossEntropy(policy_logits, target_policy)
           + MSE(value, target_value)
           + lambda_ml * MSE(moves_left, target_moves_left)
    """
    # Policy loss (cross-entropy with target probability distribution)
    # Using log_softmax and negative log likelihood for stability
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(target_policy * log_probs, dim=-1).mean()

    # Value loss (mean squared error)
    value_loss = F.mse_loss(value.squeeze(-1), target_value)

    # Moves Left loss (scaled by a factor, e.g., 0.1 to avoid dominating)
    moves_loss = F.mse_loss(moves_left.squeeze(-1), target_moves_left)

    total_loss = policy_loss + value_loss + 0.1 * moves_loss

    return total_loss, policy_loss, value_loss, moves_loss
