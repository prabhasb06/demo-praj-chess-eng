"""
MODULE 4: self_play.py
Handles self-play data generation and neural network training loops.
Uses a circular Replay Buffer to cache board states, MCTS policies, and game outcomes.
"""

import os
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.optim as optim
import chess

from board_encoder import POLICY_SIZE, encode_board, decode_move, encode_move_for_perspective
from neural_net import ChessNet, DEFAULT_ARCH, compute_loss, load_checkpoint
from mcts import MCTS, NeuralNetEvaluator

class ReplayBuffer:
    """
    A circular replay buffer storing historical board states, MCTS policy targets,
    and game outcome values.

    Memory usage: ~2 GB at capacity=100,000 (each entry is ~20 KB).
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float, moves_left: float):
        """Adds a single state-policy-value-moves_left transition tuple to the buffer."""
        self.buffer.append((state, policy, value, moves_left))

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a mini-batch of transitions, formatted as PyTorch Tensors."""
        if batch_size <= 0 or not self.buffer:
            raise ValueError('A positive batch size and nonempty replay buffer are required')
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        states, policies, values, moves_left = zip(*batch)
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        policies_t = torch.tensor(np.array(policies), dtype=torch.float32)
        values_t = torch.tensor(np.array(values), dtype=torch.float32)
        moves_left_t = torch.tensor(np.array(moves_left), dtype=torch.float32)
        
        return states_t, policies_t, values_t, moves_left_t

    def __len__(self):
        return len(self.buffer)


def _mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a 4096-element policy vector from original action space to mirrored action space."""
    mirrored = np.zeros(POLICY_SIZE, dtype=np.float32)
    for action in range(POLICY_SIZE):
        if policy[action] > 0:
            if action >= 4096:
                move = decode_move(action, chess.Board())
                mirrored[encode_move_for_perspective(move, True)] = policy[action]
                continue
            from_sq = chess.square_mirror(action // 64)
            to_sq = chess.square_mirror(action % 64)
            mirrored[from_sq * 64 + to_sq] = policy[action]
    return mirrored


def play_game(mcts: MCTS, temp_threshold: int = 30) -> List[Tuple[np.ndarray, np.ndarray, float, int]]:
    """
    Generates a single self-play game.
    Runs MCTS at each turn to calculate policies.
    Applies temperature scheduling:
      - temp = 1.0 for the first temp_threshold moves (for exploration)
      - temp = 0.0 thereafter (for exploitation)
    Returns a list of (board_tensor, MCTS_policy, current_player_turn, moves_from_start) tuples.
    Note: game outcome values are backpropagated once the game finishes.
    """
    board = chess.Board()
    game_history = []
    
    move_count = 0
    while not board.is_game_over() and move_count < 200:  # Cap at 200 moves to prevent infinite games
        # Temperature schedule
        temp = 1.0 if move_count < temp_threshold else 0.0
        
        # Run search to get move probabilities (in original/unmirrored action space)
        # MCTS handles board perspective internally
        policy = mcts.get_move_probs(board, temp=temp)
        
        # Save encoding of the current state and policy
        # Always encode board from active player's perspective
        encoded_state = encode_board(board)

        # Mirror the policy for Black so it aligns with the mirrored board encoding.
        # The NN sees a mirrored board for Black, so training targets must also be mirrored.
        is_black = (board.turn == chess.BLACK)
        training_policy = _mirror_policy(policy) if is_black else policy

        game_history.append((encoded_state, training_policy, board.turn, move_count))
        
        # Select action (in original action space for board.push)
        action = np.random.choice(POLICY_SIZE, p=policy)
        move = decode_move(action, board)
        
        # Execute move
        board.push(move)
        move_count += 1

    # Backpropagate game outcome
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        z = 0.0  # Draw
    else:
        # 1.0 if winner is White, -1.0 if winner is Black
        z = 1.0 if outcome.winner == chess.WHITE else -1.0

    # Format training examples (state, policy, value, moves_left)
    train_examples = []
    total_moves = len(game_history)
    for state, policy, turn, idx in game_history:
        # Outcome relative to the player whose turn it was
        # If it was White's turn, value is z. If Black's turn, value is -z.
        value = z if turn == chess.WHITE else -z
        moves_left = float(total_moves - idx)
        train_examples.append((state, policy, value, moves_left))

    return train_examples


def train_step(model: torch.nn.Module, optimizer: optim.Optimizer, 
               states: torch.Tensor, policies: torch.Tensor, 
               values: torch.Tensor, moves_left: torch.Tensor, 
               device: str = "cpu") -> float:
    """Performs a single gradient descent optimization step on a mini-batch."""
    model.train()
    
    # Move batch to device
    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)
    moves_left = moves_left.to(device)

    # Reset gradients
    optimizer.zero_grad()

    # Forward pass
    policy_logits, value, pred_moves = model(states)

    # Loss calculation
    loss, p_loss, v_loss, m_loss = compute_loss(
        policy_logits, value, pred_moves, 
        policies, values, moves_left
    )

    # Backward pass & Optimizer step
    loss.backward()
    optimizer.step()

    return loss.item()


def run_self_play_training(model_path: str = "best_model.pt",
                           out_path: Optional[str] = None,
                           num_games: int = 5,
                           num_epochs: int = 2, batch_size: int = 16,
                           num_sims: int = 10):
    """
    Orchestrates a small, lightweight self-play and training loop.
    Default arguments are scaled down to run fast and safely on any machine.

    Args:
        model_path: Checkpoint to load weights from (created if missing).
        out_path: Where to save the trained model. Defaults to model_path.
                  The training pipeline passes a separate path so the
                  candidate differs from the incumbent best model.
        num_games: Number of self-play games to generate.
        num_epochs: Training epochs over sampled batches.
        batch_size: Mini-batch size (clamped to the buffer size).
        num_sims: MCTS simulations per move during self-play.
    """
    if min(num_games, num_epochs, batch_size, num_sims) <= 0:
        raise ValueError('Training counts and batch size must be positive')
    if out_path is None:
        out_path = model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting self-play loop on device: {device} (scaled parameters for safety).")

    # Initialize network using the shared architecture constants so every
    # component (pipeline, self-play, evaluation) builds compatible models.
    if os.path.exists(model_path):
        print(f"Loading existing network from {model_path}")
        model = ChessNet(**DEFAULT_ARCH)
        load_checkpoint(model, model_path, device)
    else:
        print(f"Creating a new {DEFAULT_ARCH['num_blocks']}-block model.")
        model = ChessNet(**DEFAULT_ARCH)
    model.to(device)

    evaluator = NeuralNetEvaluator(model, device=device)
    mcts = MCTS(evaluator=evaluator, c_puct=1.5, num_simulations=num_sims)
    buffer = ReplayBuffer(capacity=100000)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Data generation
    for g in range(num_games):
        print(f"Generating game {g+1}/{num_games}...")
        game_data = play_game(mcts, temp_threshold=15)
        for state, policy, value, ml in game_data:
            buffer.add(state, policy, value, ml)

    # Network training
    if len(buffer) > 0:
        print(f"Training on buffer of size {len(buffer)} for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            examples = list(buffer.buffer)
            random.shuffle(examples)
            losses = []
            for start in range(0, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                states, policies, values, moves = [torch.as_tensor(np.array(items), dtype=torch.float32)
                                                   for items in zip(*batch)]
                losses.append(train_step(model, optimizer, states, policies, values, moves, device=device))
            loss_val = sum(losses) / len(losses)
            print(f"  Epoch {epoch+1}/{num_epochs} Loss: {loss_val:.4f}")

        # Save checkpoint to out_path (NOT necessarily overwriting model_path)
        torch.save(model.state_dict(), out_path)
        print(f"Model saved to {out_path}")
    else:
        print("Not enough data to train. Increase num_games or batch_size.")

    print("Lightweight self-play cycle complete.")
