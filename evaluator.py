"""
MODULE 5: evaluator.py
Manages matches between candidate networks and the current best network.
Computes Elo difference and determines promotion based on win-rate benchmarks.
"""

import math
import chess
import numpy as np
from typing import Tuple
from mcts import MCTS
from board_encoder import decode_move

def play_match_game(mcts_white: MCTS, mcts_black: MCTS) -> float:
    """
    Plays a single match game between two MCTS engines.
    Temperature is set to 0.0 for deterministic play.
    Returns:
      1.0 if White wins
      0.5 if Draw
      0.0 if Black wins
    """
    board = chess.Board()
    move_count = 0
    
    while not board.is_game_over() and move_count < 250:
        # Determine player
        active_mcts = mcts_white if board.turn == chess.WHITE else mcts_black
        
        # Select best move (temperature 0.0 for competitive play)
        policy = active_mcts.get_move_probs(board, temp=0.0)
        action = int(np.argmax(policy)) if hasattr(policy, 'ndim') else int(policy.argmax())
        move = decode_move(action, board)
        
        # Apply move
        board.push(move)
        move_count += 1

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.5  # Draw
    
    # 1.0 for White win, 0.0 for Black win
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def evaluate_candidate(candidate_evaluator, best_evaluator, num_games=400, num_sims=3200) -> Tuple[float, float, bool]:
    """
    Plays a match: Candidate vs. Current Best.
    Candidate plays White for half the games and Black for the other half.
    Returns:
      - win_rate: Candidate's total score divided by num_games (Wins=1, Draws=0.5, Losses=0)
      - delta_elo: Elo rating difference
      - promoted: True if win_rate >= 0.55, False otherwise
    """
    candidate_mcts = MCTS(candidate_evaluator, c_puct=1.5, num_simulations=num_sims)
    best_mcts = MCTS(best_evaluator, c_puct=1.5, num_simulations=num_sims)

    candidate_score = 0.0
    
    print(f"Starting evaluation match: Candidate vs. Current Best ({num_games} games, {num_sims} simulations/move).")

    for i in range(num_games):
        # Alternate sides
        if i % 2 == 0:
            # Candidate is White
            score = play_match_game(candidate_mcts, best_mcts)
            candidate_score += score
            winner_str = "Candidate(W)" if score == 1.0 else "Best(B)" if score == 0.0 else "Draw"
        else:
            # Candidate is Black
            score = play_match_game(best_mcts, candidate_mcts)
            # Candidate score is 1.0 - White score (since Candidate is Black)
            score_black = 1.0 - score
            candidate_score += score_black
            winner_str = "Candidate(B)" if score_black == 1.0 else "Best(W)" if score_black == 0.0 else "Draw"
        
        print(f"  Game {i+1}/{num_games}: {winner_str} (Candidate Score: {candidate_score}/{i+1})")

    win_rate = candidate_score / num_games
    promoted = win_rate >= 0.55

    # Compute Elo difference
    # Safe guard division by zero or log of negative
    safe_win_rate = max(1e-4, min(1.0 - 1e-4, win_rate))
    delta_elo = -400.0 * math.log10((1.0 - safe_win_rate) / safe_win_rate)

    print("Evaluation Complete!")
    print(f"  Candidate Win Rate: {win_rate:.2%} ({candidate_score}/{num_games} points)")
    print(f"  Elo Difference: {delta_elo:+.1f}")
    print(f"  Promotion Status: {'PROMOTED' if promoted else 'REJECTED'}")

    return win_rate, delta_elo, promoted

# Dummy wrapper to run a quick test from the CLI
if __name__ == "__main__":
    from neural_net import ChessNet
    from mcts import NeuralNetEvaluator

    print("Running quick evaluator validation...")
    model_c = ChessNet(num_blocks=2, filters=16)
    model_b = ChessNet(num_blocks=2, filters=16)
    
    eval_c = NeuralNetEvaluator(model_c)
    eval_b = NeuralNetEvaluator(model_b)

    # 2 games with 5 sims each (very fast, safe)
    win_rate, elo, promoted = evaluate_candidate(eval_c, eval_b, num_games=2, num_sims=5)
