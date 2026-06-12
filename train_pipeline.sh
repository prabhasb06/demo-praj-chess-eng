#!/bin/bash
# MODULE 7: train_pipeline.sh
# Orchestrates the AlphaZero reinforcement learning loop:
# 1. Asynchronous Self-Play data generation using current best model
# 2. Model training on accumulated replay buffer data
# 3. Match evaluation of candidate vs. current best model
# 4. Conditional promotion and checkpoint management

# Exit on any error
set -e

# Configuration
PROJECT_DIR="c:/Users/prabh/chess_engine"
MODEL_DIR="${PROJECT_DIR}/models"
DATA_DIR="${PROJECT_DIR}/data"
BEST_MODEL="${MODEL_DIR}/best_model.pt"
CANDIDATE_MODEL="${MODEL_DIR}/candidate_model.pt"

# Hyperparameters (scaled for full training, adjust as needed)
NUM_SELF_PLAY_GAMES=1000
SIMULATION_COUNT=800
BATCH_SIZE=2048
TRAIN_EPOCHS=10
EVAL_GAMES=400
EVAL_SIMS=3200

# Initialize directories
mkdir -p "${MODEL_DIR}"
mkdir -p "${DATA_DIR}"

echo "=========================================================="
echo " Starting Antigravity Chess Engine Training Pipeline"
echo "=========================================================="

# Bootstrap model if it does not exist
if [ ! -f "${BEST_MODEL}" ]; then
    echo "No best model found. Initializing a new model with random weights..."
    python -c "
import torch
from neural_net import ChessNet
model = ChessNet(num_blocks=20, filters=256)
torch.save(model.state_dict(), '${BEST_MODEL}')
"
    echo "Initial model created."
fi

# Infinite training loop
ITERATION=1
while true; do
    echo "----------------------------------------------------------"
    echo " Iteration ${ITERATION}"
    echo "----------------------------------------------------------"
    
    # 1. Generate Self-Play Data
    echo "[Phase 1] Launching parallel self-play data generation..."
    # In a full run, we generate self-play games and save them as PGNs or numpy tensors.
    # We run the self_play.py module with appropriate args:
    python -c "
import os
from self_play import run_self_play_training
run_self_play_training(
    model_path='${BEST_MODEL}',
    num_games=${NUM_SELF_PLAY_GAMES},
    num_epochs=${TRAIN_EPOCHS},
    batch_size=${BATCH_SIZE},
    num_sims=${SIMULATION_COUNT}
)
"
    # Copy the newly trained model to candidate
    cp "${BEST_MODEL}" "${CANDIDATE_MODEL}"
    
    # 2. Evaluate Candidate vs. Best
    echo "[Phase 2] Evaluating Candidate Model vs. Current Best Model..."
    # Play a match of 400 games. If candidate wins >= 55% of points, promote.
    # The evaluation output can be checked to decide promotion.
    PROMOTION_CHECK=$(python -c "
import torch
from neural_net import ChessNet
from mcts import NeuralNetEvaluator
from evaluator import evaluate_candidate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load candidate and best models
candidate_net = ChessNet(num_blocks=20, filters=256).to(device)
candidate_net.load_state_dict(torch.load('${CANDIDATE_MODEL}', map_location=device))
candidate_eval = NeuralNetEvaluator(candidate_net, device=device)

best_net = ChessNet(num_blocks=20, filters=256).to(device)
best_net.load_state_dict(torch.load('${BEST_MODEL}', map_location=device))
best_eval = NeuralNetEvaluator(best_net, device=device)

# Run evaluation with fewer games/sims for local test cycles if needed,
# or full parameters for production.
win_rate, elo, promoted = evaluate_candidate(
    candidate_eval, 
    best_eval, 
    num_games=${EVAL_GAMES}, 
    num_sims=${EVAL_SIMS}
)

if promoted:
    print('PROMOTE')
else:
    print('REJECT')
")

    # 3. Handle Promotion
    if [ "${PROMOTION_CHECK}" = "PROMOTE" ]; then
        echo "[Phase 3] SUCCESS: Candidate promoted! Updating best model."
        cp "${CANDIDATE_MODEL}" "${BEST_MODEL}"
        # Keep a history of promoted checkpoints
        cp "${BEST_MODEL}" "${MODEL_DIR}/model_checkpoint_iter_${ITERATION}.pt"
    else
        echo "[Phase 3] REJECTED: Candidate did not meet promotion threshold (55%). Retaining current best model."
    fi

    echo "Iteration ${ITERATION} complete. Starting next cycle..."
    ITERATION=$((ITERATION + 1))
    
    # Break to prevent running continuously on user's PC (user safeguard)
    echo "Safeguard exit: Loop interrupted to prevent high resource consumption."
    break
done
echo "Pipeline execution finished successfully."
