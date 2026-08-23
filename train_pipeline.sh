#!/bin/bash
# MODULE 7: train_pipeline.sh
# Orchestrates an AlphaZero-style reinforcement learning loop:
# 1. Self-play data generation using the current best model (weights frozen)
# 2. Training a CANDIDATE model on that data (best model is NOT modified)
# 3. Match evaluation of candidate vs. current best model
# 4. Conditional promotion and checkpoint management

# Exit on any error
set -e

# Configuration — directory of this script, so it works from any CWD
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"
DATA_DIR="${PROJECT_DIR}/data"
BEST_MODEL="${MODEL_DIR}/best_model.pt"
CANDIDATE_MODEL="${MODEL_DIR}/candidate_model.pt"
MAX_ITERATIONS=10

# Hyperparameters (scaled for full training, adjust as needed)
NUM_SELF_PLAY_GAMES=1000
SIMULATION_COUNT=800
BATCH_SIZE=512          # must be <= replay buffer contents; sample_batch clamps anyway
TRAIN_EPOCHS=10
EVAL_GAMES=40           # 40 games is enough to resolve a >=55% score at ~1.5 sigma
EVAL_SIMS=200           # NN-eval MCTS: keep modest or evaluation takes days

# Initialize directories
mkdir -p "${MODEL_DIR}"
mkdir -p "${DATA_DIR}"

echo "=========================================================="
echo " Starting Chess Engine Training Pipeline"
echo "=========================================================="

# Bootstrap model if it does not exist — MUST use DEFAULT_ARCH so the
# candidate/best architectures always match what self_play.py builds.
if [ ! -f "${BEST_MODEL}" ]; then
    echo "No best model found. Initializing a new model with random weights..."
    python - "${BEST_MODEL}" <<'PYEOF'
import sys, torch
from neural_net import ChessNet, DEFAULT_ARCH
model = ChessNet(**DEFAULT_ARCH)
torch.save(model.state_dict(), sys.argv[1])
PYEOF
    echo "Initial model created."
fi

ITERATION=1
while [ $ITERATION -le $MAX_ITERATIONS ]; do
    echo "----------------------------------------------------------"
    echo " Iteration ${ITERATION}"
    echo "----------------------------------------------------------"

    # 1. Generate self-play data with the BEST model and train a CANDIDATE.
    #    run_self_play_training loads from model_path and saves the trained
    #    result to out_path, so the best checkpoint stays untouched until
    #    promotion actually happens.
    echo "[Phase 1] Self-play + candidate training..."
    python - "${BEST_MODEL}" "${CANDIDATE_MODEL}" "${NUM_SELF_PLAY_GAMES}" "${TRAIN_EPOCHS}" "${BATCH_SIZE}" "${SIMULATION_COUNT}" <<'PYEOF'
import sys
from self_play import run_self_play_training
run_self_play_training(
    model_path=sys.argv[1],
    out_path=sys.argv[2],
    num_games=int(sys.argv[3]),
    num_epochs=int(sys.argv[4]),
    batch_size=int(sys.argv[5]),
    num_sims=int(sys.argv[6])
)
PYEOF

    # 2. Evaluate Candidate vs. Best (genuinely different weights now).
    echo "[Phase 2] Evaluating Candidate Model vs. Current Best Model..."
    PROMOTION_CHECK=$(python - "${CANDIDATE_MODEL}" "${BEST_MODEL}" "${EVAL_GAMES}" "${EVAL_SIMS}" <<'PYEOF'
import sys, torch
from neural_net import ChessNet, DEFAULT_ARCH
from mcts import NeuralNetEvaluator
from evaluator import evaluate_candidate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

candidate_net = ChessNet(**DEFAULT_ARCH).to(device)
candidate_net.load_state_dict(torch.load(sys.argv[1], map_location=device))
candidate_eval = NeuralNetEvaluator(candidate_net, device=device)

best_net = ChessNet(**DEFAULT_ARCH).to(device)
best_net.load_state_dict(torch.load(sys.argv[2], map_location=device))
best_eval = NeuralNetEvaluator(best_net, device=device)

win_rate, elo, promoted = evaluate_candidate(
    candidate_eval,
    best_eval,
    num_games=int(sys.argv[3]),
    num_sims=int(sys.argv[4])
)

print('PROMOTE' if promoted else 'REJECT')
PYEOF
)

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
done
echo "Pipeline execution finished successfully."
