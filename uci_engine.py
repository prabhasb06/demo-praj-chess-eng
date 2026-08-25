"""
MODULE 6b: uci_engine.py
Python-based UCI Wrapper for the Praj MCTS Hybrid Chess Engine.
Integrates Alpha-Beta tactical filtering, MCTS strategic search, and
configurable difficulty settings for En Croissant and other UCI GUIs.
"""

import sys
import os
import queue
import threading
import time
import chess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mcts import MCTS, StockfishEvaluator, uci_print
from board_encoder import decode_move
from alpha_beta import AlphaBetaFilter
from difficulty import DifficultyController, select_move_with_variety

# Uses uci_print from mcts.py


def parse_time_from_go(tokens, board, max_think_time):
    """Parse time control from 'go' command tokens."""
    wtime = btime = winc = binc = movestogo = movetime = depth = None
    infinite = False
    
    for i, tok in enumerate(tokens):
        if tok == "wtime" and i + 1 < len(tokens): wtime = int(tokens[i + 1])
        elif tok == "btime" and i + 1 < len(tokens): btime = int(tokens[i + 1])
        elif tok == "winc" and i + 1 < len(tokens): winc = int(tokens[i + 1])
        elif tok == "binc" and i + 1 < len(tokens): binc = int(tokens[i + 1])
        elif tok == "movestogo" and i + 1 < len(tokens): movestogo = int(tokens[i + 1])
        elif tok == "movetime" and i + 1 < len(tokens): movetime = int(tokens[i + 1])
        elif tok == "depth" and i + 1 < len(tokens): depth = int(tokens[i + 1])
        elif tok == "nodes" and i + 1 < len(tokens): pass # Currently ignored, mapped to depth if needed
        elif tok == "infinite": infinite = True

    if infinite:
        return 31536000.0, True # 1 year, infinite flag
    
    if movetime is not None:
        return movetime / 1000.0, False
        
    if depth is not None:
        return min(max(0.8, depth * 0.25), max_think_time), False

    if board.turn == chess.WHITE and wtime is not None:
        remaining_ms = wtime
        inc_ms = winc or 0
    elif board.turn == chess.BLACK and btime is not None:
        remaining_ms = btime
        inc_ms = binc or 0
    else:
        return None, False
    
    if movestogo is not None and movestogo > 0:
        allocated_ms = remaining_ms / (movestogo + 1) + inc_ms * 0.8
    else:
        allocated_ms = remaining_ms / 30.0 + inc_ms * 0.8
    
    allocated_ms = min(allocated_ms, remaining_ms / 5.0)
    allocated_ms = max(allocated_ms, 100)
    
    return allocated_ms / 1000.0, False


def stdin_reader(input_queue):
    """Isolated thread for reading stdin to avoid blocking."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                input_queue.put(("exit", ""))
                break
            line = line.strip()
            if not line:
                continue
            input_queue.put(("cmd", line))
        except EOFError:
            input_queue.put(("exit", ""))
            break
        except Exception as e:
            uci_print(f"info string Error reading stdin: {e}")
            input_queue.put(("exit", ""))
            break


def _pick_promotion(board: chess.Board, from_square: int, to_square: int):
    """Returns the best legal promotion move matching a from/to pair, if any.

    The 4096 action space collapses q/r/b/n promotions into one index and
    decode_move defaults to queen. If the queen promotion is illegal (or the
    search selected an action whose only legal realizations are other
    promotions), pick the highest-value legal promotion instead.
    Preference order: queen, knight, rook, bishop.
    """
    preference = [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]
    candidates = [m for m in board.legal_moves
                  if m.from_square == from_square and m.to_square == to_square]
    for promo in preference:
        for m in candidates:
            if m.promotion == promo:
                return m
    return None


def execute_go(mcts, ab_filter, evaluator, difficulty, tokens, board, move_variety,
               search_id, state, abort_event):
    """Executes the deep search asynchronously to unblock the main UCI event loop.
    
    Args:
        abort_event: threading.Event — set by the main thread to signal abort.
                     Cleared only by this function when it starts a fresh search.
    """
    # Only clear the abort flag if this search hasn't already been superseded
    # AND no 'stop' is pending (a stop arriving between submit and execution
    # must be honoured, not wiped — the search below will then finish with
    # zero simulations and emit a fallback bestmove immediately).
    if not _superseded(search_id, state) and not state.get("stop_requested", False):
        abort_event.clear()
        mcts.abort_search = False
    mcts.is_searching = True
    try:
        if board.is_game_over(claim_draw=True) or not board.legal_moves:
            if not _superseded(search_id, state):
                uci_print("bestmove 0000")
            return

        params = difficulty.get_params()
        
        allocated_time, is_infinite = parse_time_from_go(tokens, board, params.max_think_time)
        if allocated_time is not None:
            max_time = allocated_time if is_infinite else min(allocated_time, params.max_think_time)
        else:
            max_time = params.max_think_time
        
        search_start = time.time()
        
        # =============================================
        # PHASE 1: Tactical Analysis / Prior Seeding
        # =============================================
        # For maximum strength (Level >= 8 or Stockfish evaluator), Stockfish-guided MCTS
        # provides deep, accurate tactical and strategic evaluation. Dedicate 100% time to MCTS.
        use_ab = params.level < 8 and not isinstance(evaluator, StockfishEvaluator)
        ab_candidates = None
        ab_elapsed = 0.0

        if use_ab:
            ab_result = ab_filter.analyze(board)
            ab_elapsed = time.time() - search_start
            
            # Check if aborted during AB phase
            if _superseded(search_id, state):
                return
            if abort_event.is_set() and not state.get("stop_requested", False):
                return
            
            if ab_result.best_move:
                uci_print(f"info string AB filter: score={ab_result.score_cp}cp "
                           f"tactical={ab_result.is_tactical} "
                           f"candidates={len(ab_result.candidate_moves)} "
                           f"time={int(ab_elapsed*1000)}ms")
            
            # Only play immediately on verified checkmate
            if ab_result.score_cp >= 90000 and ab_result.best_move in board.legal_moves:
                board_test = board.copy()
                board_test.push(ab_result.best_move)
                if board_test.is_checkmate():
                    if not _superseded(search_id, state):
                        uci_print(f"bestmove {ab_result.best_move.uci()}")
                    return
            
            ab_candidates = ab_result.candidate_moves if ab_result.candidate_moves else None

        # =============================================
        # PHASE 2: MCTS Strategic Search
        # =============================================
        mcts.num_simulations = 1_000_000_000 if is_infinite else params.num_simulations
        mcts.c_puct = params.c_puct
        evaluator.depth = params.stockfish_depth
        if hasattr(evaluator, "multipv"):
            evaluator.multipv = params.stockfish_multipv
        
        remaining_time = max_time - ab_elapsed
        if remaining_time < 0.05:
            remaining_time = 0.05
        # Reserve a small safety margin for move selection + output.
        deadline = time.time() + remaining_time * 0.95
        
        visit_counts = mcts.get_visit_counts(
            board,
            add_noise=params.add_noise if move_variety else False,
            noise_epsilon=params.noise_epsilon,
            noise_alpha=params.noise_alpha,
            ab_candidates=ab_candidates,
            deadline=deadline
        )

        # Check again after MCTS returns
        if _superseded(search_id, state):
            return
        if abort_event.is_set() and not state.get("stop_requested", False):
            return
        
        # =============================================
        # PHASE 3: Move Selection with Difficulty Control
        # =============================================
        if move_variety:
            action = select_move_with_variety(visit_counts, params, board)
        else:
            action = int(np.argmax(visit_counts))
        
        best_move = decode_move(action, board)
        
        if best_move not in board.legal_moves:
            # The 4096 action space collapses under-promotions into the queen
            # promotion index; try other promotion variants of the same move first.
            promo_alt = _pick_promotion(board, action // 64, action % 64)
            if promo_alt is not None:
                best_move = promo_alt
            else:
                uci_print(f"info string Warning: selected illegal move "
                           f"{best_move.uci()}, falling back")
                legal = list(board.legal_moves)
                best_move = legal[0] if legal else chess.Move.null()
        
        total_time = time.time() - search_start
        if not _superseded(search_id, state):
            uci_print(f"info string Total search: {int(total_time*1000)}ms "
                       f"(AB={int(ab_elapsed*1000)}ms, "
                       f"MCTS={int((total_time-ab_elapsed)*1000)}ms)")
            uci_print(f"bestmove {best_move.uci()}")
        
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        uci_print(f"info string Error in go command: {e}")
        
        # Only print fallback bestmove if this search is still current
        if not _superseded(search_id, state):
            if list(board.legal_moves):
                fallback_move = list(board.legal_moves)[0]
                uci_print(f"bestmove {fallback_move.uci()}")
            else:
                uci_print("bestmove 0000")
    finally:
        mcts.is_searching = False


def _superseded(search_id, state):
    """True if a newer 'go' command arrived while this search was running."""
    return search_id != state["current_search_id"]


def _parse_setoption(tokens):
    """Parses a 'setoption' token list into (name, value_or_None).

    Handles multi-word option names and the legal UCI case where no value
    is provided (e.g. button-type options).
    """
    try:
        name_idx = tokens.index("name") + 1
    except ValueError:
        return None, None

    if "value" in tokens:
        value_idx = tokens.index("value")
        opt_name = " ".join(tokens[name_idx:value_idx])
        opt_value = " ".join(tokens[value_idx + 1:])
        return opt_name, opt_value

    opt_name = " ".join(tokens[name_idx:])
    return opt_name, None


def main():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    stockfish_path = os.path.join(base_dir, "stockfish.exe")

    difficulty = DifficultyController(level=10)
    params = difficulty.get_params()

    evaluator = StockfishEvaluator(
        stockfish_path=stockfish_path,
        depth=params.stockfish_depth
    )
    mcts = MCTS(
        evaluator=evaluator,
        c_puct=params.c_puct,
        num_simulations=params.num_simulations
    )
    mcts.abort_search = False
    mcts.is_searching = False

    # Thread-safe abort signal: the main thread sets it, the worker checks it.
    abort_event = threading.Event()

    ab_filter = AlphaBetaFilter(depth=params.ab_depth)
    
    # GUIs pipe stdout (block-buffered by default). Force line buffering so
    # protocol lines are never stuck in an 8 KB buffer, even if some code
    # path prints without flushing.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    input_queue = queue.Queue()
    
    # Isolate stdin reading
    reader_thread = threading.Thread(target=stdin_reader, args=(input_queue,), daemon=True)
    reader_thread.start()

    # Worker pool for async Stockfish/MCTS heavy calculations.
    # Single worker: searches run one at a time; superseded searches are
    # detected via search IDs so exactly one bestmove is printed per 'go'.
    worker_pool = ThreadPoolExecutor(max_workers=1)

    # Search bookkeeping: every 'go' increments the counter; only the search
    # whose ID matches the latest counter may print bestmove.
    state = {"current_search_id": 0, "stop_requested": False}
    
    board = chess.Board()
    move_variety = False

    while True:
        try:
            msg_type, line = input_queue.get()
            if msg_type == "exit":
                break
                
            tokens = line.split()
            if not tokens:
                continue
            cmd = tokens[0]

            if cmd == "uci":
                uci_print("id name Praj MCTS v2.0")
                uci_print("id author Praj")
                uci_print("option name Difficulty type spin default 10 min 1 max 10")
                uci_print("option name MoveVariety type check default false")
                uci_print("option name StockfishPath type string default stockfish.exe")
                uci_print("uciok")
                
            elif cmd == "isready":
                uci_print("readyok")
                
            elif cmd == "setoption":
                opt_name, opt_value = _parse_setoption(tokens)
                
                if opt_name == "Difficulty" and opt_value is not None:
                    # Safe to apply mid-search (plain attribute reads used by
                    # the next simulation). Do NOT supersede the in-flight
                    # search here, or it would go silent without a bestmove.
                    try:
                        level = int(opt_value)
                    except ValueError:
                        level = difficulty.level
                    difficulty.set_level(level)
                    params = difficulty.get_params()
                    
                    mcts.num_simulations = params.num_simulations
                    mcts.c_puct = params.c_puct
                    evaluator.depth = params.stockfish_depth
                    if hasattr(evaluator, "multipv"):
                        evaluator.multipv = params.stockfish_multipv
                    ab_filter.depth = params.ab_depth
                    
                    uci_print(f"info string Difficulty set to {level} "
                               f"(sims={params.num_simulations}, "
                               f"temp={params.temperature}, "
                               f"ab_depth={params.ab_depth})")
                    
                elif opt_name == "MoveVariety" and opt_value is not None:
                    move_variety = opt_value.lower() == "true"
                    uci_print(f"info string MoveVariety set to {move_variety}")
                    
                elif opt_name == "StockfishPath" and opt_value:
                    abort_event.set()
                    mcts.abort_search = True
                    state["current_search_id"] += 1
                    # Security: validate that the path points to a Stockfish binary
                    basename = os.path.basename(opt_value).lower()
                    allowed_names = {"stockfish", "stockfish.exe"}
                    if basename not in allowed_names:
                        uci_print(f"info string Rejected StockfishPath: basename '{basename}' "
                                   f"must be 'stockfish' or 'stockfish.exe'")
                    elif not os.path.isfile(opt_value):
                        uci_print(f"info string Rejected StockfishPath: file not found at '{opt_value}'")
                    else:
                        stockfish_path = opt_value
                        evaluator.close()
                        evaluator = StockfishEvaluator(
                            stockfish_path=stockfish_path,
                            depth=params.stockfish_depth
                        )
                        # Cached evaluations came from the old evaluator; drop them.
                        mcts.evaluator = evaluator
                        mcts.clear_cache()
                        uci_print(f"info string Re-initialized Stockfish at {stockfish_path}")

            elif cmd == "ucinewgame":
                abort_event.set()
                mcts.abort_search = True
                board = chess.Board()
                mcts.transposition_table.clear()
                
            elif cmd == "position":
                abort_event.set()
                mcts.abort_search = True
                try:
                    if len(tokens) < 2:
                        continue
                    
                    temp_board = None
                    moves_idx = -1
                    if tokens[1] == "startpos":
                        temp_board = chess.Board()
                        if "moves" in tokens:
                            moves_idx = tokens.index("moves")
                    elif tokens[1] == "fen":
                        fen_tokens = []
                        idx = 2
                        while idx < len(tokens) and tokens[idx] != "moves":
                            fen_tokens.append(tokens[idx])
                            idx += 1
                        fen_str = " ".join(fen_tokens)
                        temp_board = chess.Board(fen_str)
                        if idx < len(tokens) and tokens[idx] == "moves":
                            moves_idx = idx
                    
                    if temp_board is None:
                        uci_print(f"info string Unrecognised position format: {tokens[1]}; "
                                  f"reverting to startpos")
                        temp_board = chess.Board()

                    if moves_idx != -1:
                        for move_str in tokens[moves_idx+1:]:
                            move = chess.Move.from_uci(move_str)
                            temp_board.push(move)
                    
                    board = temp_board
                except Exception as e:
                    # Never keep a half-updated/stale board: the GUI believes
                    # its position command succeeded. Revert to a known state.
                    uci_print(f"info string Error in position command ({e}); "
                              f"reverting to startpos")
                    board = chess.Board()
                        
            elif cmd == "go":
                if "ponder" in tokens:
                    continue
                # Clear stop flag
                state["stop_requested"] = False
                # Invalidate any in-flight search: it must not print bestmove.
                abort_event.set()
                mcts.abort_search = True
                state["current_search_id"] += 1
                my_search_id = state["current_search_id"]
                # Offload heavy intense calculations immediately
                worker_pool.submit(
                    execute_go, 
                    mcts, 
                    ab_filter, 
                    evaluator, 
                    difficulty, 
                    tokens, 
                    board.copy(), 
                    move_variety,
                    my_search_id,
                    state,
                    abort_event
                )
                
            elif cmd == "quit":
                abort_event.set()
                mcts.abort_search = True
                break
                
            elif cmd == "stop":
                state["stop_requested"] = True
                abort_event.set()
                mcts.abort_search = True

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            uci_print(f"info string Error in main loop: {e}")

    worker_pool.shutdown(wait=False)
    evaluator.close()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
