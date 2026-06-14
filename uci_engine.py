"""
MODULE 6b: uci_engine.py
Python-based UCI Wrapper for the Antigravity Chess Engine.
Allows running the engine directly in Chess GUIs (e.g. Arena, Cute Chess) without C++ compilation.
Exposes advanced MCTS configurations directly to GUI sliders and settings (setoption support).
"""

import sys
import os
import queue
import threading
import chess
import numpy as np
from mcts import MCTS, StockfishEvaluator
from board_encoder import decode_move

def stdin_reader(mcts, input_queue):
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                input_queue.put(("exit", ""))
                break
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if tokens:
                cmd = tokens[0]
                if cmd in ["stop", "quit", "position", "go"]:
                    mcts.abort_search = True
            input_queue.put(("cmd", line))
        except Exception:
            input_queue.put(("exit", ""))
            break

def main():
    # Resolve Stockfish path relative to the executable or script directory
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    stockfish_path = os.path.join(base_dir, "stockfish.exe")

    stockfish_depth = 5
    stockfish_time_limit = 0.005
    num_simulations = 400
    c_puct = 1.5

    # Initialize the Stockfish-guided MCTS evaluator
    evaluator = StockfishEvaluator(stockfish_path=stockfish_path, depth=stockfish_depth, time_limit=stockfish_time_limit)
    mcts = MCTS(evaluator=evaluator, c_puct=c_puct, num_simulations=num_simulations)
    mcts.abort_search = False
    
    input_queue = queue.Queue()
    reader_thread = threading.Thread(target=stdin_reader, args=(mcts, input_queue), daemon=True)
    reader_thread.start()

    board = chess.Board()

    # UCI input loop
    while True:
        try:
            msg_type, line = input_queue.get()
            if msg_type == "exit":
                break
                
            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                print("id name Praj MCTS")
                print("id author Praj")
                
                # Expose advanced configurations to the GUI
                print("option name NumSimulations type spin default 400 min 1 max 10000")
                print("option name CPuct type string default 1.5")
                print("option name StockfishPath type string default stockfish.exe")
                print("option name StockfishDepth type spin default 5 min 1 max 20")
                print("option name StockfishTimeLimit type string default 0.005")
                print("uciok")
                sys.stdout.flush()
                
            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()
                
            elif cmd == "setoption":
                # Syntax: setoption name <name> value <value>
                if "name" in tokens and "value" in tokens:
                    name_idx = tokens.index("name") + 1
                    value_idx = tokens.index("value") + 1
                    
                    # Reconstruct name and value in case of spaces
                    opt_name = ""
                    for i in range(name_idx, tokens.index("value")):
                        opt_name += tokens[i]
                    
                    opt_value = " ".join(tokens[value_idx:])
                    
                    # Update variables dynamically
                    if opt_name == "NumSimulations":
                        num_simulations = int(opt_value)
                        mcts.num_simulations = num_simulations
                        print(f"info string Updated NumSimulations to {num_simulations}", file=sys.stderr)
                    elif opt_name == "CPuct":
                        c_puct = float(opt_value)
                        mcts.c_puct = c_puct
                        print(f"info string Updated c_puct to {c_puct}", file=sys.stderr)
                    elif opt_name == "StockfishDepth":
                        stockfish_depth = int(opt_value)
                        evaluator.depth = stockfish_depth
                        print(f"info string Updated StockfishDepth to {stockfish_depth}", file=sys.stderr)
                    elif opt_name == "StockfishTimeLimit":
                        stockfish_time_limit = float(opt_value)
                        evaluator.time_limit = stockfish_time_limit
                        print(f"info string Updated StockfishTimeLimit to {stockfish_time_limit}", file=sys.stderr)
                    elif opt_name == "StockfishPath":
                        stockfish_path = opt_value
                        evaluator.close()
                        # Reinitialize Stockfish evaluator with new path
                        evaluator = StockfishEvaluator(
                            stockfish_path=stockfish_path, 
                            depth=stockfish_depth, 
                            time_limit=stockfish_time_limit
                        )
                        mcts.evaluator = evaluator
                        print(f"info string Re-initialized Stockfish at {stockfish_path}", file=sys.stderr)
                sys.stderr.flush()

            elif cmd == "ucinewgame":
                board = chess.Board()
                mcts.transposition_table.clear()
                
            elif cmd == "position":
                try:
                    # Parsing: position [startpos | fen ...] [moves ...]
                    if len(tokens) < 2:
                        continue
                        
                    moves_idx = -1
                    if tokens[1] == "startpos":
                        temp_board = chess.Board()
                        if "moves" in tokens:
                            moves_idx = tokens.index("moves")
                    elif tokens[1] == "fen":
                        # Reconstruct Fen string
                        fen_tokens = []
                        idx = 2
                        while idx < len(tokens) and tokens[idx] != "moves":
                            fen_tokens.append(tokens[idx])
                            idx += 1
                        fen_str = " ".join(fen_tokens)
                        temp_board = chess.Board(fen_str)
                        if idx < len(tokens) and tokens[idx] == "moves":
                            moves_idx = idx

                    if moves_idx != -1:
                        # Apply moves to a temporary board to prevent corrupting the main board on error
                        for move_str in tokens[moves_idx+1:]:
                            move = chess.Move.from_uci(move_str)
                            temp_board.push(move)
                    
                    # Successfully parsed all moves; update the main board
                    board = temp_board
                except Exception as e:
                    import traceback
                    print(f"Error in position command: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                        
            elif cmd == "go":
                mcts.abort_search = False
                try:
                    # Check if game is already over
                    if board.is_game_over(claim_draw=True) or not board.legal_moves:
                        print("bestmove 0000")
                        sys.stdout.flush()
                        continue

                    # Runs search and outputs bestmove
                    policy = mcts.get_move_probs(board, temp=0.0)
                    action = int(np.argmax(policy))
                    best_move = decode_move(action, board)
                    
                    # Double check move legality
                    if best_move not in board.legal_moves:
                        print(f"info string Warning: MCTS selected illegal move {best_move.uci()}, falling back", file=sys.stderr)
                        if list(board.legal_moves):
                            best_move = list(board.legal_moves)[0]
                        else:
                            best_move = chess.Move.null()
                    
                    print(f"bestmove {best_move.uci()}")
                    sys.stdout.flush()
                except Exception as e:
                    import traceback
                    print(f"Error in go command: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    
                    # Fallback to prevent GUI freeze
                    if list(board.legal_moves):
                        fallback_move = list(board.legal_moves)[0]
                        print(f"bestmove {fallback_move.uci()}")
                    else:
                        print("bestmove 0000")
                    sys.stdout.flush()
                
            elif cmd == "quit":
                break
                
            elif cmd == "stop":
                pass

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.stderr.flush()

    # Clean up engine processes
    evaluator.close()

if __name__ == "__main__":
    main()
