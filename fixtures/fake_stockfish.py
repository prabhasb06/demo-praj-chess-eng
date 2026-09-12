"""Deterministic process fixture for cancellation and failure tests."""
import sys
import threading
import time

stop = threading.Event()
worker = None


def search(depth):
    if depth == 78:
        import os
        os._exit(1)
    if depth == 77:
        time.sleep(10)
        return
    stop.wait(0.1)
    print('info depth 2 multipv 1 score cp 30 pv e2e4', flush=True)
    # Extra output between cancellation and bestmove catches incomplete drains.
    print('info string draining', flush=True)
    time.sleep(0.005)
    print('bestmove e2e4', flush=True)


for line in sys.stdin:
    tokens = line.split()
    if not tokens:
        continue
    if tokens[0] == 'uci':
        print('uciok', flush=True)
    elif tokens[0] == 'isready':
        print('readyok', flush=True)
    elif tokens[0] == 'go':
        stop.clear()
        worker = threading.Thread(target=search, args=(int(tokens[2]),), daemon=True)
        worker.start()
    elif tokens[0] == 'stop':
        stop.set()
    elif tokens[0] == 'quit':
        break
