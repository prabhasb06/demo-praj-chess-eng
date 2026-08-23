"""
Entry point for the custom hybrid chess engine.
Delegates to uci_engine.py to run the main UCI loop.
"""
from uci_engine import main

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
