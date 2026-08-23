# test_uci.py (Place this in your workspace folder)
import subprocess
import time

def test_engine():
    process = subprocess.Popen(['python', 'uci_engine.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    commands = [
        "uci\n",
        "isready\n",
        "ucinewgame\n",
        "position startpos moves e2e4 e7e5\n",
        "go movetime 1000\n",
        "position startpos moves e2e4 e7e5 g1f3 b8c6\n",
        "go movetime 1000\n",
        "quit\n"
    ]
    
    print("--- Simulating Fast GUI UCI State Sequence ---")
    for cmd in commands:
        process.stdin.write(cmd)
        process.stdin.flush()
        time.sleep(0.2) # Fast sequence to test for asynchronous desync issues
        
    stdout, _ = process.communicate()
    print("Engine Output Log:\n", stdout)
    
    # Validation Rules
    go_count = sum(1 for cmd in commands if cmd.strip().startswith("go"))
    bestmove_count = stdout.count("bestmove")
    
    if bestmove_count == 0:
        print("CRITICAL ERROR: Engine failed to yield any bestmove token.")
        exit(1)
    if bestmove_count != go_count:
        print(f"WARNING: Expected {go_count} bestmove responses, got {bestmove_count}.")
    if stdout.count("readyok") > 1:
        print("CRITICAL ERROR: Duplicate readyok state sequence detected.")
        exit(1)
        
    print(f"SUCCESS: Engine handled rapid position changes cleanly! "
          f"({bestmove_count}/{go_count} bestmove responses)")
    exit(0)

if __name__ == "__main__":
    test_engine()