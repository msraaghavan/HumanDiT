"""Pull a Colab session's packed results now (the watchdog also does this before it stops a VM).

python3 scripts/colab/pull.py SESSION [SESSION ...]   ->  runs/colab/pulled/<session>/ (sha256-verified)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watchdog

if __name__ == "__main__":
    bad = [s for s in sys.argv[1:] if not watchdog.pull(s)]
    print("all pulled and verified" if not bad else f"PULL FAILED: {bad}")
    sys.exit(1 if bad else 0)
