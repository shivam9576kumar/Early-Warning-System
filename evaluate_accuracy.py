"""Root-level entrypoint for evaluate_accuracy.py in ews-credit-risk.

Allows running `python evaluate_accuracy.py` directly from the root repository folder.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "ews-credit-risk"

if __name__ == "__main__":
    script_path = PROJECT_DIR / "evaluate_accuracy.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    ret = subprocess.call(cmd, cwd=str(PROJECT_DIR))
    sys.exit(ret)

