"""Root-level entrypoint for predict.py in ews-credit-risk.

Allows running `python predict.py <file.csv>` directly from the root repository folder.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "ews-credit-risk"

if __name__ == "__main__":
    script_path = PROJECT_DIR / "predict.py"
    # Resolve relative paths in arguments so they remain valid when cwd changes to PROJECT_DIR
    resolved_args = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists():
            resolved_args.append(str(p.resolve()))
        else:
            resolved_args.append(arg)

    cmd = [sys.executable, str(script_path)] + resolved_args
    ret = subprocess.call(cmd, cwd=str(PROJECT_DIR))
    sys.exit(ret)

