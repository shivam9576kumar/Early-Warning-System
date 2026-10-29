"""Root-level entrypoint to execute the EWS Credit Risk pipeline."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "ews-credit-risk"

if __name__ == "__main__":
    script_path = PROJECT_DIR / "src" / "pipeline.py"
    print(f"[*] Running Early-Warning Credit Risk Pipeline from {script_path} ...\n")
    ret = subprocess.call([sys.executable, str(script_path)], cwd=str(PROJECT_DIR))
    sys.exit(ret)
