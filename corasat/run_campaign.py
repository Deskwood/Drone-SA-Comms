"""Deprecated compatibility wrapper.

Campaign execution is now handled by main.py using config.json campaign entries.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


CORASAT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Deprecated wrapper for campaign runs.")
    parser.add_argument(
        "--config",
        default=str(CORASAT_ROOT / "config.json"),
        help="Campaign config path (default: corasat/config.json).",
    )
    args = parser.parse_args()

    print("run_campaign.py is deprecated. Delegating to main.py campaign runner.")
    cmd = [sys.executable, str(CORASAT_ROOT / "main.py"), "--config", str(args.config)]
    proc = subprocess.run(cmd, cwd=str(CORASAT_ROOT), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
