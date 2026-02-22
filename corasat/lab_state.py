"""Deprecated compatibility wrapper.

Lab activation is no longer required. Use config.json campaign labs in main.py.
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Deprecated lab_state wrapper.")
    parser.add_argument("command", nargs="?", default="status")
    _ = parser.parse_args()

    print("lab_state.py is deprecated.")
    print("Use main.py with config.json campaign.labs instead of activate/restore steps.")
    print("Example: python main.py --config config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
