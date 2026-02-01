"""Run the L0 rules-only lab configuration (no decision-support scores)."""
from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import classes.Core as core
import main


def main_entry() -> None:
    config_path = BASE_DIR / "lab_configs" / "L0_RulesOnly_NoDecisionSupport" / "config.json"
    core.CONFIG_PATH = str(config_path.resolve())
    core.reload_config(core.CONFIG_PATH)
    main.run_all_seeds(core.CONFIG_PATH)


if __name__ == "__main__":
    main_entry()
