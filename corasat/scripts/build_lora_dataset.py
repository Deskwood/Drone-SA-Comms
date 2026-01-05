"""Build LoRA training datasets from a simulation log."""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SITUATION_RE = re.compile(r"Drone (\d+) Situation:")
RESPONSE_RE = re.compile(r"Drone (\d+) response:")
SUMMARY_RE = re.compile(r"Decision Support Summary:\s*best choice\s+(\w+)(?:\s+(\w+))?", re.IGNORECASE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_response(response_text: str) -> Optional[Dict[str, object]]:
    if not response_text or response_text.strip() == "<no parseable result>":
        return None
    try:
        parsed = ast.literal_eval(response_text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _summary_to_response(situation_text: str) -> Optional[Dict[str, object]]:
    for line in situation_text.splitlines():
        match = SUMMARY_RE.search(line.strip())
        if not match:
            continue
        action = (match.group(1) or "").lower()
        direction = (match.group(2) or "").lower() if action == "move" else None
        if action not in {"move", "broadcast", "wait"}:
            return None
        if action == "move" and not direction:
            return None
        return {
            "rationale": "Following Decision Support Summary.",
            "action": action,
            "direction": direction if action == "move" else None,
            "message": None,
            "memory": "",
        }
    return None


def _extract_samples(log_text: str) -> List[Tuple[int, str, str]]:
    lines = log_text.splitlines()
    samples: List[Tuple[int, str, str]] = []
    pending: Dict[int, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        situation_match = SITUATION_RE.search(line)
        if situation_match:
            drone_id = int(situation_match.group(1))
            start = i + 1
            j = start
            while j < len(lines):
                if lines[j].startswith("INFO:root:"):
                    if "Drone " in lines[j] and ("Situation:" in lines[j] or "response:" in lines[j]):
                        break
                    if "Simulation - Round" in lines[j]:
                        break
                j += 1
            pending[drone_id] = "\n".join(lines[start:j]).rstrip()
            i = j
            continue
        response_match = RESPONSE_RE.search(line)
        if response_match:
            drone_id = int(response_match.group(1))
            start = i + 1
            j = start
            while j < len(lines) and not lines[j].startswith("INFO:root:"):
                j += 1
            response_text = "\n".join(lines[start:j]).rstrip()
            situation_text = pending.get(drone_id)
            if situation_text and response_text:
                samples.append((drone_id, situation_text, response_text))
            i = j
            continue
        i += 1
    return samples


def _write_dataset(
    samples: List[Dict[str, object]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def _write_dataset_info(out_dir: Path, train_file: str, val_file: str) -> None:
    info = {
        "l4_train": {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        },
        "l4_val": {
            "file_name": val_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        },
    }
    path = out_dir / "dataset_info.json"
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LoRA datasets from a simulation log.")
    parser.add_argument("--log", required=True, help="Simulation log path.")
    parser.add_argument("--rules", required=True, help="Rules file for system prompt.")
    parser.add_argument("--out-dir", default="corasat/lora", help="Output directory for datasets.")
    parser.add_argument("--mode", choices=("imitation", "summary"), default="imitation")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction to reserve for validation.")
    parser.add_argument("--seed", type=int, default=1, help="Shuffle seed.")
    args = parser.parse_args()

    log_path = Path(args.log)
    rules_path = Path(args.rules)
    out_dir = Path(args.out_dir)

    log_text = _read_text(log_path)
    rules_text = _read_text(rules_path).strip()

    raw_samples = _extract_samples(log_text)
    samples: List[Dict[str, object]] = []
    skipped = 0
    for drone_id, situation_text, response_text in raw_samples:
        if args.mode == "summary":
            response = _summary_to_response(situation_text)
        else:
            response = _parse_response(response_text)
        if response is None:
            skipped += 1
            continue
        payload = {
            "messages": [
                {"from": "system", "value": rules_text},
                {"from": "human", "value": situation_text},
                {"from": "gpt", "value": json.dumps(response, ensure_ascii=True)},
            ]
        }
        samples.append(payload)

    random.seed(args.seed)
    random.shuffle(samples)
    val_count = int(len(samples) * args.val_split)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    train_path = out_dir / "l4_train.jsonl"
    val_path = out_dir / "l4_val.jsonl"
    _write_dataset(train_samples, train_path)
    _write_dataset(val_samples, val_path)
    _write_dataset_info(out_dir, train_path.name, val_path.name)

    print(f"Samples: {len(samples)} (train {len(train_samples)}, val {len(val_samples)}), skipped {skipped}.")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Dataset info: {out_dir / 'dataset_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
