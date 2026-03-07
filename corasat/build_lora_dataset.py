"""Build LoRA training datasets from a simulation log."""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

STRUCTURE_PROMPT_PATH = Path(__file__).resolve().parent / "profiles" / "structure" / "output_contract.txt"


SITUATION_RE = re.compile(r"Drone (\d+) Situation:")
RESPONSE_RE = re.compile(r"Drone (\d+) response:")
SUMMARY_RE = re.compile(r"Decision Support Summary:\s*best choice\s+(\w+)(?:\s+(\w+))?", re.IGNORECASE)
RUN_SEED_RE = re.compile(r"==== Running seed\s+(\d+)")


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


def _extract_samples(log_text: str) -> List[Tuple[Optional[int], int, str, str]]:
    lines = log_text.splitlines()
    samples: List[Tuple[Optional[int], int, str, str]] = []
    pending: Dict[int, Tuple[Optional[int], str]] = {}
    current_seed: Optional[int] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        seed_match = RUN_SEED_RE.search(line)
        if seed_match:
            try:
                current_seed = int(seed_match.group(1))
            except Exception:
                current_seed = None
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
            pending[drone_id] = (current_seed, "\n".join(lines[start:j]).rstrip())
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
            pending_item = pending.get(drone_id)
            if pending_item and response_text:
                sample_seed, situation_text = pending_item
                samples.append((sample_seed, drone_id, situation_text, response_text))
            i = j
            continue
        i += 1
    return samples


def _parse_seed_spec(seed_text: str) -> Optional[Set[int]]:
    seed_text = (seed_text or "").strip()
    if not seed_text:
        return None
    seeds: Set[int] = set()
    for token in [part.strip() for part in seed_text.split(",") if part.strip()]:
        if "-" in token:
            start_txt, end_txt = token.split("-", 1)
            start = int(start_txt.strip())
            end = int(end_txt.strip())
            step = 1 if end >= start else -1
            for seed in range(start, end + step, step):
                seeds.add(seed)
            continue
        seeds.add(int(token))
    return seeds


def _write_dataset(
    samples: List[Dict[str, object]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def _write_dataset_info(out_dir: Path, prefix: str, train_file: str, val_file: str) -> None:
    train_key = f"{prefix}_train"
    val_key = f"{prefix}_val"
    info = {
        train_key: {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        },
        val_key: {
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
    parser.add_argument(
        "--dataset-prefix",
        default="l4",
        help="Dataset file/key prefix, e.g. l4 -> l4_train.jsonl and l4_val.jsonl.",
    )
    parser.add_argument(
        "--include-seeds",
        default="",
        help="Optional seed filter for samples, e.g. 1-80,95,100-120.",
    )
    parser.add_argument(
        "--train-seeds",
        default="",
        help="Optional explicit seed split for train set (overrides val-split).",
    )
    parser.add_argument(
        "--val-seeds",
        default="",
        help="Optional explicit seed split for validation set (overrides val-split).",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    rules_path = Path(args.rules)
    out_dir = Path(args.out_dir)

    log_text = _read_text(log_path)
    rules_text = _read_text(rules_path).strip()
    structure_text = _read_text(STRUCTURE_PROMPT_PATH).strip() if STRUCTURE_PROMPT_PATH.exists() else ""
    system_prompt = "\n\n".join(part for part in (structure_text, rules_text) if part)
    include_seeds = _parse_seed_spec(args.include_seeds)
    train_seed_set = _parse_seed_spec(args.train_seeds)
    val_seed_set = _parse_seed_spec(args.val_seeds)
    explicit_seed_split = train_seed_set is not None or val_seed_set is not None
    if explicit_seed_split and train_seed_set is None:
        train_seed_set = set()
    if explicit_seed_split and val_seed_set is None:
        val_seed_set = set()

    raw_samples = _extract_samples(log_text)
    staged_samples: List[Tuple[Optional[int], Dict[str, object]]] = []
    skipped = 0
    skipped_seed_filter = 0
    for sample_seed, drone_id, situation_text, response_text in raw_samples:
        if include_seeds is not None:
            if sample_seed is None or sample_seed not in include_seeds:
                skipped_seed_filter += 1
                continue
        if args.mode == "summary":
            response = _summary_to_response(situation_text)
        else:
            response = _parse_response(response_text)
        if response is None:
            skipped += 1
            continue
        payload = {
            "messages": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": situation_text},
                {"from": "gpt", "value": json.dumps(response, ensure_ascii=True)},
            ]
        }
        staged_samples.append((sample_seed, payload))

    if explicit_seed_split:
        train_samples: List[Dict[str, object]] = []
        val_samples: List[Dict[str, object]] = []
        split_skipped = 0
        for sample_seed, payload in staged_samples:
            if sample_seed is None:
                split_skipped += 1
                continue
            if sample_seed in train_seed_set:
                train_samples.append(payload)
                continue
            if sample_seed in val_seed_set:
                val_samples.append(payload)
                continue
            split_skipped += 1
        random.seed(args.seed)
        random.shuffle(train_samples)
        random.shuffle(val_samples)

        # Keep smoke/truncated runs robust: explicit seed filters can yield an empty split.
        # For training pipelines that require both datasets, duplicate one sample if needed.
        total_split_samples = len(train_samples) + len(val_samples)
        if total_split_samples == 0:
            raise RuntimeError(
                "No samples available after explicit seed split. "
                "Check --train-seeds/--val-seeds and source log coverage."
            )
        if not train_samples:
            train_samples.append(dict(val_samples[0]))
        if not val_samples:
            val_samples.append(dict(train_samples[0]))
    else:
        samples = [payload for _, payload in staged_samples]
        random.seed(args.seed)
        random.shuffle(samples)
        val_count = int(len(samples) * args.val_split)
        val_samples = samples[:val_count]
        train_samples = samples[val_count:]
        split_skipped = 0

    dataset_prefix = (args.dataset_prefix or "l4").strip()
    if not dataset_prefix:
        dataset_prefix = "l4"

    train_path = out_dir / f"{dataset_prefix}_train.jsonl"
    val_path = out_dir / f"{dataset_prefix}_val.jsonl"
    _write_dataset(train_samples, train_path)
    _write_dataset(val_samples, val_path)
    _write_dataset_info(out_dir, dataset_prefix, train_path.name, val_path.name)

    total_kept = len(train_samples) + len(val_samples)
    print(
        "Samples: "
        f"{total_kept} (train {len(train_samples)}, val {len(val_samples)}), "
        f"skipped_parse {skipped}, skipped_seed_filter {skipped_seed_filter}, skipped_split {split_skipped}."
    )
    print(f"Dataset prefix: {dataset_prefix}")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Dataset info: {out_dir / 'dataset_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
