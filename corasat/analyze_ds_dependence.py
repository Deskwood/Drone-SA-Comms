"""Aggregate LM-vs-DS dependence metrics from campaign trace logs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


SUMMARY_FIELDS = [
    "lab_id",
    "label",
    "seed_count",
    "turns_compared",
    "match_action_count",
    "match_action_rate",
    "match_top_choice_count",
    "match_top_choice_rate",
    "rationale_mentions_ds_count",
    "rationale_mentions_ds_rate",
    "rationale_mentions_best_choice_count",
    "rationale_mentions_best_choice_rate",
    "mean_norm_score",
]


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_lab_results(campaign_dir: Path) -> Dict[str, Dict[str, str]]:
    path = campaign_dir / "lab_results.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row.get("lab_id") or "").strip(): row
        for row in rows
        if str(row.get("lab_id") or "").strip()
    }


def _iter_seed_reports(campaign_dir: Path) -> Iterable[Path]:
    yield from sorted(campaign_dir.rglob("seed_*.json"))


def _resolve_trace_path(campaign_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path
    return (campaign_dir / path).resolve()


def _iter_alignment_events(trace_path: Path) -> Iterable[Dict[str, object]]:
    if not trace_path.exists():
        return
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("event") or "").strip().lower() != "lm_ds_alignment":
                continue
            yield payload


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 5)


def summarize_campaign(campaign_dir: Path) -> List[Dict[str, object]]:
    lab_results = _load_lab_results(campaign_dir)
    grouped: Dict[str, Dict[str, object]] = {}

    for seed_report_path in _iter_seed_reports(campaign_dir):
        seed_report = _read_json(seed_report_path)
        lab_id = str(seed_report.get("lab_id") or "").strip()
        if not lab_id:
            continue
        trace_path = _resolve_trace_path(campaign_dir, str(seed_report.get("lm_trace_log") or ""))
        if not trace_path.exists():
            continue

        agg = grouped.setdefault(
            lab_id,
            {
                "lab_id": lab_id,
                "label": lab_results.get(lab_id, {}).get("label", lab_id),
                "seed_ids": set(),
                "turns_compared": 0,
                "match_action_count": 0,
                "match_top_choice_count": 0,
                "rationale_mentions_ds_count": 0,
                "rationale_mentions_best_choice_count": 0,
            },
        )
        agg["seed_ids"].add(str(seed_report.get("seed") or ""))

        for event in _iter_alignment_events(trace_path):
            agg["turns_compared"] += 1
            if bool(event.get("match_action", False)):
                agg["match_action_count"] += 1
            if bool(event.get("match_top_choice", False)):
                agg["match_top_choice_count"] += 1
            if bool(event.get("rationale_mentions_decision_support", False)):
                agg["rationale_mentions_ds_count"] += 1
            if bool(event.get("rationale_mentions_best_choice", False)):
                agg["rationale_mentions_best_choice_count"] += 1

    rows: List[Dict[str, object]] = []
    for lab_id in sorted(grouped.keys()):
        agg = grouped[lab_id]
        turns_compared = int(agg.get("turns_compared", 0) or 0)
        seed_count = len(agg.get("seed_ids", set()) or set())
        match_action_count = int(agg.get("match_action_count", 0) or 0)
        match_top_choice_count = int(agg.get("match_top_choice_count", 0) or 0)
        rationale_ds_count = int(agg.get("rationale_mentions_ds_count", 0) or 0)
        rationale_best_count = int(agg.get("rationale_mentions_best_choice_count", 0) or 0)
        row = {
            "lab_id": lab_id,
            "label": agg.get("label", lab_id),
            "seed_count": seed_count,
            "turns_compared": turns_compared,
            "match_action_count": match_action_count,
            "match_action_rate": _safe_rate(match_action_count, turns_compared),
            "match_top_choice_count": match_top_choice_count,
            "match_top_choice_rate": _safe_rate(match_top_choice_count, turns_compared),
            "rationale_mentions_ds_count": rationale_ds_count,
            "rationale_mentions_ds_rate": _safe_rate(rationale_ds_count, turns_compared),
            "rationale_mentions_best_choice_count": rationale_best_count,
            "rationale_mentions_best_choice_rate": _safe_rate(rationale_best_count, turns_compared),
            "mean_norm_score": lab_results.get(lab_id, {}).get("mean_norm_score", ""),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize LM-vs-DS dependence from Corasat campaign traces.")
    parser.add_argument("--campaign-dir", required=True, help="Campaign run directory containing lab_results.csv and seed reports.")
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional output CSV path. Defaults to <campaign-dir>/ds_dependence_summary.csv.",
    )
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else campaign_dir / "ds_dependence_summary.csv"

    rows = summarize_campaign(campaign_dir)
    write_csv(out_csv, rows)
    print(f"Wrote {len(rows)} lab row(s) to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
