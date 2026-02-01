#!/usr/bin/env python3
"""Plot per-seed lab distributions for a selected metric."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _resolve_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _norm_path(value: str) -> str:
    if not value:
        return ""
    return os.path.normpath(value).lower()


def _lab_logfiles(lab_results_path: Path) -> Dict[str, str]:
    lab_logs: Dict[str, str] = {}
    with lab_results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label.startswith("L"):
                continue
            lab_id = label.split(" ")[0]
            logfile = row.get("logfile") or ""
            if logfile:
                lab_logs[lab_id] = logfile
    return lab_logs


def _metric_by_lab(
    results_path: Path,
    lab_logs: Dict[str, str],
    metric: str,
) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {lab: [] for lab in lab_logs}
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            logfile = _norm_path(row.get("logfile") or "")
            value = row.get(metric)
            if value in (None, ""):
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            for lab, log in lab_logs.items():
                if logfile == _norm_path(log):
                    values[lab].append(parsed)
                    break
    return values


def _ordered_labs(lab_logs: Dict[str, str]) -> List[str]:
    def _lab_key(item: str) -> Tuple[int, str]:
        try:
            return (int(item[1:]), item)
        except Exception:
            return (999, item)

    return sorted(lab_logs.keys(), key=_lab_key)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-seed metric distributions for lab runs."
    )
    parser.add_argument(
        "--lab-results",
        default=str(_resolve_root() / "lab_results.csv"),
        help="Path to lab_results.csv",
    )
    parser.add_argument(
        "--results",
        default=str(_resolve_root() / "results.csv"),
        help="Path to results.csv",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric column to plot (from results.csv).",
    )
    parser.add_argument(
        "--ylabel",
        required=True,
        help="Y-axis label.",
    )
    parser.add_argument(
        "--yscale",
        default="linear",
        choices=["linear", "log"],
        help="Y-axis scale (linear or log).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    lab_results_path = Path(args.lab_results)
    results_path = Path(args.results)
    output_path = Path(args.output)

    lab_logs = _lab_logfiles(lab_results_path)
    if not lab_logs:
        raise SystemExit("No lab logfiles found in lab_results.csv.")

    values = _metric_by_lab(results_path, lab_logs, args.metric)
    labs = _ordered_labs(lab_logs)
    data = [values.get(lab, []) for lab in labs]

    if not any(data):
        raise SystemExit(f"No values found for metric '{args.metric}'.")

    plt.figure(figsize=(7.5, 3.6))
    try:
        plt.boxplot(data, tick_labels=labs, showfliers=True)
    except TypeError:
        plt.boxplot(data, labels=labs, showfliers=True)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.ylabel(args.ylabel)
    plt.xlabel("Lab")
    if args.yscale != "linear":
        plt.yscale(args.yscale)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
