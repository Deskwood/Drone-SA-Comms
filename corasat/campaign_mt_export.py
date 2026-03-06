"""Export campaign results into MT-ready LaTeX tables and figures."""
from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy import stats
except Exception:
    stats = None


PROFILE_SPECS: Dict[str, Dict[str, str]] = {
    "rules": {"subdir": "rules", "suffix": "_rules.txt", "format": "text"},
    "prompt": {"subdir": "prompt_requests", "suffix": "_prompt_requests.json", "format": "json"},
    "model": {"subdir": "model", "suffix": "_model.json", "format": "json"},
    "fine_tuning": {"subdir": "fine_tuning", "suffix": "_fine_tuning.json", "format": "json"},
    "action_policy": {"subdir": "action_policy", "suffix": "_action_policy.json", "format": "json"},
    "communication": {"subdir": "communication", "suffix": "_communication.json", "format": "json"},
    "decision_support": {
        "subdir": "decision_support",
        "suffix": "_decision_support.json",
        "format": "json",
    },
}

LAB_PROFILE_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("rules_id", "R", "rules"),
    ("prompt_id", "P", "prompt"),
    ("drone_support_id", "DS", "decision_support"),
    ("model_id", "M", "model"),
    ("fine_tuning_id", "FT", "fine_tuning"),
    ("action_policy_id", "A", "action_policy"),
    ("communication_id", "C", "communication"),
)

RULES_NAME_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "R0": ("No rules", "No LLM rules prompt is provided."),
    "R1": ("Reduced rules", "Compact rule set with core constraints."),
    "R2": ("Full rules", "Full rule set with complete mission directives."),
    "R3": ("Waypoint-clarified rules", "Full rules plus waypoint-vs-sector clarification."),
}


def _resolve_path(base_dir: Path, raw_value: str, fallback: str) -> Path:
    token = str(raw_value or "").strip()
    if not token:
        token = fallback
    path = Path(token)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _id_sort_key(token: str) -> Tuple[str, int, str]:
    text = str(token or "").strip().upper()
    match = re.match(r"^([A-Z]+)(\d+)$", text)
    if match:
        return (match.group(1), int(match.group(2)), text)
    return (text, -1, text)


def _latex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _latex_softbreak(value: Any) -> str:
    text = _latex_escape(value)
    text = text.replace(r"\_", r"\_\allowbreak{}")
    text = text.replace("-", r"-\allowbreak{}")
    return text


def _fmt_mu_sigma(values: Sequence[float], digits: int = 4) -> str:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not cleaned:
        return "n/a"
    mu = statistics.mean(cleaned)
    sigma = statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0
    return f"{mu:.{digits}f} $\\pm$ {sigma:.{digits}f}"


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _profile_path(corasat_root: Path, kind: str, profile_id: str) -> Path:
    spec = PROFILE_SPECS[kind]
    return corasat_root / "profiles" / spec["subdir"] / f"{profile_id}{spec['suffix']}"


def _profile_metadata(corasat_root: Path, kind: str, profile_id: str) -> Dict[str, str]:
    profile_id = str(profile_id or "").strip().upper()
    if not profile_id:
        return {"id": "", "name": "", "description": ""}
    if kind == "rules":
        if profile_id in RULES_NAME_OVERRIDES:
            name, description = RULES_NAME_OVERRIDES[profile_id]
            return {"id": profile_id, "name": name, "description": description}
        return {"id": profile_id, "name": "Rules profile", "description": profile_id}

    path = _profile_path(corasat_root, kind, profile_id)
    if not path.exists():
        return {"id": profile_id, "name": "missing profile", "description": str(path)}
    payload = _read_json(path)
    return {
        "id": profile_id,
        "name": str(payload.get("name") or profile_id),
        "description": str(payload.get("description") or ""),
    }


def _lab_lookup(labs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        lab_id = str(lab.get("id") or lab.get("lab_id") or "").strip()
        if lab_id:
            by_id[lab_id] = lab
    return by_id


def _infer_changed_item(lab: Dict[str, Any], parent: Optional[Dict[str, Any]]) -> str:
    explicit = str(lab.get("changed_item") or "").strip()
    if explicit:
        return explicit
    if not isinstance(parent, dict):
        return "baseline"
    changed: List[str] = []
    short_name = {
        "rules_id": "R",
        "prompt_id": "P",
        "drone_support_id": "DS",
        "decision_support_id": "DS",
        "model_id": "M",
        "fine_tuning_id": "FT",
        "action_policy_id": "A",
        "communication_id": "C",
    }
    for field, _, _ in LAB_PROFILE_FIELDS:
        if str(lab.get(field) or "").upper() != str(parent.get(field) or "").upper():
            changed.append(short_name.get(field, field.replace("_id", "")))
    if not changed:
        return "none"
    return ", ".join(changed)


def _resolve_lab_runtime_model(corasat_root: Path, lab: Dict[str, Any]) -> str:
    model_id = str(lab.get("model_id") or "").strip().upper()
    ft_id = str(lab.get("fine_tuning_id") or "").strip().upper()
    model_name = ""
    if model_id:
        model_path = _profile_path(corasat_root, "model", model_id)
        if model_path.exists():
            payload = _read_json(model_path)
            model_name = str(payload.get("selected_model") or payload.get("runtime_model") or "")
    if ft_id:
        ft_path = _profile_path(corasat_root, "fine_tuning", ft_id)
        if ft_path.exists():
            payload = _read_json(ft_path)
            override_model = str(payload.get("runtime_model_override") or "").strip()
            if override_model:
                model_name = override_model
    return model_name


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _float_from_row(row: Dict[str, str], field: str) -> Optional[float]:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _int_from_row(row: Dict[str, str], field: str) -> Optional[int]:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _build_lab_matrix_tex(corasat_root: Path, labs: Sequence[Dict[str, Any]]) -> str:
    used_ids: Dict[str, set] = {kind: set() for _, _, kind in LAB_PROFILE_FIELDS}
    for lab in labs:
        for field, _, kind in LAB_PROFILE_FIELDS:
            token = str(lab.get(field) or "").strip().upper()
            if token:
                used_ids[kind].add(token)

    lines: List[str] = []
    lines.append("% Auto-generated by campaign_mt_export.py")
    lines.append("\\begingroup")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\setlength{\\LTpre}{6pt}")
    lines.append("\\setlength{\\LTpost}{6pt}")
    lines.append("\\begin{longtable}{p{0.08\\linewidth} p{0.16\\linewidth} p{0.22\\linewidth} p{0.45\\linewidth}}")
    lines.append("\\caption{Configuration-item identifiers used in this campaign.}\\label{tab:config-item-ids}\\\\")
    lines.append("\\hline")
    lines.append("ID & Category & Name & Description \\\\")
    lines.append("\\hline")
    lines.append("\\endfirsthead")
    lines.append("\\hline")
    lines.append("ID & Category & Name & Description \\\\")
    lines.append("\\hline")
    lines.append("\\endhead")

    category_names = {
        "rules": "Rules",
        "prompt": "Prompt",
        "decision_support": "Drone support",
        "model": "Model",
        "fine_tuning": "Fine tuning",
        "action_policy": "Action policy",
        "communication": "Communication",
    }
    for _, _, kind in LAB_PROFILE_FIELDS:
        ids = sorted(used_ids[kind], key=_id_sort_key)
        for profile_id in ids:
            meta = _profile_metadata(corasat_root, kind, profile_id)
            lines.append(
                f"{_latex_escape(profile_id)} & {_latex_escape(category_names[kind])} & "
                f"{_latex_escape(meta.get('name', ''))} & {_latex_escape(meta.get('description', ''))} \\\\"
            )

    lines.append("\\hline")
    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    lines.append("")

    lines.append("\\begingroup")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append("\\setlength{\\LTpre}{6pt}")
    lines.append("\\setlength{\\LTpost}{6pt}")
    lines.append(
        "\\begin{longtable}{l l l l l l l l p{0.17\\linewidth} p{0.15\\linewidth} p{0.19\\linewidth}}"
    )
    lines.append("\\caption{Lab-to-configuration mapping for campaign execution.}\\label{tab:lab-configs}\\\\")
    lines.append("\\hline")
    lines.append("Lab & R & P & DS & M & FT & A & C & Runtime model & Changed item & Idea \\\\")
    lines.append("\\hline")
    lines.append("\\endfirsthead")
    lines.append("\\hline")
    lines.append("Lab & R & P & DS & M & FT & A & C & Runtime model & Changed item & Idea \\\\")
    lines.append("\\hline")
    lines.append("\\endhead")

    lab_by_id = _lab_lookup(labs)
    for lab in labs:
        lab_id = str(lab.get("id") or lab.get("lab_id") or "").strip()
        parent = lab_by_id.get(str(lab.get("parent_lab") or "").strip())
        row = [
            _latex_escape(lab_id),
            _latex_escape(str(lab.get("rules_id") or "")),
            _latex_escape(str(lab.get("prompt_id") or "")),
            _latex_escape(str(lab.get("drone_support_id") or lab.get("decision_support_id") or "")),
            _latex_escape(str(lab.get("model_id") or "")),
            _latex_escape(str(lab.get("fine_tuning_id") or "")),
            _latex_escape(str(lab.get("action_policy_id") or "")),
            _latex_escape(str(lab.get("communication_id") or "")),
            _latex_softbreak(_resolve_lab_runtime_model(corasat_root, lab)),
            _latex_escape(_infer_changed_item(lab, parent)),
            _latex_escape(str(lab.get("title") or lab.get("notes") or lab.get("label") or "")),
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    lines.append("")
    return "\n".join(lines)


def _build_lab_results_tex(labs: Sequence[Dict[str, Any]], lab_rows: Sequence[Dict[str, str]]) -> str:
    rows_by_id = {str(row.get("lab_id") or "").strip(): row for row in lab_rows}
    lines: List[str] = []
    lines.append("% Auto-generated by campaign_mt_export.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begingroup")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l p{0.37\\linewidth} c c c c}")
    lines.append("\\hline")
    lines.append("Lab & Configuration & Mean $\\pm$ Std (norm score) & Seeds & Runtime (s/seed) & LM tokens \\\\")
    lines.append("\\hline")

    for lab in labs:
        lab_id = str(lab.get("id") or lab.get("lab_id") or "").strip()
        row = rows_by_id.get(lab_id, {})
        tuple_text = (
            f"[{lab.get('rules_id','')},{lab.get('prompt_id','')},{lab.get('drone_support_id','')},"
            f"{lab.get('model_id','')},{lab.get('fine_tuning_id','')},{lab.get('action_policy_id','')},"
            f"{lab.get('communication_id','')}]"
        )
        mean_value = _float_from_row(row, "mean_norm_score")
        std_value = _float_from_row(row, "std_norm_score")
        metric = (
            f"{mean_value:.5f} $\\pm$ {std_value:.5f}"
            if mean_value is not None and std_value is not None
            else "n/a"
        )
        seeds_done = _int_from_row(row, "seed_count_completed")
        seeds_planned = _int_from_row(row, "seed_count_planned")
        seeds_text = f"{seeds_done}/{seeds_planned}" if seeds_done is not None and seeds_planned is not None else "n/a"
        runtime_total = _float_from_row(row, "total_runtime_s")
        runtime_mean = f"{runtime_total / seeds_done:.2f}" if runtime_total is not None and seeds_done and seeds_done > 0 else "n/a"
        lm_tokens = _int_from_row(row, "total_lm_tokens")
        lines.append(
            " & ".join(
                [
                    _latex_escape(lab_id),
                    _latex_escape(tuple_text),
                    metric,
                    _latex_escape(seeds_text),
                    _latex_escape(runtime_mean),
                    _latex_escape(str(lm_tokens if lm_tokens is not None else "n/a")),
                ]
            )
            + " \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\caption{Per-lab aggregated normalized score and runtime statistics for the campaign.}")
    lines.append("\\label{tab:lab-results}")
    lines.append("\\endgroup")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def _build_aux_metrics_tex(labs: Sequence[Dict[str, Any]], results_rows: Sequence[Dict[str, str]]) -> str:
    metrics = ("coverage", "broadcast_rate", "rendezvous_success", "wait_rate", "correct_edge_rate", "avg_turn_duration_s")
    per_lab: Dict[str, Dict[str, List[float]]] = {}
    for row in results_rows:
        lab_id = str(row.get("lab_id") or "").strip()
        if not lab_id:
            continue
        bucket = per_lab.setdefault(lab_id, {field: [] for field in metrics})
        for field in metrics:
            value = _float_from_row(row, field)
            if value is not None:
                bucket[field].append(value)

    lines: List[str] = []
    lines.append("% Auto-generated by campaign_mt_export.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begingroup")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l c c c c c c}")
    lines.append("\\hline")
    lines.append(
        "Lab & Coverage ($\\mu \\pm \\sigma$) & Broadcast rate ($\\mu \\pm \\sigma$) & "
        "Rendezvous ($\\mu \\pm \\sigma$) & Wait rate ($\\mu \\pm \\sigma$) & "
        "Correct-edge rate ($\\mu \\pm \\sigma$) & Avg turn duration (s) \\\\"
    )
    lines.append("\\hline")
    for lab in labs:
        lab_id = str(lab.get("id") or lab.get("lab_id") or "").strip()
        bucket = per_lab.get(lab_id, {})
        lines.append(
            " & ".join(
                [
                    _latex_escape(lab_id),
                    _fmt_mu_sigma(bucket.get("coverage", [])),
                    _fmt_mu_sigma(bucket.get("broadcast_rate", [])),
                    _fmt_mu_sigma(bucket.get("rendezvous_success", [])),
                    _fmt_mu_sigma(bucket.get("wait_rate", [])),
                    _fmt_mu_sigma(bucket.get("correct_edge_rate", [])),
                    _fmt_mu_sigma(bucket.get("avg_turn_duration_s", [])),
                ]
            )
            + " \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\caption{Auxiliary mission metrics aggregated over campaign seed runs.}")
    lines.append("\\label{tab:aux-metrics}")
    lines.append("\\endgroup")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def _bootstrap_ci(values: Sequence[float], samples: int = 10000) -> Tuple[Optional[float], Optional[float]]:
    if not values or np is None:
        return None, None
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(12345)
    means = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(samples)], dtype=float)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _build_pairwise_tex(labs: Sequence[Dict[str, Any]], results_rows: Sequence[Dict[str, str]]) -> str:
    order = [str(lab.get("id") or lab.get("lab_id") or "").strip() for lab in labs]
    per_lab_seed: Dict[str, Dict[str, float]] = {}
    for row in results_rows:
        lab_id = str(row.get("lab_id") or "").strip()
        seed = str(row.get("seed") or "").strip()
        score = _float_from_row(row, "norm_score")
        if not lab_id or not seed or score is None:
            continue
        per_lab_seed.setdefault(lab_id, {})[seed] = score

    lines: List[str] = []
    lines.append("% Auto-generated by campaign_mt_export.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\hline")
    lines.append("Comparison & Mean diff. & 95\\% CI & Cohen's $d$ & Test ($p$) \\\\")
    lines.append("\\hline")

    for idx in range(1, len(order)):
        prev_lab = order[idx - 1]
        curr_lab = order[idx]
        prev = per_lab_seed.get(prev_lab, {})
        curr = per_lab_seed.get(curr_lab, {})
        shared = sorted(set(prev.keys()) & set(curr.keys()), key=lambda token: int(token))
        diffs = [curr[s] - prev[s] for s in shared]
        if diffs:
            mean_diff = statistics.mean(diffs)
            ci_low, ci_high = _bootstrap_ci(diffs)
            if len(diffs) > 1:
                sd = statistics.stdev(diffs)
                cohen = 0.0 if sd <= 0 else mean_diff / sd
            else:
                cohen = 0.0
            test_name = "bootstrap only"
            p_value = None
            if stats is not None and np is not None and len(diffs) >= 3:
                arr = np.array(diffs, dtype=float)
                try:
                    _, shapiro_p = stats.shapiro(arr)
                except Exception:
                    shapiro_p = 0.0
                if shapiro_p > 0.05:
                    try:
                        _, p_value = stats.ttest_1samp(arr, popmean=0.0)
                        test_name = "paired t-test"
                    except Exception:
                        pass
                else:
                    try:
                        _, p_value = stats.wilcoxon(arr)
                        test_name = "wilcoxon"
                    except Exception:
                        pass
            p_text = test_name if p_value is None else f"{test_name} ({p_value:.3g})"
            ci_text = "n/a" if ci_low is None or ci_high is None else f"[{ci_low:+.5f}, {ci_high:+.5f}]"
            lines.append(
                f"{prev_lab} $\\rightarrow$ {curr_lab} & {mean_diff:+.5f} & {ci_text} & {cohen:+.2f} & {p_text} \\\\"
            )
        else:
            lines.append(f"{prev_lab} $\\rightarrow$ {curr_lab} & n/a & n/a & n/a & n/a \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Sequential pairwise differences in normalized score across lab runs.}")
    lines.append("\\label{tab:pairwise-diffs}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def _equal_with_tolerance(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a in (None, "") and b in (None, ""):
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a) == str(b)


def _build_reproducibility_tex(results_rows: Sequence[Dict[str, str]], reference_rows: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("% Auto-generated by campaign_mt_export.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l c c}")
    lines.append("\\hline")
    lines.append("Field & Exact match rate & Mismatches \\\\")
    lines.append("\\hline")

    if not reference_rows:
        lines.append("No reference configured & n/a & n/a \\\\")
    else:
        key = lambda row: (str(row.get("lab_id") or "").strip(), str(row.get("seed") or "").strip())
        current_by_key = {key(row): row for row in results_rows if key(row) != ("", "")}
        ref_by_key = {key(row): row for row in reference_rows if key(row) != ("", "")}
        shared = sorted(set(current_by_key.keys()) & set(ref_by_key.keys()))
        fields = [
            "mission_score",
            "norm_score",
            "correct_edges",
            "false_edges",
            "rendezvous_success",
            "prompt_tokens_total",
            "completion_tokens_total",
            "lm_total_tokens",
        ]
        total = max(1, len(shared))
        for field in fields:
            matches = 0
            for item in shared:
                if _equal_with_tolerance(current_by_key[item].get(field), ref_by_key[item].get(field)):
                    matches += 1
            mismatches = total - matches
            lines.append(f"{_latex_escape(field)} & {matches}/{total} ({matches/total:.1%}) & {mismatches} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Reproducibility comparison against reference campaign data.}")
    lines.append("\\label{tab:repro-summary}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def _metric_series_by_lab(labs: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, str]], field: str) -> Tuple[List[str], List[List[float]]]:
    order = [str(lab.get("id") or lab.get("lab_id") or "").strip() for lab in labs]
    grouped: Dict[str, List[float]] = {lab_id: [] for lab_id in order}
    for row in rows:
        lab_id = str(row.get("lab_id") or "").strip()
        if lab_id not in grouped:
            continue
        value = _float_from_row(row, field)
        if value is not None:
            grouped[lab_id].append(value)
    labels = [lab_id for lab_id in order if grouped.get(lab_id)]
    values = [grouped[lab_id] for lab_id in labels]
    return labels, values


def _plot_box(
    labs: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, str]],
    field: str,
    ylabel: str,
    title: str,
    output: Path,
    log_scale: bool = False,
) -> Optional[Path]:
    if plt is None:
        return None
    labels, values = _metric_series_by_lab(labs, rows, field)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5))
    axis = fig.add_subplot(1, 1, 1)
    if labels and values:
        axis.boxplot(values, tick_labels=labels, showfliers=True)
    else:
        axis.text(
            0.5,
            0.5,
            "No campaign data available yet.",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title(title)
    axis.set_xlabel("Lab")
    axis.set_ylabel(ylabel)
    if log_scale and labels and values:
        axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def _build_campaign_overview_tex(campaign_name: str, manifest_path: Path, results_path: Path, lab_results_path: Path, labs: Sequence[Dict[str, Any]], seed_spec: Any) -> str:
    def _short(path: Path) -> str:
        text = str(path).replace("\\", "/")
        for marker in ("/Code/", "/Document/"):
            pos = text.find(marker)
            if pos >= 0:
                return text[pos + 1 :]
        return text

    lines: List[str] = []
    lines.append("% Auto-generated by campaign_mt_export.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l p{0.67\\linewidth}}")
    lines.append("\\hline")
    lines.append(f"Campaign & {_latex_escape(campaign_name)} \\\\")
    lines.append(f"Generated at & {_latex_escape(datetime.now().astimezone().isoformat())} \\\\")
    lines.append(f"Lab count & {_latex_escape(len(list(labs)))} \\\\")
    lines.append(f"Seed spec & {_latex_escape(json.dumps(seed_spec))} \\\\")
    lines.append(f"Campaign report & \\path{{{_latex_escape(_short(manifest_path))}}} \\\\")
    lines.append(f"Seed-level results & \\path{{{_latex_escape(_short(results_path))}}} \\\\")
    lines.append(f"Lab-level results & \\path{{{_latex_escape(_short(lab_results_path))}}} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Campaign metadata and output file traceability.}")
    lines.append("\\label{tab:campaign-overview}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def export_campaign_artifacts(
    *,
    corasat_root: Path,
    config_path: Path,
    master_config: Dict[str, Any],
    campaign_cfg: Dict[str, Any],
    output_dir: Path,
    results_path: Path,
    lab_results_path: Path,
    manifest_path: Path,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    log = logger or print

    mt_cfg = campaign_cfg.get("mt_export", {}) if isinstance(campaign_cfg.get("mt_export"), dict) else {}
    overleaf_root = _resolve_path(corasat_root, str(mt_cfg.get("overleaf_root") or ""), "../../Document/Overleaf")
    tex_generated = _resolve_path(overleaf_root, str(mt_cfg.get("tex_generated_dir") or ""), "tex/generated")
    fig_generated = _resolve_path(overleaf_root, str(mt_cfg.get("figures_generated_dir") or ""), "figures/generated")
    data_dir = _resolve_path(overleaf_root, str(mt_cfg.get("data_dir") or ""), "review/generated_data")

    tex_generated.mkdir(parents=True, exist_ok=True)
    fig_generated.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    labs = campaign_cfg.get("labs", []) if isinstance(campaign_cfg.get("labs"), list) else []
    labs = [lab for lab in labs if isinstance(lab, dict) and bool(lab.get("enabled", True))]
    seed_spec = campaign_cfg.get("seed_list")
    if seed_spec in (None, "", {}):
        sim_cfg = master_config.get("simulation", {}) if isinstance(master_config.get("simulation"), dict) else {}
        seed_spec = sim_cfg.get("seed_list")

    result_rows = _read_csv_rows(results_path)
    lab_rows = _read_csv_rows(lab_results_path)

    campaign_name = str(campaign_cfg.get("name") or campaign_cfg.get("campaign_name") or "campaign").strip() or "campaign"
    copy_dir = data_dir / campaign_name
    copy_dir.mkdir(parents=True, exist_ok=True)
    copied_results = copy_dir / "results.csv"
    copied_lab_results = copy_dir / "lab_results.csv"
    copied_manifest = copy_dir / "campaign_report.json"
    copied_repro_report = copy_dir / "reproducibility_report.json"
    copied_repro_mismatches = copy_dir / "reproducibility_mismatches.csv"
    if results_path.exists():
        shutil.copy2(results_path, copied_results)
    if lab_results_path.exists():
        shutil.copy2(lab_results_path, copied_lab_results)
    if manifest_path.exists():
        shutil.copy2(manifest_path, copied_manifest)
    repro_report_src = output_dir / "reproducibility_report.json"
    repro_mismatches_src = output_dir / "reproducibility_mismatches.csv"
    if repro_report_src.exists() and repro_report_src.is_file():
        shutil.copy2(repro_report_src, copied_repro_report)
    if repro_mismatches_src.exists() and repro_mismatches_src.is_file():
        shutil.copy2(repro_mismatches_src, copied_repro_mismatches)

    outputs: Dict[str, str] = {}

    matrix_path = tex_generated / "lab_matrix_table.tex"
    _write_text(matrix_path, _build_lab_matrix_tex(corasat_root, labs))
    outputs["lab_matrix_table"] = str(matrix_path)

    lab_table_path = tex_generated / "lab_results_table.tex"
    _write_text(lab_table_path, _build_lab_results_tex(labs, lab_rows))
    outputs["lab_results_table"] = str(lab_table_path)

    aux_path = tex_generated / "aux_metrics_table.tex"
    _write_text(aux_path, _build_aux_metrics_tex(labs, result_rows))
    outputs["aux_metrics_table"] = str(aux_path)

    pairwise_path = tex_generated / "pairwise_table.tex"
    _write_text(pairwise_path, _build_pairwise_tex(labs, result_rows))
    outputs["pairwise_table"] = str(pairwise_path)

    repro_cfg = mt_cfg.get("reproducibility", {}) if isinstance(mt_cfg.get("reproducibility"), dict) else {}
    ref_results_token = str(repro_cfg.get("reference_results_csv") or "").strip()
    if not ref_results_token:
        campaign_repro_cfg = campaign_cfg.get("reproducibility", {}) if isinstance(campaign_cfg.get("reproducibility"), dict) else {}
        ref_results_token = str(campaign_repro_cfg.get("reference_results_csv") or "").strip()
    ref_results_path: Optional[Path] = None
    ref_rows: List[Dict[str, str]] = []
    if ref_results_token:
        candidate = _resolve_path(output_dir, ref_results_token, "")
        if candidate.exists() and candidate.is_file():
            ref_results_path = candidate
            ref_rows = _read_csv_rows(candidate)
    repro_path = tex_generated / "reproducibility_table.tex"
    _write_text(repro_path, _build_reproducibility_tex(result_rows, ref_rows))
    outputs["reproducibility_table"] = str(repro_path)

    overview_path = tex_generated / "campaign_overview.tex"
    _write_text(
        overview_path,
        _build_campaign_overview_tex(
            campaign_name,
            copied_manifest if copied_manifest.exists() else manifest_path,
            copied_results if copied_results.exists() else results_path,
            copied_lab_results if copied_lab_results.exists() else lab_results_path,
            labs,
            seed_spec,
        ),
    )
    outputs["campaign_overview"] = str(overview_path)

    score_fig = _plot_box(labs, result_rows, "norm_score", "Normalized score", f"{campaign_name}: normalized score", fig_generated / "lab_score_distributions.png")
    if score_fig:
        outputs["figure_score_distributions"] = str(score_fig)
    bcast_fig = _plot_box(labs, result_rows, "broadcast_rate", "Broadcast rate", f"{campaign_name}: broadcast rate", fig_generated / "lab_broadcast_rate.png")
    if bcast_fig:
        outputs["figure_broadcast_rate"] = str(bcast_fig)
    wait_fig = _plot_box(labs, result_rows, "wait_rate", "Wait rate", f"{campaign_name}: wait rate", fig_generated / "lab_wait_rate.png")
    if wait_fig:
        outputs["figure_wait_rate"] = str(wait_fig)
    turn_fig = _plot_box(labs, result_rows, "avg_turn_duration_s", "Average turn duration (s)", f"{campaign_name}: turn duration", fig_generated / "lab_turn_duration.png", log_scale=True)
    if turn_fig:
        outputs["figure_turn_duration"] = str(turn_fig)

    outputs["copied_results_csv"] = str(copied_results)
    outputs["copied_lab_results_csv"] = str(copied_lab_results)
    outputs["copied_manifest_json"] = str(copied_manifest)
    if copied_repro_report.exists():
        outputs["copied_reproducibility_report_json"] = str(copied_repro_report)
    if copied_repro_mismatches.exists():
        outputs["copied_reproducibility_mismatches_csv"] = str(copied_repro_mismatches)
    outputs["config_used"] = str(config_path)

    log(f"MT artifacts exported to {tex_generated} and {fig_generated}.")
    return outputs
