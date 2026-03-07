"""Create an Ollama model from a Llama-Factory export YAML.

This script is intentionally lightweight and avoids extra dependencies.
It reads `export_dir` from the given YAML, writes a Modelfile, and runs:

    ollama create <model-name> -f <modelfile>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict


CORASAT_ROOT = Path(__file__).resolve().parent


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (CORASAT_ROOT / path).resolve()


def _parse_simple_yaml(path: Path) -> Dict[str, str]:
    """Parse flat `key: value` YAML files used by export configs."""
    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            out[key] = value
    return out


def _write_modelfile(modelfile_path: Path, export_dir: Path) -> None:
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    export_modelfile = export_dir / "Modelfile"
    if export_modelfile.exists():
        content = export_modelfile.read_text(encoding="utf-8")
        from_line = f"FROM {export_dir.as_posix()}"
        content = content.replace("FROM .", from_line, 1)
    else:
        content = f"FROM {export_dir.as_posix()}\n"
    modelfile_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Register an exported LoRA model in Ollama.")
    parser.add_argument("--export-yaml", required=True, help="Path to Llama-Factory export YAML.")
    parser.add_argument("--model-name", required=True, help="Target Ollama model name.")
    parser.add_argument("--modelfile", default="lora/Modelfile.l4", help="Output Modelfile path.")
    parser.add_argument("--dry-run", action="store_true", help="Write Modelfile but do not call Ollama.")
    parser.add_argument(
        "--skip-export-check",
        action="store_true",
        help="Allow missing export_dir path (useful for config smoke tests).",
    )
    args = parser.parse_args()

    export_yaml_path = _resolve_path(args.export_yaml)
    if not export_yaml_path.exists():
        raise SystemExit(f"Export YAML not found: {export_yaml_path}")

    payload = _parse_simple_yaml(export_yaml_path)
    export_dir_raw = (payload.get("export_dir") or "").strip()
    if not export_dir_raw:
        raise SystemExit(f"Missing 'export_dir' in export YAML: {export_yaml_path}")

    export_dir = _resolve_path(export_dir_raw)
    if not args.skip_export_check and not export_dir.exists():
        raise SystemExit(
            f"Export directory not found: {export_dir}\n"
            "Run the LoRA export step before model registration, or use --skip-export-check."
        )

    modelfile_path = _resolve_path(args.modelfile)
    _write_modelfile(modelfile_path, export_dir)
    print(f"[ok] Wrote Modelfile: {modelfile_path}")

    cmd = ["ollama", "create", str(args.model_name), "-f", str(modelfile_path)]
    if args.dry_run:
        print("[dry-run] " + " ".join(cmd))
        return 0

    try:
        subprocess.run(cmd, cwd=str(CORASAT_ROOT), check=True)
    except FileNotFoundError:
        raise SystemExit("Ollama CLI not found in PATH. Install Ollama or run with --dry-run.")

    print(f"[ok] Registered Ollama model: {args.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
