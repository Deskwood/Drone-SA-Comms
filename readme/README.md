# Corasat Quick Start

## What this repository does
Corasat runs reproducible multi-drone simulation campaigns for the thesis.
A single `config.json` defines:
- global runtime settings,
- all lab runs (`L0` to `L14`),
- optional Optuna and LoRA stages,
- MT export targets for Overleaf.

## Main entrypoint
- Run campaign: `python corasat/main.py --config corasat/config.json`
- Run cleanup dry-run: `python corasat/clean_generated_artifacts.py`
- Run cleanup apply (including Overleaf generated files):
  `python corasat/clean_generated_artifacts.py --apply --include-overleaf`

## Campaign output levels
- Campaign level: `campaign_runs/<campaign_name>/campaign.log`, `campaign_report.json`
- Lab level: `campaign_runs/<campaign_name>/labs/<lab_id>/lab.log`, `lab_report.json`
- Seed level: `campaign_runs/<campaign_name>/labs/<lab_id>/seed_reports/seed_XXXX.json` and per-seed log files

## MT integration
When `campaign.mt_export.enabled=true`, the run also writes:
- LaTeX tables to `Document/Overleaf/tex/generated/`
- Figures to `Document/Overleaf/figures/generated/`
- copied campaign data to `Document/Overleaf/review/generated_data/<campaign_name>/`

## Profile IDs
Every lab references profile IDs:
- Rules: `R0..R3`
- Prompt requests: `P0..P3`
- Decision support: `DS0..DS3`
- Model: `M0..M2`
- Fine tuning: `FT0..FT1`
- Action policy: `A0..A3`

This keeps each lab definition compact and traceable.
