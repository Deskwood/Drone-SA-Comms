# Corasat Code Guide

## What this repository does
Corasat runs reproducible multi-drone simulation campaigns for the thesis.
A single `corasat/config.json` defines:
- global runtime settings,
- lab families (`L0.x` to `L3.x`),
- optional Optuna and LoRA stages,
- MT export targets for Overleaf.

## Main entrypoint
- Run the configured campaign flow: `python corasat/main.py --config corasat/config.json`

## One Campaign Run Directory
Each run now keeps its structured outputs and runtime logs together:
- `corasat/campaign_runs/<campaign_name>/campaign.log`
- `corasat/campaign_runs/<campaign_name>/results.csv`
- `corasat/campaign_runs/<campaign_name>/lab_results.csv`
- `corasat/campaign_runs/<campaign_name>/campaign_report.json`
- `corasat/campaign_runs/<campaign_name>/error_summary.json`, `error_summary.csv`
- `corasat/campaign_runs/<campaign_name>/runtime_configs/`
- `corasat/campaign_runs/<campaign_name>/labs/<lab_id>/...`
- `corasat/campaign_runs/<campaign_name>/runtime_logs/<lab_id>/...`

## MT integration
When `campaign.mt_export.enabled=true`, the run also writes:
- LaTeX tables to `Document/Overleaf/tex/generated/`
- Figures to `Document/Overleaf/figures/generated/`
- copied campaign data to `Document/Overleaf/review/generated_data/<campaign_name>/`

## Documentation in This Folder
- `repository_structure.md`: Consolidated source-tree and runtime inventory.
- `global_config_workflow.md`: Global config and campaign orchestration notes.
- `decision_support_parameters.md`: Decision-support scoring explanation.

## Profile ID Families
Every lab references compact profile IDs:
- Rules: `R0..R4`
- Prompt requests: `P0..P4`
- Decision support: `DS0..DS6`
- Model: `M0..M2`
- Fine tuning: `FT0..FT1`
- Action policy: `A0..A3`
- Communication: `C0..C2`
