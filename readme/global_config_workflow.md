# Global Config Workflow

Corasat now uses a single master config file:
- `corasat/config.json`

## Configuration model
`config.json` contains:
- global runtime config (`board`, `simulation`, `decision_support`, `prompt_requests`),
- campaign controls (`campaign.*`),
- full lab list (`campaign.labs`).

Each lab is a compact tuple of profile IDs:
- `rules_id` (`R*`)
- `prompt_id` (`P*`)
- `drone_support_id` (`DS*`)
- `model_id` (`M*`)
- `fine_tuning_id` (`FT*`)
- `action_policy_id` (`A*`)
- `communication_id` (`C*`)

Additional lab graph fields:
- `parent_lab`: Declared parent for one-change-at-a-time comparisons.
- `reference_lab`: Reuse an earlier lab's results without rerunning the simulation.
- `changed_item`: Human-readable marker for the intended intervention.
- `evaluate_runtime`: Optional flag for preparation-only labs that run setup steps without executing seeds.

Optional per-lab execution blocks:
- `optuna.enabled`
- `lora.enabled` + command list

## Runtime flow (`main.py`)
1. Load `config.json`.
2. Validate the enabled lab graph, profile references, and one-change parent-child comparisons.
3. For each enabled lab, build runtime config by resolving profile IDs.
4. Run optional Optuna and LoRA stages for that lab.
5. If `evaluate_runtime=true`, execute the configured seeds and write seed, lab, and campaign outputs.
6. If `reference_lab` is set, copy the referenced metrics into the current lab entry without rerunning.
7. Export MT-ready tables and figures when `campaign.mt_export.enabled=true`.

## Outputs
All run artifacts now live under one directory:
- `corasat/campaign_runs/<campaign_name>/...`

Important subpaths inside a run directory:
- `runtime_configs/`: Resolved per-lab runtime configs.
- `labs/<lab_id>/`: Per-lab reports and per-seed reports.
- `runtime_logs/<lab_id>/`: Verbose lab logs, per-seed logs, and LM conversation traces.

MT-ready generated artifacts are written to:
- `Document/Overleaf/tex/generated/`
- `Document/Overleaf/figures/generated/`
- `Document/Overleaf/review/generated_data/<campaign_name>/`
