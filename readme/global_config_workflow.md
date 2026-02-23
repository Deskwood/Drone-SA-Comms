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

Optional per-lab execution blocks:
- `optuna.enabled`
- `lora.enabled` + command list

## Runtime flow (`main.py`)
1. Load `config.json`.
2. For each enabled lab, build runtime config by resolving profile IDs.
3. Run optional Optuna and LoRA stages for that lab.
4. Execute configured seeds and write seed/lab/campaign logs and reports.
5. Export MT-ready tables/figures when `campaign.mt_export.enabled=true`.

## Outputs
Primary outputs are written to:
- `corasat/campaign_runs/<campaign_name>/...`

MT-ready generated artifacts are written to:
- `Document/Overleaf/tex/generated/`
- `Document/Overleaf/figures/generated/`
- `Document/Overleaf/review/generated_data/<campaign_name>/`
