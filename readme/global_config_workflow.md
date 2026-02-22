# Global Config Workflow

Corasat uses a global configuration model.

Core files:
- `corasat/config.base.json`: baseline runtime config template.
- `corasat/lab_overrides.json`: per-lab config overrides.
- `corasat/lab_matrix.json`: lab IDs and profile allocations.
- `corasat/campaign_config.json`: campaign run selection and seed ranges.

Profiles:
- `corasat/profiles/rules/R*.txt`
- `corasat/profiles/prompt_requests/P*.json`
- `corasat/profiles/model/M*.json`
- `corasat/profiles/fine_tuning/FT*.json`
- `corasat/profiles/action_policy/A*.json`

Activation flow:
1. Select lab by `lab` or `lab_id`.
2. Merge `config.base.json` + selected override from `lab_overrides.json`.
3. Resolve `rules_id` from `lab_matrix.json` to `profiles/rules/Rx_rules.txt`.
4. Write merged runtime config to `corasat/config.json`.

Main entry points:
- `corasat/lab_state.py`: activate/list/run/restore lab states.
- `corasat/run_campaign.py`: executes full campaigns and writes reports.
