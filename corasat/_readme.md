# Corasat Runtime Inventory

This file documents what remains in `Code/corasat` after cleanup for clean-slate campaign reruns.

## Required Runtime Files
- `main.py`: Single entry point. Runs `campaign.smoke_run` and/or `campaign.campaign_run` from `config.json` (smoke first if both enabled).
- `config.json`: Master config. Contains global simulation settings, campaign settings, and clustered lab matrix (`L0.x` to `L3.x`).
- `campaign_mt_export.py`: Converts campaign outputs into MT-ready LaTeX tables/figures under `Document/Overleaf`.
- `clean_generated_artifacts.py`: Clean-slate utility for generated artifacts (Corasat and optional Overleaf generated outputs).
- `optuna_tuner.py`: Optuna-based tuning utility used by labs with `optuna.enabled=true`.
- `build_lora_dataset.py`: Builds LoRA train/validation datasets from a selected lab log.
- `register_ollama_model.py`: Registers exported LoRA model artifacts in Ollama.

## Core Simulation Classes
- `classes/Core.py`: Config loading, constants, helpers, seed handling, and shared utilities.
- `classes/Simulation.py`: Simulation loop, board generation, scoring, runtime metrics, and watchdog behavior.
- `classes/Drone.py`: Drone turn orchestration and integration of support pipeline classes.
- `classes/Drone_Support.py`: Drone knowledge, mission support, decision support, LM policy logic, aftermath execution.
- `classes/Exporter.py`: Shared logger and per-seed `results.csv` persistence.
- `classes/GUI.py`: Optional Pygame GUI (disabled for campaigns by default).

## Lab Profile System
Each campaign lab references profile IDs. Profile files are global and reusable.

- Rules: `profiles/rules/R0_rules.txt`, `profiles/rules/R1_rules.txt`, `profiles/rules/R2_rules.txt`, `profiles/rules/R3_rules.txt`, `profiles/rules/R4_rules.txt`
- Prompt requests: `profiles/prompt_requests/P0_prompt_requests.json`, `profiles/prompt_requests/P1_prompt_requests.json`, `profiles/prompt_requests/P2_prompt_requests.json`, `profiles/prompt_requests/P3_prompt_requests.json`
- Decision support: `profiles/decision_support/DS0_decision_support.json`, `profiles/decision_support/DS1_decision_support.json`, `profiles/decision_support/DS2_decision_support.json`, `profiles/decision_support/DS3_decision_support.json`
- Model: `profiles/model/M0_model.json`, `profiles/model/M1_model.json`, `profiles/model/M2_model.json`
- Fine tuning: `profiles/fine_tuning/FT0_fine_tuning.json`, `profiles/fine_tuning/FT1_fine_tuning.json`
- Action policy: `profiles/action_policy/A0_action_policy.json`, `profiles/action_policy/A1_action_policy.json`, `profiles/action_policy/A2_action_policy.json`, `profiles/action_policy/A3_action_policy.json`
- Communication: `profiles/communication/C0_communication.json`, `profiles/communication/C1_communication.json`, `profiles/communication/C2_communication.json`

## LoRA Configuration Files
- `lora/train_lora.yaml`: LlamaFactory training configuration used in the LoRA lab.
- `lora/export_lora.yaml`: LlamaFactory export configuration used after LoRA training.

## Static Assets
- `img/*.png`: Chess figure sprites used only when GUI is enabled.

## Generated/Runtime Directories (Not Source)
These are created during execution and can be cleaned:
- `campaign_runs/`
- `logs/`
- `.lab_state/` (legacy artifacts)
- `lora/` generated datasets, outputs, checkpoints, and Modelfiles.

## Entry Commands
- Smoke + full campaign (as configured): `python main.py --config config.json`
- Cleanup dry run: `python clean_generated_artifacts.py`
- Cleanup apply (include Overleaf generated artifacts): `python clean_generated_artifacts.py --apply --include-overleaf`
