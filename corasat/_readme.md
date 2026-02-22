# Corasat File Purpose Inventory

This file lists the purpose of each current file under `Code/corasat` and whether it is needed to run all labs via campaign and generate MT-relevant outputs.

Legend:
- `Keep: required` = required for campaign/lab execution and MT outputs.
- `Keep: optional` = useful utility or optional runtime feature.
- `Keep: generated` = generated/runtime artifact; can be cleaned.

## Root Runtime and Orchestration
- `_readme.md`: File inventory and keep/remove assessment for this folder. Keep: optional.
- `main.py`: Primary orchestrator for both single-runtime runs and full multi-lab campaigns from `config.json`. Keep: required.
- `run_campaign.py`: Deprecated compatibility wrapper that delegates to `main.py`. Keep: optional.
- `lab_state.py`: Deprecated compatibility wrapper (lab activation flow removed). Keep: optional.
- `optuna_tuner.py`: Optuna tuning runner for decision-support parameters (used when `optuna.enabled=true`). Keep: required.
- `build_lora_dataset.py`: Builds LoRA train/val datasets from simulation logs. Keep: required (for LoRA labs).
- `register_ollama_model.py`: Registers exported LoRA model in Ollama. Keep: required (for LoRA labs).
- `clean_generated_artifacts.py`: Cleanup utility for generated artifacts. Keep: optional.

## Root Configuration
- `config.json`: Master runtime+campaign config (global settings, lab matrix, and run flags). Keep: required.
- `campaign_config.json`: Legacy campaign config (superseded by `config.json`). Keep: optional.
- `lab_matrix.json`: Legacy lab matrix metadata. Keep: optional.
- `lab_overrides.json`: Legacy per-lab override store. Keep: optional.
- `config.base.json`: Legacy baseline runtime template. Keep: optional.
- `rules.txt`: Legacy fallback rules file; runtime normally uses `rules_path` from config/profile. Keep: optional.

## Core Classes
- `classes/Core.py`: Shared config loader, constants, helper functions, and global utilities used by runtime. Keep: required.
- `classes/Drone.py`: Drone agent lifecycle and per-turn behavior integration. Keep: required.
- `classes/Drone_Support.py`: Decision support, prompting context build, action parsing, memory handling, and policy helpers. Keep: required.
- `classes/Simulation.py`: Simulation loop, round/turn execution, scoring, and game-state orchestration. Keep: required.
- `classes/Exporter.py`: Logging and CSV export of run results/metrics. Keep: required.
- `classes/GUI.py`: Pygame visualization support. Keep: optional (required only when GUI runs are used).

## Profile Files
### Action Policy Profiles
- `profiles/action_policy/A0_action_policy.json`: Random legal action policy profile. Keep: required.
- `profiles/action_policy/A1_action_policy.json`: Decision-support-only policy profile. Keep: required.
- `profiles/action_policy/A2_action_policy.json`: LLM-only policy profile. Keep: required.
- `profiles/action_policy/A3_action_policy.json`: Decision-support-assisted LLM policy profile. Keep: required.

### Decision Support Profiles
- `profiles/decision_support/DS0_decision_support.json`: Decision support disabled profile. Keep: required.
- `profiles/decision_support/DS1_decision_support.json`: Original handcrafted decision-support profile. Keep: required.
- `profiles/decision_support/DS2_decision_support.json`: Optuna-1 decision-support profile. Keep: required.
- `profiles/decision_support/DS3_decision_support.json`: Optuna-2 decision-support profile. Keep: required.

### Fine-Tuning Profiles
- `profiles/fine_tuning/FT0_fine_tuning.json`: No fine-tuning profile. Keep: required.
- `profiles/fine_tuning/FT1_fine_tuning.json`: LoRA-adapted profile. Keep: required.

### Model Profiles
- `profiles/model/M0_model.json`: No-LM runtime model profile. Keep: required.
- `profiles/model/M1_model.json`: Base LM runtime model profile (`llama3.1:8b`). Keep: required.
- `profiles/model/M2_model.json`: DeepSeek model-swap profile. Keep: required.

### Prompt Request Profiles
- `profiles/prompt_requests/P0_prompt_requests.json`: No prompt requests (not applied). Keep: required.
- `profiles/prompt_requests/P1_prompt_requests.json`: Reduced prompt requests. Keep: required.
- `profiles/prompt_requests/P2_prompt_requests.json`: Baseline prompt requests. Keep: required.
- `profiles/prompt_requests/P3_prompt_requests.json`: Waypoint-clarified prompt requests. Keep: required.

### Rules Profiles
- `profiles/rules/R0_rules.txt`: No rules (empty, not applied). Keep: required.
- `profiles/rules/R1_rules.txt`: Reduced rules profile. Keep: required.
- `profiles/rules/R2_rules.txt`: Full rules profile. Keep: required.
- `profiles/rules/R3_rules.txt`: Full rules with waypoint clarification. Keep: required.

## LoRA Run Folder
- `lora/l4_lora.yaml`: Llama-Factory LoRA training config used by campaign LoRA step. Keep: required (if LoRA labs enabled).
- `lora/l4_export.yaml`: Llama-Factory export config used by campaign LoRA step. Keep: required (if LoRA labs enabled).
- `lora/dataset_info.json`: Generated dataset metadata written by dataset build step. Keep: generated.
- `lora/train_l4.log`: Training run log file. Keep: generated.

## GUI Assets
- `img/BlackBishop.png`: GUI chess piece sprite. Keep: optional.
- `img/BlackKing.png`: GUI chess piece sprite. Keep: optional.
- `img/BlackKnight.png`: GUI chess piece sprite. Keep: optional.
- `img/BlackPawn.png`: GUI chess piece sprite. Keep: optional.
- `img/BlackQueen.png`: GUI chess piece sprite. Keep: optional.
- `img/BlackRook.png`: GUI chess piece sprite. Keep: optional.
- `img/WhiteBishop.png`: GUI chess piece sprite. Keep: optional.
- `img/WhiteKing.png`: GUI chess piece sprite. Keep: optional.
- `img/WhiteKnight.png`: GUI chess piece sprite. Keep: optional.
- `img/WhitePawn.png`: GUI chess piece sprite. Keep: optional.
- `img/WhiteQueen.png`: GUI chess piece sprite. Keep: optional.
- `img/WhiteRook.png`: GUI chess piece sprite. Keep: optional.

## Runtime State and Generated Artifacts
- `.lab_state/manifest.json`: Legacy lab-state artifact from old flow. Keep: generated (safe to remove).
- `.lab_state/backup/config.json`: Legacy backup artifact from old flow. Keep: generated (safe to remove).
- `.lab_state/backup/Drone_Support.py`: Legacy backup artifact from old flow. Keep: generated (safe to remove).
- `logs/simulation_2026-02-16_01.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_02.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_03.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_04.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_05.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_06.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_07.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_08.log`: Simulation run log. Keep: generated.
- `logs/simulation_2026-02-16_09.log`: Simulation run log. Keep: generated.

## Legacy Empty Folder
- `scripts/` (currently empty): Legacy location of helper scripts, now unused. Keep: optional (safe to remove once Windows lock is released).
