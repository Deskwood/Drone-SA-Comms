# Code Repository Structure

This document consolidates the old repository file map and the Corasat runtime inventory.
It describes the source layout under `Code/` and the purpose of the main runtime files.

## Root `Code/` Folder
- `.gitignore`: Ignore rules for generated artifacts, logs, model outputs, checkpoints, and local caches.
- `LICENSE`: Repository license text.
- `corasat/`: Main simulator, campaign runner, profiles, and generated runtime outputs.
- `readme/`: Human-facing documentation for the codebase.

## Documentation Folder
- `readme/README.md`: Quick start and documentation index.
- `readme/repository_structure.md`: This merged repository and runtime structure map.
- `readme/global_config_workflow.md`: Notes on the global config and campaign orchestration flow.
- `readme/decision_support_parameters.md`: Explanation of decision-support scoring parameters.

## Corasat Runtime Root
- `corasat/config.json`: Master runtime and campaign config. Defines global settings, lab families (`L0.x` to `L3.x`), optional Optuna stages, optional LoRA stages, and MT export targets.
- `corasat/main.py`: Campaign execution entrypoint.
- `corasat/campaign_mt_export.py`: Exports campaign outputs into MT-ready Overleaf tables and figures.
- `corasat/optuna_tuner.py`: Optuna-based decision-support tuning routine.
- `corasat/build_lora_dataset.py`: Converts campaign logs into LoRA train/validation datasets.
- `corasat/register_ollama_model.py`: Registers exported LoRA model artifacts in Ollama.
- `corasat/analyze_ds_dependence.py`: Aggregates LM-vs-DS agreement metrics from campaign traces.

## Core Simulation Classes
- `corasat/classes/Core.py`: Config loading, constants, board utilities, and shared helpers.
- `corasat/classes/Simulation.py`: Simulation loop, board generation, scoring, metrics, and watchdog behavior.
- `corasat/classes/Drone.py`: Drone turn orchestration.
- `corasat/classes/Drone_Support.py`: Decision support, LM prompting, memory handling, and action execution.
- `corasat/classes/Exporter.py`: Shared logger and per-seed `results.csv` persistence.
- `corasat/classes/GUI.py`: Optional Pygame GUI.

## Profile System
Each campaign lab references reusable profile IDs instead of embedding full settings inline.

- Shared output structure: `corasat/profiles/structure/output_contract.txt`
- Rules: `corasat/profiles/rules/R0_rules.txt` to `corasat/profiles/rules/R4_rules.txt`
- Prompt requests: `corasat/profiles/prompt_requests/P0_prompt_requests.json` to `corasat/profiles/prompt_requests/P4_prompt_requests.json`
- Decision support: `corasat/profiles/decision_support/DS0_decision_support.json` to `corasat/profiles/decision_support/DS6_decision_support.json`
- Model: `corasat/profiles/model/M0_model.json` to `corasat/profiles/model/M2_model.json`
- Fine tuning: `corasat/profiles/fine_tuning/FT0_fine_tuning.json` to `corasat/profiles/fine_tuning/FT1_fine_tuning.json`
- Action policy: `corasat/profiles/action_policy/A0_action_policy.json` to `corasat/profiles/action_policy/A3_action_policy.json`
- Communication: `corasat/profiles/communication/C0_communication.json` to `corasat/profiles/communication/C2_communication.json`

## LoRA and Assets
- `corasat/lora/`: Training configs, exports, merged model artifacts, and related LoRA outputs.
- `corasat/img/`: Chess figure sprites used when the GUI is enabled.

## Generated Runtime Directories
These directories are runtime outputs rather than hand-maintained source files.

- `corasat/campaign_runs/`: One folder per campaign run. Each run contains the results CSVs, reports, runtime configs, per-lab reports, seed reports, and nested `runtime_logs/`.
- `corasat/__pycache__/`: Python bytecode cache.

## Main Entry Command
- Run the configured campaign flow: `python corasat/main.py --config corasat/config.json`
