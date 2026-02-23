# Code Repository File Map

This file describes each tracked file in `Code/` and its purpose for campaign execution and thesis data generation.

## Root Files
- `.gitignore`: Ignore rules for generated artifacts, logs, model outputs, checkpoints, and local caches.
- `LICENSE`: License text for the repository.
- `setup_venv.ipynb`: Notebook with environment setup/bootstrap steps.
- `_readme.md` (this file): Global file-purpose index.

## Documentation Folder
- `readme/README.md`: Human-facing overview of the codebase and usage flow.
- `readme/global_config_workflow.md`: Notes on global config and campaign orchestration design.
- `readme/decision_support_parameters.md`: Explanation of decision-support scoring parameters.

## Corasat Runtime Root
- `corasat/config.json`: Master runtime and campaign config (global settings + all lab definitions L0-L14).
- `corasat/main.py`: Main execution entrypoint for single run and campaign mode.
- `corasat/campaign_mt_export.py`: MT export layer from campaign outputs to Overleaf generated tables/figures.
- `corasat/clean_generated_artifacts.py`: Utility to clean generated runtime/MT artifacts.
- `corasat/optuna_tuner.py`: Optuna tuning routine for decision-support parameters.
- `corasat/build_lora_dataset.py`: Converts lab logs into LoRA train/validation datasets.
- `corasat/register_ollama_model.py`: Registers exported LoRA model in Ollama.
- `corasat/_readme.md`: Corasat-specific file and architecture inventory.

## Corasat Core Classes
- `corasat/classes/Core.py`: Shared config loader, constants, board utilities, and helper functions.
- `corasat/classes/Simulation.py`: Simulation orchestration, board generation, scoring, rendezvous checks, metrics.
- `corasat/classes/Drone.py`: Drone turn pipeline integration.
- `corasat/classes/Drone_Support.py`: Decision support, LM policy, mission handling, memory and action execution.
- `corasat/classes/Exporter.py`: Shared logging and per-seed results CSV export.
- `corasat/classes/GUI.py`: Optional GUI implementation.

## Corasat Profiles
### Rules Profiles
- `corasat/profiles/rules/R0_rules.txt`: No-rules profile.
- `corasat/profiles/rules/R1_rules.txt`: Reduced rules profile.
- `corasat/profiles/rules/R2_rules.txt`: Full rules profile.
- `corasat/profiles/rules/R3_rules.txt`: Full rules with waypoint clarification.

### Prompt Request Profiles
- `corasat/profiles/prompt_requests/P0_prompt_requests.json`: No prompt-request guidance.
- `corasat/profiles/prompt_requests/P1_prompt_requests.json`: Reduced prompt requests.
- `corasat/profiles/prompt_requests/P2_prompt_requests.json`: Full prompt requests.
- `corasat/profiles/prompt_requests/P3_prompt_requests.json`: Full prompt requests with waypoint clarification.

### Decision Support Profiles
- `corasat/profiles/decision_support/DS0_decision_support.json`: Decision support disabled.
- `corasat/profiles/decision_support/DS1_decision_support.json`: Original handcrafted scoring.
- `corasat/profiles/decision_support/DS2_decision_support.json`: Optuna-tuned profile stage 1.
- `corasat/profiles/decision_support/DS3_decision_support.json`: Optuna-tuned profile stage 2.

### Model Profiles
- `corasat/profiles/model/M0_model.json`: No-LM runtime model profile.
- `corasat/profiles/model/M1_model.json`: Llama 3.1 8B model profile.
- `corasat/profiles/model/M2_model.json`: DeepSeek model profile.

### Fine-Tuning Profiles
- `corasat/profiles/fine_tuning/FT0_fine_tuning.json`: No fine-tuning.
- `corasat/profiles/fine_tuning/FT1_fine_tuning.json`: LoRA-adapted runtime profile.

### Action Policy Profiles
- `corasat/profiles/action_policy/A0_action_policy.json`: Random legal policy.
- `corasat/profiles/action_policy/A1_action_policy.json`: Decision-support-only policy.
- `corasat/profiles/action_policy/A2_action_policy.json`: LLM-only policy.
- `corasat/profiles/action_policy/A3_action_policy.json`: Decision-support-assisted LLM policy.

## LoRA Config Files
- `corasat/lora/train_lora.yaml`: LlamaFactory training config for campaign LoRA stage.
- `corasat/lora/export_lora.yaml`: LlamaFactory export config for campaign LoRA stage.

## GUI Assets
- `corasat/img/BlackBishop.png`: Black bishop sprite.
- `corasat/img/BlackKing.png`: Black king sprite.
- `corasat/img/BlackKnight.png`: Black knight sprite.
- `corasat/img/BlackPawn.png`: Black pawn sprite.
- `corasat/img/BlackQueen.png`: Black queen sprite.
- `corasat/img/BlackRook.png`: Black rook sprite.
- `corasat/img/WhiteBishop.png`: White bishop sprite.
- `corasat/img/WhiteKing.png`: White king sprite.
- `corasat/img/WhiteKnight.png`: White knight sprite.
- `corasat/img/WhitePawn.png`: White pawn sprite.
- `corasat/img/WhiteQueen.png`: White queen sprite.
- `corasat/img/WhiteRook.png`: White rook sprite.
