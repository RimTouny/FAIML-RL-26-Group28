# Part 2 – Sim-to-Real Transfer with Stable-Baselines3
This folder trains and evaluates PPO/SAC agents on **PandaPush-v3** for the sim-to-real transfer part of the project. The code supports:

- PPO and SAC training with Stable-Baselines3.
- Source and target domains.
- Lower/upper bound experiments.
- Uniform Domain Randomization (UDR).
- Automatic Domain Randomization (ADR).
- 50-episode evaluation.
- Sensitivity analysis over cube mass.

---

## Files
```text
part2/
├── train_sb3.py          # Training script for PPO/SAC, UDR, ADR
├── eval_sb3.py           # Evaluation and mass sensitivity analysis
├── rand_wrapper.py       # UDR/ADR mass randomization wrapper
├── compare_plot.py       # Training-curve comparison plot
├── analysis_plots.py     # Domain gap, sensitivity, UDR, ADR plots
├── test_random_policy.py # Random policy test, if included
├── Logs/                 # Training logs and plots
├── models/               # Saved models
└── results/              # Summaries, sensitivity CSVs, randomization CSVs
```

---

## Install
```bash
pip install torch stable-baselines3[extra] gymnasium panda-gym numpy matplotlib
```

If your project includes a local `panda-gym/` package:
```bash
pip install -e panda-gym/
```


## Task 3 – Environment exploration
```bash
# ── TASK 3 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
python test_random_policy.py
python test_random_policy.py --env-type target
```

```bash
# ── TASK 4 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
## Task 4 – PPO vs SAC comparison
```bash
python train_sb3.py --algo ppo --env-type source --timesteps 500000 --seed 42 67 128  # Train PPO on the source domain
python train_sb3.py --algo sac --env-type source --timesteps 500000 --seed 42 67 128   #Train SAC on the source domain
```
### The best model is copied to:
```text
models/<algo>_push_none_source_best/model_<algo>_push_none_source_best.zip
```
### Sensitivity Evaluation:
```bash
python eval_sb3.py --model-path models/ppo_push_none_source_best/model_ppo_push_none_source_best.zip --env-type target --episodes 50 
python eval_sb3.py --model-path models/sac_push_none_source_best/model_sac_push_none_source_best.zip --env-type target --episodes 50 

python eval_sb3.py --model-path models/ppo_push_none_source_best/model_ppo_push_none_source_best.zip --episodes 50 --sensitivity
python eval_sb3.py --model-path models/sac_push_none_source_best/model_sac_push_none_source_best.zip --episodes 50 --sensitivity
```
```bash
# ── TASK 5 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
## Task 5 – Lower (Source → Source and Source → Target) and Upper bound (Target → Target) baselines
```bash
python train_sb3.py --algo sac --env-type source --timesteps 500000 --seed 42 67 128   # (Train on source- Already ran on Task4)
python train_sb3.py --algo sac --env-type target --timesteps 500000 --seed 42 67 128   # Train on Target
```
### Evaluation :
```bash
python eval_sb3.py --model-path models/sac_push_none_source_best/model_sac_push_none_source_best.zip --env-type source --episodes 50   # Evaluate source-trained model on source (Lower bound)
python eval_sb3.py --model-path models/sac_push_none_source_best/model_sac_push_none_source_best.zip --env-type target --episodes 50   # Evaluate source-trained model on target (Lower bound)
python eval_sb3.py --model-path models/sac_push_none_target_best/model_sac_push_none_target_best.zip --env-type target --episodes 50   # Evaluate target-trained model on target(Upper bound)
python eval_sb3.py --model-path models/sac_push_none_source_best/model_sac_push_none_source_best.zip --episodes 50  --sensitivity # Sensitivity analysis
```
```bash
# ── TASK 6 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
## Task 6 – Uniform Domain Randomization-Train SAC with UDR:
```bash
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --udr-low 0.1 --udr-high 10.0 --timesteps 500000 --seed 42 67 128
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --udr-low 0.5 --udr-high 6.0 --timesteps 500000 --seed 42 67 128
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --udr-low 1.0 --udr-high 5.0  --timesteps 500000 --seed 42 67 128
 #A narrower range such as `[1.0, 5.0]` can be tested, but avoid presenting it as the main result because it uses strong knowledge of the target mass.
```
### Evaluate UDR model:
```bash
python eval_sb3.py --model-path models/sac_push_udr_source_best_0.1_10.0/sac_push_udr_source_best_0.1_10.0.zip --env-type target --episodes 50
python eval_sb3.py --model-path models/sac_push_udr_source_best_0.5_6.0/ssac_push_udr_source_best_0.5_6.0.zip --env-type target --episodes 50
python eval_sb3.py --model-path models/sac_push_udr_source_best_1.0_0.5/sac_push_udr_source_best_1.0_0.5.zip --env-type target --episodes 50

```

```bash
# ── TASK 7 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
## Task 7 – Automatic Domain Randomization- Train SAC with ADR:
```bash
python train_sb3.py --algo sac --env-type source --sampling-strategy adr --timesteps 500000 --seed 42 67 128
```

### Evaluate ADR model:
```bash
python eval_sb3.py --model-path models/sac_push_adr_source_best/model_sac_push_adr_source_best.zip --env-type source --episodes 50
python eval_sb3.py --model-path models/sac_push_adr_source_best/model_sac_push_adr_source_best.zip --env-type target --episodes 50
python eval_sb3.py --model-path models/sac_push_adr_source_best/model_sac_push_adr_source_best.zip --episodes 50 --sensitivity # Sensitivity analysis
```


## Generate final plots- After training and evaluation:
```bash
python compare_plot.py
python analysis_plots.py
```

---

## Outputs:
```text
Logs/comparison_all_configurations.png
Logs/domain_gap.png
Logs/sensitivity_overlay.png
Logs/udr_mass_histogram.png
Logs/adr_evolution.png
```


### Output naming
The clean best model is saved as:
```text
models/<algo>_push_<strategy>_<env>_best/model_<algo>_push_<strategy>_<env>_best.zip
```

Examples:
```text
models/sac_push_none_source_best/model_sac_push_none_source_best.zip
models/sac_push_none_target_best/model_sac_push_none_target_best.zip
models/sac_push_udr_source_best/model_sac_push_udr_source_best.zip
models/sac_push_adr_source_best/model_sac_push_adr_source_best.zip
```

The script also saves:
```text
results/summary_<tag>.txt
results/sensitivity_<model>.csv
results/udr_mass_samples_<low>_<high>.csv
results/adr_history.csv
```


## Ranking rule
`train_sb3.py` selects the best hyperparameter configuration using:
1. Highest average success rate across seeds.
2. Highest average return as a tie-breaker.

This avoids misleading selection when all models have the same success rate, especially if all success rates are 0%.

---

## Report note
For the report, evaluate every final selected model over **50 episodes** and report:
- Mean return.
- Standard deviation of return.
- Success rate.
- Source → Source.
- Source → Target.
- Target → Target.
- UDR → Target.
- ADR → Target.