# plot.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define labels for each run
labels = {
    "run_0": "Baseline",
    "run_1": "Proposed Experiment",
    "run_2": "Increased Poly Order",
    "run_3": "Decreased Threshold"
}

# 1) Find all run directories
run_dirs = sorted(d for d in os.listdir('.') if d.startswith('run_') and d in labels)

# 2) Collect RMSEs
rmse_list = []
for run in run_dirs:
    info_path = os.path.join(run, "final_info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    rmse_list.append(info["rmse"]["means"])

# 3) Pick the best run (lowest RMSE) for the detailed plots
best_idx = np.argmin(rmse_list)
best_run = run_dirs[best_idx]

# 4) Load the time‐series arrays from the best run
t_train = np.load(os.path.join(best_run, "t_train.npy"))
x_train = np.load(os.path.join(best_run, "x_train.npy"))
x_sim   = np.load(os.path.join(best_run, "x_sim.npy"))

# 5) Load discovered coefficients
coef_path = os.path.join(best_run, "coefficients.json")
with open(coef_path, "r") as f:
    coef_dict = json.load(f)
terms  = list(coef_dict.keys())
values = list(coef_dict.values())

# 6) Make the 2×2 figure
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# (a) Time‐series: x0 and x1 over time
ax = axs[0, 0]
ax.plot(t_train, x_train[:, 0], "r-", label="x0 (true)", linewidth=2)
ax.plot(t_train, x_sim[:, 0], "r--", label="x0 (model)", linewidth=2)
ax.plot(t_train, x_train[:, 1], "b-", label="x1 (true)", linewidth=2)
ax.plot(t_train, x_sim[:, 1], "b--", label="x1 (model)", linewidth=2)
ax.set(title="Time Series (Best Run)", xlabel="t", ylabel="x")
ax.legend()

# (b) Phase portrait: x0 vs. x1
ax = axs[0, 1]
ax.plot(x_train[:, 0], x_train[:, 1], "r-", label="true", linewidth=2)
ax.plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", linewidth=2)
ax.set(title="Phase Portrait (Best Run)", xlabel="x0", ylabel="x1")
ax.legend()

# (c) RMSE vs. run number
ax = axs[1, 0]
x = range(len(run_dirs))
ax.bar(x, rmse_list)
ax.set_xticks(x)
ax.set_xticklabels([labels[run] for run in run_dirs], rotation=45, ha='right')
ax.set(title="RMSE per Run", xlabel="Run", ylabel="RMSE")
for i, v in enumerate(rmse_list):
    ax.text(i, v, f'{v:.6f}', ha='center', va='bottom')

# (d) Discovered coefficient magnitudes
ax = axs[1, 1]
indices = np.arange(len(terms))
ax.bar(indices, np.abs(values))
ax.set_xticks(indices)
ax.set_xticklabels(terms, rotation=90, fontsize=8)
ax.set(title=f"Learned SINDy Coefficients (Best Run: {labels[best_run]})")
ax.set_yscale('log')

plt.tight_layout()
plt.savefig("results.png", dpi=300, bbox_inches='tight')

print(f"Best run: {labels[best_run]}")
print(f"Best RMSE: {min(rmse_list):.6f}")
