import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from scipy.odr import ODR, Model, RealData

# Physical constants 
eta   = 8.9e-4   # Pa·s  (water viscosity at room temp)
r     = 2.08e-6  # m     (bead radius = 4.16/2 μm)
dr    = 0.105e-6 # m     (uncertainty in radius from ±0.21 μm diameter)
gamma = 6 * np.pi * eta * r  # drag coefficient [N·s/m]

# Load data 
df = pd.read_csv("./figures/results_summary.csv")
df.columns = df.columns.str.strip()
# Use absolute velocity
df["v"]    = df["Velocity (μm/sec)"].abs() * 1e-6          # m/s
df["dv"]   = df["Velocity Uncertainty (μm/sec)"].abs() * 1e-6
df["x"]    = df["Range of Motion (microns)"] * 1e-6        # m
df["dx"]   = df["Range of Motion Uncertainty (microns)"] * 1e-6

# Compute F and propagate uncertainty 
# F = gamma * v,  sigma_F = gamma * sigma_v  (dominant term)
df["F"]    = gamma * df["v"] * 1e12    # pN
df["dF"]   = gamma * df["dv"] * 1e12  # pN
df["x_um"] = df["x"] * 1e6            # back to μm for display
df["dx_um"]= df["dx"] * 1e6

# Per-flow-rate means and SEMs 
grouped = df.groupby("Flow Rate (μL/min)")
summary = grouped.agg(
    x_mean=("x_um", "mean"), x_sem=("x_um", lambda s: s.std(ddof=1)/np.sqrt(len(s))),
    F_mean=("F",    "mean"), F_sem=("F",    lambda s: s.std(ddof=1)/np.sqrt(len(s))),
).reset_index()

print("\n── Per flow rate summary ──────────────────────────────────────────────")
print(f"{'Flow':>6} {'x̄ (μm)':>10} {'σx̄':>8} {'F̄ (pN)':>10} {'σF̄':>8}")
for _, row in summary.iterrows():
    print(f"{row['Flow Rate (μL/min)']:>6.0f} {row['x_mean']:>10.4f} {row['x_sem']:>8.4f} "
          f"{row['F_mean']:>10.4f} {row['F_sem']:>8.4f}")

# ODR fit (errors in both x and F) 
def linear(B, x):
    return B[0]*x + B[1]

odr_model = Model(linear)
odr_data  = RealData(df["x_um"].values, df["F"].values,
                     sx=df["dx_um"].values, sy=df["dF"].values)
odr_obj   = ODR(odr_data, odr_model, beta0=[1.0, 0.0])
odr_res   = odr_obj.run()

k_fit     = odr_res.beta[0]   # pN/μm
b_fit     = odr_res.beta[1]
dk_fit    = odr_res.sd_beta[0]
db_fit    = odr_res.sd_beta[1]

# Also simple linregress for R²
slope_lr, intercept_lr, r_val, _, _ = linregress(df["x_um"].values, df["F"].values)
R2 = r_val**2

print(f"\n── Linear fit (ODR) F = k·x + b ──────────────────────────────────────")
print(f"  k = {k_fit:.4f} ± {dk_fit:.4f}  pN/μm")
print(f"  b = {b_fit:.4f} ± {db_fit:.4f}  pN")
print(f"  R² = {R2:.4f}")

# Max trapping force 
# Last STABLE flow rate before 135 drops — use 130 μL/min
df_130 = df[df["Flow Rate (μL/min)"] == 130]
F_max_mean = df_130["F"].mean()
F_max_sem  = df_130["F"].std(ddof=1) / np.sqrt(len(df_130))
print(f"\n── Max trapping force (130 μL/min) ───────────────────────────────────")
print(f"  F_max = {F_max_mean:.4f} ± {F_max_sem:.4f}  pN")

# Plot 
fig = plt.figure(figsize=(10, 8))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Panel 1: F vs x (main)
ax1 = fig.add_subplot(gs[0, :])
ax1.errorbar(df["x_um"], df["F"], xerr=df["dx_um"], yerr=df["dF"],
             fmt='o', alpha=0.45, color='steelblue', capsize=3, label="Individual")
ax1.errorbar(summary["x_mean"], summary["F_mean"],
             xerr=summary["x_sem"], yerr=summary["F_sem"],
             fmt='s', color='navy', capsize=5, markersize=7, label="Mean ± SEM", zorder=5)
x_fit = np.linspace(0, df["x_um"].max()*1.15, 200)
ax1.plot(x_fit, k_fit*x_fit + b_fit, 'r-', lw=2,
         label=f"ODR fit: k = {k_fit:.3f} ± {dk_fit:.3f} pN/μm\n$R^2$ = {R2:.3f}")
ax1.fill_between(x_fit,
                 (k_fit-dk_fit)*x_fit + b_fit,
                 (k_fit+dk_fit)*x_fit + b_fit,
                 alpha=0.15, color='red', label="±1σ band")
ax1.set_xlabel("Displacement x (μm)", fontsize=12)
ax1.set_ylabel("Drag Force F (pN)", fontsize=12)
ax1.set_title("Optical Trap: Force vs Displacement", fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: displacement per flow rate
ax2 = fig.add_subplot(gs[1, 0])
ax2.errorbar(summary["Flow Rate (μL/min)"], summary["x_mean"],
             yerr=summary["x_sem"], fmt='o-', color='teal', capsize=4)
ax2.set_xlabel("Flow Rate (μL/min)")
ax2.set_ylabel("Mean Displacement (μm)")
ax2.set_title("Displacement vs Flow Rate")
ax2.grid(True, alpha=0.3)

# Panel 3: force per flow rate
ax3 = fig.add_subplot(gs[1, 1])
ax3.errorbar(summary["Flow Rate (μL/min)"], summary["F_mean"],
             yerr=summary["F_sem"], fmt='o-', color='darkorange', capsize=4)
ax3.set_xlabel("Flow Rate (μL/min)")
ax3.set_ylabel("Mean Drag Force (pN)")
ax3.set_title("Drag Force vs Flow Rate")
ax3.grid(True, alpha=0.3)

plt.suptitle("Optical Tweezers", fontsize=14, fontweight='bold', y=1.01)
plt.savefig("./optical_tweezers_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved.")