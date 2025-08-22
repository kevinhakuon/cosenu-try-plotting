# plot_rhoee_box_t600_with_error.py
import os, re, glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------- User settings ----------------
ROOTS = {
    "FD + KO3": "output_from_fd_box",
    "FV + WENO7": "output_from_fv_box",
}
JOB_ID = "2000_1_0.2"
TARGET_T = 600.0
BOX_L, BOX_R = -100.0, 100.0
SAVE_FIG = "rhoee_vs_z_t600_with_error.png"

# ---------------- Helpers ----------------
def read_job_config(job_dir):
    path = os.path.join(job_dir, "job.config")
    cfg = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line: continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            try:
                cfg[k] = float(v) if any(c in v for c in ".eE") else int(v)
            except ValueError:
                cfg[k] = v
    return {"z0": float(cfg["z0"]), "z1": float(cfg["z1"]),
            "nz": int(cfg["nz"]), "nvz": int(cfg["nvz"]), "dt": float(cfg["dt"])}

def list_rho_files(job_dir): return sorted(glob.glob(os.path.join(job_dir, "rho_*.dat")))
def rho_n_from_path(p): 
    m = re.search(r"rho_(\d+)\.dat$", os.path.basename(p))
    return int(m.group(1)) if m else None

def load_rho(path):
    arr = np.loadtxt(path)
    z, v, rho = arr[:,0], arr[:,1], arr[:,2]
    return z, v[0], rho

def periodic_top_hat(z, t, v, z0, z1, zL0=BOX_L, zR0=BOX_R):
    L = z1 - z0
    wrap = lambda x: ((x - z0) % L) + z0
    Lm, Rm = wrap(zL0 + v*t), wrap(zR0 + v*t)
    if Lm <= Rm: mask = (z >= Lm) & (z <= Rm)
    else:        mask = (z >= Lm) | (z <= Rm)
    return mask.astype(float)

def pick_best_snapshot(job_dir, dt, target_t):
    files = list_rho_files(job_dir)
    if not files: raise FileNotFoundError
    ideal_n = int(round(target_t/dt))
    ideal_path = os.path.join(job_dir, f"rho_{ideal_n}.dat")
    if os.path.exists(ideal_path): return ideal_path, ideal_n, ideal_n*dt
    cand = [(abs(n*dt - target_t), p, n, n*dt) for p in files if (n:=rho_n_from_path(p)) is not None]
    _, p, n, t = min(cand, key=lambda x:x[0])
    return p, n, t

# ---------------- Main ----------------
def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={'height_ratios':[2,1]})

    for label, root in ROOTS.items():
        job_dir = os.path.join(root, JOB_ID)
        if not os.path.isdir(job_dir): continue
        cfg = read_job_config(job_dir)
        path, n, t_act = pick_best_snapshot(job_dir, cfg["dt"], TARGET_T)
        z, vbin, rho_num = load_rho(path)
        rho_ex = periodic_top_hat(z, t_act, vbin, cfg["z0"], cfg["z1"])

        # --- Top panel: rho vs z ---
        ax1.plot(z, rho_num, lw=1.4, label=f"{label} (n={n}, t≈{t_act:.3f})")
        ax1.plot(z, rho_ex, ls="--", lw=1.0, alpha=0.6,
                 label=f"Exact (periodic) — {label}")

        # --- Bottom panel: error ---
        eps = rho_ex - rho_num
        ax2.plot(z, eps, lw=1.2, label=f"ε = exact − {label}")

    # Cosmetics
    ax1.set_ylabel(r"$\rho_{ee}$")
    ax1.set_title(r"Box-wave advection at $t \approx 600$")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("z")
    ax2.set_ylabel(r"$\epsilon$")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(SAVE_FIG, dpi=220)
    print(f"Saved {SAVE_FIG}")
    plt.show()

if __name__ == "__main__":
    main()
