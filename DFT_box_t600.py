# dft_rhoee_box_t600_fix.py
import os, re, glob
import numpy as np
import matplotlib.pyplot as plt

ROOTS = {"FD + KO3": "output_from_fd_box", "FV + WENO7": "output_from_fv_box"}
JOB_ID = "2000_1_0.2"
TARGET_T = 600.0
BOX_L, BOX_R = -100.0, 100.0
SAVE_FIG = "dft_rhoee_t600_fixed.png"
USE_LOGY = True

# ---------- IO ----------
def read_job_config(job_dir):
    path = os.path.join(job_dir, "job.config")
    if not os.path.isfile(path): raise FileNotFoundError(path)
    raw = {}
    for line in open(path, "r", encoding="utf-8", errors="ignore"):
        if ":" not in line: continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            raw[k] = float(v) if any(c in v for c in ".eE") else int(v)
        except ValueError:
            raw[k] = v
    out = {}
    for key in ("z0","z1","nz","nvz","dt"):
        if key not in raw: raise KeyError(f"missing {key} in {path}")
        out[key] = float(raw[key]) if key in ("z0","z1","dt") else int(raw[key])
    return out

def list_rho_files(job_dir): return sorted(glob.glob(os.path.join(job_dir, "rho_*.dat")))
def rho_n_from_path(p): 
    m = re.search(r"rho_(\d+)\.dat$", os.path.basename(p))
    return int(m.group(1)) if m else None

def pick_best_snapshot(job_dir, dt, target_t):
    files = list_rho_files(job_dir)
    if not files: raise FileNotFoundError(f"No rho_*.dat in {job_dir}")
    n0 = int(round(target_t/dt))
    p0 = os.path.join(job_dir, f"rho_{n0}.dat")
    if os.path.exists(p0): return p0, n0, n0*dt
    cand = []
    for p in files:
        n = rho_n_from_path(p); 
        if n is None: continue
        t = n*dt; cand.append((abs(t-target_t), p, n, t))
    _, p, n, t = min(cand, key=lambda x: x[0])
    return p, n, t

def load_rho(path):
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path} has <3 cols")
    z = arr[:,0]; v = arr[:,1]; rho = arr[:,2]
    return z, v[0], rho

# ---------- physics helpers ----------
def periodic_top_hat(z, t, v, z0, z1, zL0=BOX_L, zR0=BOX_R):
    L = z1 - z0
    left = zL0 + v*t; right = zR0 + v*t
    wrap = lambda x: ((x - z0) % L) + z0
    Lm, Rm = wrap(left), wrap(right)
    if Lm <= Rm: mask = (z >= Lm) & (z <= Rm)
    else:        mask = (z >= Lm) | (z <= Rm)
    return mask.astype(float)

def periodic_interp(z_src, f_src, z_dst, z0, z1):
    """Periodic, linear interpolation f_src(z_src) -> f_dst(z_dst)."""
    L = z1 - z0
    # Unwrap both to [0,L) so boundary is continuous
    x_src = (z_src - z0) % L
    x_dst = (z_dst - z0) % L
    # Ensure monotonic source grid for interp
    order = np.argsort(x_src)
    x_src, f_src = x_src[order], f_src[order]
    # Extend by one point for periodic closure
    x_ext = np.concatenate([x_src, [x_src[0] + L]])
    f_ext = np.concatenate([f_src, [f_src[0]]])
    # Interpolate on [0,L), values wrap automatically via extension
    return np.interp(x_dst, x_ext, f_ext)

# ---------- DFT (textbook) ----------
def build_dft_matrix(z, L):
    """
    DFT matrix W for samples at z_j (assumed uniform spacing), using
      k_m = 2π m / L,  m = -N/2, ..., N/2-1, and F = W @ f.
    NOTE: Use *L from config* (z1 - z0), not dz*N from the file.
    """
    N = z.size
    # indices centered at 0 (even N)
    m = np.arange(-N//2, N//2, dtype=int)
    k = 2.0*np.pi*m / L
    phase = -1j * np.outer(k, z)   # shape (N, N)
    W = np.exp(phase) / N          # 1/N normalization => DC ≈ mean
    return k, W

def dft_profile(z_ref, f_ref, k=None, W=None, L=None):
    if (k is None) or (W is None) or (L is None):
        raise ValueError("k, W, and L must be pre-built for consistency")
    return k, (W @ f_ref)

# ---------- main ----------
def main():
    # Load both profiles
    data = {}
    for label, root in ROOTS.items():
        job_dir = os.path.join(root, JOB_ID)
        if not os.path.isdir(job_dir):
            print(f"[WARN] {job_dir} missing; skip {label}"); continue
        cfg = read_job_config(job_dir)
        path, n, t = pick_best_snapshot(job_dir, cfg["dt"], TARGET_T)
        z, v, rho = load_rho(path)
        data[label] = {"z": z, "v": v, "rho": rho, "cfg": cfg, "n": n, "t": t}
        print(f"{label}: {os.path.basename(path)}  n={n}  t≈{t:.6f}  v={v:.6f}")

    if not data: 
        raise RuntimeError("No data found")

    # Reference grid (prefer FD)
    ref = "FD + KO3" if "FD + KO3" in data else next(iter(data))
    z_ref  = data[ref]["z"]
    cfg    = data[ref]["cfg"]
    z0,z1  = cfg["z0"], cfg["z1"]
    L_conf = z1 - z0               # **USE CONFIG L**
    v_ref  = data[ref]["v"]
    t_ref  = data[ref]["t"]

    # One exact on ref grid
    rho_ex = periodic_top_hat(z_ref, t_ref, v_ref, z0, z1)

    # Periodic‑aware interpolation for the other profile(s)
    for label, d in data.items():
        if np.array_equal(d["z"], z_ref):
            d["rho_on_ref"] = d["rho"]
        else:
            d["rho_on_ref"] = periodic_interp(d["z"], d["rho"], z_ref, z0, z1)

        # Sanity prints (these should look ~min≈0, max≈1, mean≈width/L ≈ 0.1)
        r = d["rho_on_ref"]
        print(f"{label}: min={r.min():.3g}, max={r.max():.3g}, mean={r.mean():.6g}")

    print(f"Exact: min={rho_ex.min():.3g}, max={rho_ex.max():.3g}, mean={rho_ex.mean():.6g}")

    # Build DFT matrix ONCE with L from config
    k, W = build_dft_matrix(z_ref, L_conf)

    # DFT spectra
    curves = []
    for label, d in data.items():
        _, F = dft_profile(z_ref, d["rho_on_ref"], k=k, W=W, L=L_conf)
        curves.append((f"{label} (numerical)", np.abs(F)))
    _, Fex = dft_profile(z_ref, rho_ex, k=k, W=W, L=L_conf)
    curves.append(("Exact (periodic, single)", np.abs(Fex)))

    # Plot
    plt.figure(figsize=(12,5))
    for lab, mag in curves:
        style = "--" if "Exact" in lab else "-"
        plt.plot(k, mag, style, lw=1.3, label=lab)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$|\tilde{\rho}_{ee}(k)|$")
    plt.title(r"DFT of $\rho_{ee}(z)$ at $t \approx 600$ (periodic, single exact)")
    if USE_LOGY: plt.yscale("log")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    #plt.savefig(SAVE_FIG, dpi=220)
    print(f"Saved {SAVE_FIG}")
    plt.show()

if __name__ == "__main__":
    main()
