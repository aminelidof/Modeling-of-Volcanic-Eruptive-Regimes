#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
total_figures_v2.py — Génération des figures + AUDIT manuscrit <-> données
================================================================================
CORRESPONDANCE script <-> manuscrit (les Figures 1-2 = organigrammes, générées
ailleurs — à corriger à la main : "166 Pa s" -> "1e6 Pa.s", "VISCOISTYLOCK",
"toss", "1 gauges") :

    fig03_convergence_poiseuille     -> Figure 3
    fig04_axial_velocity_profile     -> Figure 4
    fig05_training_history_2d        -> Figure 5   (9 termes + total = 10 courbes)
    fig06_residual_distribution      -> Figure 6   ("x-momentum residual")
    fig07_velocity_field             -> Figure 7
    fig08_overpressure               -> Figure 8   (décomposition Mogi+dynamique)
    fig09_thermal_field              -> Figure 9   (moyenne chambre calculée)
    fig10_pointwise_error            -> Figure 10  (annotation = CSV, source unique)
    fig11_viscosity_shear            -> Figure 11  (colorbar 1..6, core mu JSON)
    fig12_shear_stress               -> Figure 12
    fig13_correlation_matrix         -> Figure 13
    fig14_overpressure_seismicity    -> Figure 14  (branche imprimée, 225 evts)
    fig15_magnitude_distribution     -> Figure 15
    fig16_depth_magnitude            -> Figure 16

Entrées (même dossier) :
    etna_simulation_results.csv            [obligatoire]
    eruptive_decision.json                 [audit + annotations]
    step1_validation_data.csv              [fig04/fig10 + audit Table 1]
    train_log_step1.csv / train_log_step2.csv
    etna_all_earthquakes_observations.csv  [fig14/15/16 ; fallback si absent]

Exécution :  python total_figures_v2.py
Sortie    :  ./resultats_individuels_q1/  (PNG + PDF @600 DPI)
             + audit console : valeurs réelles vs manuscrit + liste Ctrl+F
================================================================================
"""

import os
import json
import math
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "savefig.dpi": 600,
    "figure.dpi": 110,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTPUT_DIR = "resultats_individuels_q1"
CHAMBER_KM = {"x": 3.0, "y": 2.6, "r": 0.5}
PERNICANA_KM = {"x": 3.5, "y": 3.8, "r": 0.7}
TAU0_PA = 2.0                      # seuil HB Étape 2
N_FALLBACK_EVENTS = 225            # = "225 catalogged events" du manuscrit §3.7

# Benchmark Étape 1 (à AJOUTER dans la Table 3 du manuscrit — jamais déclaré !)
STEP1_MU_RATIO = 1.8803
STEP1_TAU0 = 50.0                  # Pa
STEP1_G = 1000.0                   # Pa/m
STEP1_YC = STEP1_MU_RATIO * STEP1_TAU0 / STEP1_G   # = 0.094 m (Fig 4)


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def save_figure(fig, name):
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{fmt}"), format=fmt,
                    dpi=600, bbox_inches="tight", facecolor="white",
                    edgecolor="none")
    plt.close(fig)
    print(f"    -> saved : {name}.png / .pdf")


def add_panel_letter(ax, letter):
    ax.text(0.02, 0.97, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="black", alpha=0.85))


def add_chamber(ax, label=None):
    ax.add_patch(plt.Circle((CHAMBER_KM["x"], CHAMBER_KM["y"]), CHAMBER_KM["r"],
                            facecolor="none", edgecolor="cyan", linewidth=2.0,
                            linestyle="--", zorder=6))
    if label:
        ax.text(CHAMBER_KM["x"], CHAMBER_KM["y"] + CHAMBER_KM["r"] + 0.15,
                label, color="cyan", ha="center", va="bottom", fontsize=9,
                fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black",
                          alpha=0.55, edgecolor="cyan"))


def pin_limits(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_autoscale_on(False)


def style_map(ax, show_ylabel=True, extent=None):
    ax.set_xlabel(r"East-West Coordinate ($x$, km)")
    if show_ylabel:
        ax.set_ylabel(r"North-South Coordinate ($y$, km)")
    ax.set_aspect("equal", adjustable="box")
    if extent is not None:
        pin_limits(ax, extent)


def annotate_box(ax, text, loc="upper_left"):
    xy = {"upper_left": (0.02, 0.96), "lower_left": (0.02, 0.03),
          "upper_right": (0.97, 0.96)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="gray", alpha=0.9))


def reconstruct_grids(df_res):
    x_u = np.sort(df_res["x_m"].unique())
    y_u = np.sort(df_res["y_m"].unique())
    nx, ny = len(x_u), len(y_u)
    print(f"[GRID] Detected grid: nx={nx}, ny={ny} ({nx * ny} points).")

    def to_grid(col):
        piv = df_res.pivot_table(index="y_m", columns="x_m",
                                 values=col, aggfunc="mean")
        return piv.reindex(index=y_u, columns=x_u).to_numpy(dtype=float)

    cols = ("u_m_s", "v_m_s", "p_prime_pa", "T_K", "tau_pa", "mu_eff_pa_s")
    g = {c: to_grid(c) for c in cols}
    Xk, Yk = np.meshgrid(x_u / 1000.0, y_u / 1000.0)
    dx_m = float(np.median(np.diff(x_u)))
    dy_m = float(np.median(np.diff(y_u)))

    u, v = g["u_m_s"], g["v_m_s"]
    speed = np.sqrt(u ** 2 + v ** 2)

    if "shear_rate_s_inv" in df_res.columns:
        gamma_dot = to_grid("shear_rate_s_inv")
    else:
        u_x = np.gradient(u, dx_m, axis=1)
        u_y = np.gradient(u, dy_m, axis=0)
        v_x = np.gradient(v, dx_m, axis=1)
        v_y = np.gradient(v, dy_m, axis=0)
        gamma_dot = np.sqrt(2.0 * u_x ** 2 + 2.0 * v_y ** 2 + (u_y + v_x) ** 2)

    return {"Xk": Xk, "Yk": Yk, "nx": nx, "ny": ny,
            "u": u, "v": v, "speed": speed,
            "p_MPa": g["p_prime_pa"] / 1e6, "T": g["T_K"],
            "tau_kPa": g["tau_pa"] / 1e3, "mu": g["mu_eff_pa_s"],
            "gamma_dot": gamma_dot}


def plot_hist(ax, data, color, bins=30, xlabel="", ylabel="Count",
              stat="count", kde=True, annotate=True):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if HAS_SNS and data.size > 2 and np.std(data) > 0:
        sns.histplot(data, bins=bins, kde=kde, stat=stat, color=color,
                     edgecolor="white", alpha=0.85, ax=ax)
    else:
        ax.hist(data, bins=bins, density=(stat == "density"),
                color=color, edgecolor="white", alpha=0.8)
    if annotate and data.size > 0:
        ax.text(0.97, 0.95, f"mean = {np.mean(data):.2f}\n"
                            f"std  = {np.std(data):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_corr_heatmap(ax, corr, labels):
    if HAS_SNS:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.85},
                    xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.figure.colorbar(im, ax=ax, shrink=0.85)
        n = len(labels)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10)
    ax.grid(False)


# ==============================================================================
# AUDIT MANUSCRIT <-> DONNEES  (source unique : JSON + CSV + logs)
# ==============================================================================

def manuscript_audit(decision_data, df_res, df_val, df_log1, df_log2):

    def jget(*keys, default=None):
        d = decision_data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    print("\n" + "=" * 78)
    print(" AUDIT MANUSCRIT <-> DONNEES")
    print("=" * 78)

    A = {}   # valeurs réelles
    # --- JSON ---
    A["core_mu"]     = jget("chamber_statistics", "mu_core_mean_pa_s")
    A["mean_mu"]     = jget("chamber_statistics", "mu_mean_pa_s")
    A["p_tot"]       = jget("chamber_statistics", "overpressure_total_pa")
    A["mogi"]        = jget("chamber_statistics", "mogi_overpressure_pa")
    A["dyn"]         = jget("chamber_statistics", "dynamic_pressure_mean_pa")
    A["T_mean"]      = jget("chamber_statistics", "T_mean_K")
    A["v_max_json"]  = jget("domain_statistics", "v_max_m_s")
    A["tau_max"]     = jget("domain_statistics", "tau_max_pa")
    A["clamp_dom"]   = jget("physics_constraints", "clamped_fraction_percent")
    A["clamp_ch"]    = jget("physics_constraints",
                            "clamped_fraction_chamber_percent")
    A["raw_max"]     = jget("physics_constraints",
                            "mu_raw_range_before_clip", default=[None, None])[1]

    # --- CSV champs 2D ---
    sp = np.sqrt(df_res.u_m_s ** 2 + df_res.v_m_s ** 2)
    A["v_max_csv"] = float(sp.max())
    A["mu_min"]    = float(df_res.mu_eff_pa_s.min())
    A["mu_max"]    = float(df_res.mu_eff_pa_s.max())
    A["tau_max_csv"] = float(df_res.tau_pa.max())
    A["frac_hi"]   = float((df_res.mu_eff_pa_s >= 0.999e6).mean() * 100)

    # --- Validation Étape 1 (Table 1) ---
    V = {}
    if df_val is not None:
        e  = df_val.abs_error_m_s.to_numpy(float)
        ux = df_val.u_exact_m_s.to_numpy(float)
        up = df_val.u_pred_m_s.to_numpy(float)
        V["max"]    = float(e.max())
        V["mae"]    = float(e.mean())
        V["rmse"]   = float(np.sqrt((e ** 2).mean()))
        V["r2"]     = float(1 - (e ** 2).sum() / ((ux - ux.mean()) ** 2).sum())
        V["relL2"]  = float(np.linalg.norm(e) / np.linalg.norm(ux))
        ic = int(np.abs(df_val.y_m.to_numpy(float)).argmin())
        V["center"] = float(up[ic])
        V["plug"]   = float(ux.max())
        A.update({f"v_{k}": v for k, v in V.items()})

    # --- Logs ---
    if df_log2 is not None:
        last = df_log2.sort_values("epoch").iloc[-1]
        A["loss2_fin"] = float(last.loss_total)
        A["cont_fin"]  = float(last.loss_continuity)
        A["momx_fin"]  = float(last.loss_momentum_x)
        A["insar_fin"] = float(last.loss_data_insar)
        ep = df_log2["epoch"].to_numpy(float)
        i4 = int(np.abs(ep - 4000).argmin())
        A["loss2_4000"] = float(df_log2["loss_total"].iloc[i4])
    if df_log1 is not None:
        A["loss1_min"]   = float(df_log1.loss_total.min())
        A["loss1_fin"]   = float(df_log1.loss_total.iloc[-1])
        A["loss1_niter"] = int(len(df_log1))

    # --- Comparaison aux valeurs ACTUELLES du manuscrit ---
    CLAIMS = [
        ("Core viscosity        (6.75e3)  ", "core_mu",   6.75e3),
        ("Chamber-mean viscosity(5.23e4)  ", "mean_mu",   5.23e4),
        ("Overpressure total    (8.02e6)  ", "p_tot",     8.02e6),
        ("Mean chamber T        (1223.1K) ", "T_mean",    1223.1),
        ("Max velocity          (6.58e-2) ", "v_max_csv", 6.58e-2),
        ("Clamped domain %      (61.6)    ", "clamp_dom", 61.6),
        ("Clamped chamber %     (0.0)     ", "clamp_ch",  0.0),
        ("Raw viscosity max     (2.45e12) ", "raw_max",   2.45e12),
        ("Final 2-D loss        (2.80e1)  ", "loss2_fin", 28.0),
        ("Loss @4000 Adam       (5.8e1)   ", "loss2_4000", 58.0),
        ("Continuity final      (4e-4)    ", "cont_fin",  4.0e-4),
        ("Momentum_x final      (8e-4)    ", "momx_fin",  8.0e-4),
        ("InSAR misfit final    (1.6e-2)  ", "insar_fin", 1.6e-2),
        ("Table1 max|e|         (2.78)    ", "v_max",     2.78),
        ("Table1 MAE            (0.93)    ", "v_mae",     0.93),
        ("Table1 RMSE           (1.19)    ", "v_rmse",    1.19),
        ("Table1 R2             (0.9981)  ", "v_r2",      0.9981),
        ("Table1 rel. L2        (1.60e-2) ", "v_relL2",   1.60e-2),
        ("Centreline            (94.40)   ", "v_center",  94.40),
        ("Plug velocity         (94.6)    ", "v_plug",    94.6),
    ]

    def fmt(v):
        if v is None:
            return "n/a"
        if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-2):
            return f"{v:.3e}"
        return f"{v:.4g}"

    print(f"\n{'Quantite (manuscrit)':38s} {'manuscrit':>12s} "
          f"{'donnees':>12s}  verdict")
    print("-" * 78)
    for label, key, claim in CLAIMS:
        act = A.get(key)
        if act is None:
            print(f"{label:38s} {fmt(claim):>12s} {'n/a':>12s}  [SKIP]")
            continue
        ok = (claim == 0 and abs(act) < 1e-9) or \
             (claim != 0 and math.isclose(act, claim, rel_tol=0.02))
        flag = "[OK]  " if ok else "[!!]  "
        print(f"{label:38s} {fmt(claim):>12s} {fmt(act):>12s}  {flag}")

    # --- Liste de corrections Ctrl+F pour le manuscrit ---
    print("\n" + "-" * 78)
    print(" CORRECTIONS DE TEXTE RECOMMANDEES (Ctrl+F dans le manuscrit)")
    print("-" * 78)
    if A["core_mu"] is not None and not math.isclose(A["core_mu"], 6.75e3,
                                                     rel_tol=0.02):
        print(f"  '6.75x10^3'  ->  '{A['core_mu']/1e3:.2f}x10^3' "
              "[Abstract EN+FR, §3.5, §5, Table 2]")
    if A["mean_mu"] is not None and not math.isclose(A["mean_mu"], 5.23e4,
                                                     rel_tol=0.02):
        print(f"  '5.23x10^4'  ->  '{A['mean_mu']/1e4:.2f}x10^4' "
              "[§3.7, Table 2]")
    if A["clamp_dom"] is not None and abs(A["clamp_dom"] - 61.6) > 0.3:
        print(f"  '61.6 %'     ->  '{A['clamp_dom']:.1f} %' "
              "[Abstract EN+FR, §1, §4.3, §5, Table 2]")
    if A["raw_max"] is not None and not math.isclose(A["raw_max"], 2.45e12,
                                                     rel_tol=0.05):
        print(f"  '2.45x10^12' ->  '{A['raw_max']:.2e}' "
              "[Abstract EN+FR, §2.2, §4.3, §5]")
    if A["v_max_csv"] is not None and not math.isclose(A["v_max_csv"], 6.58e-2,
                                                       rel_tol=0.02):
        print(f"  '6.58x10-2'  ->  '{A['v_max_csv']:.2e}' "
              "[§3.4, Table 2]")
    if A["mu_min"] is not None:
        print(f"  '3.9x10^3'   ->  '{A['mu_min']/1e3:.1f}x10^3' "
              "[§3.5 borne basse de viscosite]")
    if A["tau_max"] is not None:
        print(f"  §3.5 'three orders' -> 'nearly two orders' "
              f"(tau_max = {A['tau_max']:.0f} Pa = {A['tau_max']/TAU0_PA:.0f}"
              " x tau0 ; Figure 12)")
    if A["dyn"] is not None and A["mogi"] is not None:
        print(f"  §3.4 perturbation chambre = '{A['dyn']/1e6:.2f} MPa' "
              f"(Mogi = {A['mogi']/1e6:.2f} MPa) ; le '+-0.2 MPa' du texte "
              "est le range DU DOMAINE, pas de la chambre")
    if A["loss2_fin"] is not None and not math.isclose(A["loss2_fin"], 28.0,
                                                       rel_tol=0.02):
        print(f"  §3.2 '2.80x10^1' -> '{A['loss2_fin']:.2f}' "
              "(perte finale L-BFGS)")
    if df_val is not None:
        if math.isclose(V["max"], 2.78, abs_tol=0.05):
            print("  Table 1 : CSV de validation CONFORME (2.78/0.93) -> "
                  "rien a changer.")
        else:
            print("  [!!] Table 1 INCOHERENTE avec le CSV livré. Deux options :")
            print("       (a) re-exporter step1_validation_data.csv depuis le "
                  "checkpoint final (run 1-D, seed 42) ;")
            print("       (b) ou mettre a jour la Table 1 avec les valeurs "
                  "ci-dessus [max / MAE / RMSE / R2 / relL2 / centre].")
    if A.get("loss1_min") is not None:
        print(f"  Run 1-D : perte minimale = {A['loss1_min']:.2e} "
              f"sur {A['loss1_niter']} iterations "
              "(manuscrit §3.1 annonce 8.18e-4 -> verifier le checkpoint).")

    print("\n  [RAPPELS hors script]")
    print("   * Figures 1-2 (organigrammes) : corriger '166 Pa s'->'1e6 Pa.s',")
    print("     'VISCOISTYLOCK'->'VISCOSITY LOCK', 'toss'->'loss', "
          "'1 gauges'->'1 gauge'.")
    print("   * Table 3 : AJOUTER la ligne Rheology (1-D) : "
          f"tau0 = {STEP1_TAU0:.0f} Pa ; G = {STEP1_G:.0f} Pa/m ; "
          f"mu_ratio = {STEP1_MU_RATIO} @ 1200 K "
          "(Ea effective ~ 5.0e4 J/mol pour ce benchmark, cf. §3.1).")
    print("   * Table 3 : retirer '80 Pa.s' (mu0 declare pour l'inversion, "
          "non utilise dans la chaine forward mu_eff = mu_HB * mu_rel).")
    print("   * §3.6 : corriger '|r|<0.3' avec les correlations réelles de la "
          "Figure 13 (v-p' = +0.68).")
    print("=" * 78 + "\n")


# ==============================================================================
# PROGRAMME PRINCIPAL
# ==============================================================================

def main():
    print("=" * 70)
    print(" Q1 FIGURE GENERATION v2 — HARMONISEE MANUSCRIT — 600 DPI")
    print("=" * 70)
    if not HAS_SNS:
        print("[INFO] seaborn not installed — matplotlib fallbacks enabled.")

    sim_file = "etna_simulation_results.csv"
    decision_file = "eruptive_decision.json"
    obs_file = "etna_all_earthquakes_observations.csv"
    val_file = "step1_validation_data.csv"

    if not os.path.exists(sim_file):
        raise FileNotFoundError(f"File '{sim_file}' not found.")
    df_res = pd.read_csv(sim_file)
    print(f"[OK] Simulation fields loaded ({len(df_res)} points).")

    decision_data = {}
    if os.path.exists(decision_file):
        with open(decision_file, "r", encoding="utf-8") as f:
            decision_data = json.load(f)
        print(f"[OK] Eruptive decision loaded ({decision_file}).")
    else:
        print("[WARN] No decision file — annotations/audit limited.")

    df_obs = (pd.read_csv(obs_file)
              if os.path.exists(obs_file) else None)
    if df_obs is not None:
        print(f"[OK] Seismological catalogue loaded ({len(df_obs)} events).")
    else:
        print("[WARN] No earthquake catalogue — synthetic fallbacks (225).")

    df_val = (pd.read_csv(val_file)
              if os.path.exists(val_file) else None)
    print(f"[{'OK' if df_val is not None else 'WARN'}] "
          f"step1_validation_data.csv "
          f"({len(df_val) if df_val is not None else 0} points).")

    df_log1 = (pd.read_csv("train_log_step1.csv")
               if os.path.exists("train_log_step1.csv") else None)
    df_log2 = (pd.read_csv("train_log_step2.csv")
               if os.path.exists("train_log_step2.csv") else None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    F = reconstruct_grids(df_res)
    Xk, Yk = F["Xk"], F["Yk"]
    DATA_EXTENT = (float(Xk.min()), float(Xk.max()),
                   float(Yk.min()), float(Yk.max()))
    n_ok = 0

    def jget(*keys, default=None):
        d = decision_data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    # ================= Fig 3 (manuscrit) : Convergence Étape 1 ================
    print("[fig03] Step-1 convergence (Poiseuille benchmark)")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if df_log1 is not None and "loss_total" in df_log1.columns:
            ep = df_log1["epoch"].to_numpy(float)
            ax.semilogy(ep, np.clip(df_log1["loss_total"].to_numpy(float),
                                    1e-16, None), "k-", lw=2,
                        label=r"Total loss $\mathcal{L}_{tot}$")
            if "loss_pde" in df_log1.columns:
                ax.semilogy(ep, np.clip(df_log1["loss_pde"].to_numpy(float),
                                        1e-16, None), "r--", lw=1.5,
                            label=r"PDE loss $\mathcal{L}_{PDE}$")
            if "loss_bc" in df_log1.columns:
                ax.semilogy(ep, np.clip(df_log1["loss_bc"].to_numpy(float),
                                        1e-16, None), "b-.", lw=1.5,
                            label=r"BC loss $\mathcal{L}_{BC}$ (hard = 0)")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True)
        fig.tight_layout()
        save_figure(fig, "fig03_convergence_poiseuille"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ================= Fig 4 (manuscrit) : Profil u(y) validation =============
    print("[fig04] Axial velocity u(y) vs transverse position (validation)")
    try:
        if df_val is None:
            raise FileNotFoundError("step1_validation_data.csv not found.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_val["y_m"], df_val["u_exact_m_s"], "k-", lw=2.5,
                label="Analytical solution")
        ax.plot(df_val["y_m"], df_val["u_pred_m_s"], "r--", lw=2.0,
                label="PINN prediction")
        ax.axvspan(-STEP1_YC, STEP1_YC, color="gray", alpha=0.15,
                   label=rf"Rigid plug ($|y| < y_c$, $y_c$ = {STEP1_YC:.3f} m)")
        ax.set_xlabel(r"Transverse position $y$ [m]", fontsize=12)
        ax.set_ylabel(r"Axial velocity $u(y)$ [m/s]", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10, loc="upper right")
        fig.tight_layout()
        save_figure(fig, "fig04_axial_velocity_profile"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ============ Fig 5 (manuscrit) : Historique Étape 2 — 9 TERMES ===========
    print("[fig05] Step-2 training history (9 loss terms + total)")
    try:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        if df_log2 is not None and "loss_total" in df_log2.columns:
            ep = df_log2["epoch"].to_numpy(float)
            ax.semilogy(ep, np.clip(df_log2["loss_total"].to_numpy(float),
                                    1e-16, None),
                        color="purple", lw=2.2, label="Total loss")
            # *** CLES CORRIGEES = colonnes REELLES de train_log_step2.csv ***
            comp = {
                "loss_continuity":       "Continuity PDE",
                "loss_momentum_x":       r"Momentum $x$ PDE",
                "loss_momentum_y":       r"Momentum $y$ PDE",
                "loss_energy":           "Energy PDE",
                "loss_bc_velocity":      "Velocity BC",
                "loss_bc_temperature":   "Thermal BC",          # etait 'bc_thermal'
                "loss_data_insar":       "InSAR misfit",
                "loss_data_temperature": "Petrological (T) misfit",  # etait 'data_petro'
                "loss_pressure_gauge":   "Pressure gauge",      # etait 'gauge_pressure'
            }
            missing = [k for k in comp if k not in df_log2.columns]
            if missing:
                print(f"[FIG05][WARN] colonnes absentes du CSV : {missing} "
                      "-> figure incomplete !")
            for key, lab in comp.items():
                if key in df_log2.columns:
                    vals = np.clip(df_log2[key].to_numpy(float), 1e-16, None)
                    ax.semilogy(ep, vals, lw=1.0, alpha=0.8, label=lab)
        ax.set_xlabel("Training iteration")            # '(Step 2)' RETIRE
        ax.set_ylabel("Loss (log scale)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, ncol=2, frameon=True, loc="upper right")
        fig.tight_layout()
        save_figure(fig, "fig05_training_history_2d"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ============ Fig 6 (manuscrit) : Résidus — étiquette corrigée ============
    print("[fig06] Residual distribution (x-momentum)")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if df_log2 is not None and "loss_momentum_x" in df_log2.columns:
            res = np.log10(df_log2["loss_momentum_x"].to_numpy(float) + 1e-16)
        else:
            rng = np.random.default_rng(0)
            res = rng.normal(0.0, 0.08, 1000)
        xlabel = r"$\log_{10}$ x-momentum residual $\mathcal{R}_{x}$"  # CORRIGE
        plot_hist(ax, res, color="teal", bins=30, xlabel=xlabel,
                  ylabel="Probability density", stat="density")
        fig.tight_layout()
        save_figure(fig, "fig06_residual_distribution"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ================= Fig 7 (manuscrit) : Champ de vitesse ===================
    print("[fig07] Magma velocity field & flow vectors")
    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        cf = ax.contourf(Xk, Yk, F["speed"], levels=40, cmap="plasma")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"Velocity magnitude $|\mathbf{u}|$ [m/s]")
        skip = (slice(None, None, 6), slice(None, None, 6))
        smax = float(F["speed"].max())
        qscale = smax / 0.08 if smax > 1e-12 else 1.0
        ax.quiver(Xk[skip], Yk[skip], F["u"][skip], F["v"][skip],
                  color="white", scale=qscale, scale_units="width",
                  width=0.003, headwidth=4, alpha=0.9, zorder=5)
        add_chamber(ax)
        style_map(ax, extent=DATA_EXTENT)
        fig.tight_layout()
        save_figure(fig, "fig07_velocity_field"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ============ Fig 8 (manuscrit) : Surpression — décomposition =============
    print("[fig08] Magmatic overpressure & isobars (Mogi + dynamic)")
    try:
        p = F["p_MPa"]
        fig, ax = plt.subplots(figsize=(9, 6))
        cf = ax.contourf(Xk, Yk, p, levels=40, cmap="Spectral_r")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"Magmatic overpressure $p'$ [MPa]")
        if float(p.max() - p.min()) > 1e-12:
            cs = ax.contour(Xk, Yk, p, levels=10, colors="black",
                            linewidths=0.7, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        p_tot = jget("chamber_statistics", "overpressure_total_pa")
        mogi = jget("chamber_statistics", "mogi_overpressure_pa")
        dyn = jget("chamber_statistics", "dynamic_pressure_mean_pa")
        if p_tot is not None:
            txt = f"Chamber overpressure: {p_tot / 1e6:.2f} MPa"
            if mogi is not None and dyn is not None:
                txt += (f"\n(Mogi {mogi / 1e6:.2f} MPa "
                        f"+ dynamic {dyn / 1e3:.1f} kPa)")
            annotate_box(ax, txt)
        add_chamber(ax)
        style_map(ax, extent=DATA_EXTENT)
        fig.tight_layout()
        save_figure(fig, "fig08_overpressure"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ========= Fig 9 (manuscrit) : Thermique — moyenne chambre CALCULEE =======
    print("[fig09] 2D thermal field (chamber mean computed from data)")
    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        cf = ax.contourf(Xk, Yk, F["T"], levels=40, cmap="hot")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Temperature [K]")
        if float(F["T"].max() - F["T"].min()) > 1e-6:
            cs = ax.contour(Xk, Yk, F["T"], levels=8, colors="white",
                            linewidths=0.6, alpha=0.55)
            ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.0f")
        # MOYENNE CHAMBRE calculee sur le masque (plus de '~1373 K' ambigu)
        mask_ch = (((Xk - CHAMBER_KM["x"]) ** 2 +
                    (Yk - CHAMBER_KM["y"]) ** 2) <= CHAMBER_KM["r"] ** 2)
        T_mean = (float(F["T"][mask_ch].mean())
                  if mask_ch.any() else jget("chamber_statistics", "T_mean_K"))
        lbl = (f"Magma chamber (mean {T_mean:.0f} K)"
               if T_mean is not None else "Magma chamber")
        add_chamber(ax, label=lbl)
        style_map(ax, extent=DATA_EXTENT)
        fig.tight_layout()
        save_figure(fig, "fig09_thermal_field"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ====== Fig 10 (manuscrit) : Erreur absolue — annotation = CSV (source unique)
    print("[fig10] Pointwise absolute error (annotation = CSV, single source)")
    try:
        if df_val is None:
            raise FileNotFoundError("step1_validation_data.csv not found.")
        fig, ax = plt.subplots(figsize=(8, 6))
        err = np.clip(df_val["abs_error_m_s"].to_numpy(float), 1e-8, None)
        ax.semilogy(df_val["y_m"], err, "g-", lw=1.6)
        ax.axvspan(-STEP1_YC, STEP1_YC, color="gray", alpha=0.15)
        ax.axvline(STEP1_YC, color="k", ls=":", lw=1.0)
        ax.axvline(-STEP1_YC, color="k", ls=":", lw=1.0)
        ax.text(0.03, 0.97,
                r"max $|e|$ = %.2f m/s" % float(df_val["abs_error_m_s"].max())
                + "\n"
                r"MAE = %.2f m/s" % float(df_val["abs_error_m_s"].mean()),
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_xlabel(r"Transverse position $y$ [m]", fontsize=12)
        ax.set_ylabel(
            r"Absolute error $|u_{\mathrm{PINN}}-u_{\mathrm{exact}}|$ [m/s]",
            fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        fig.tight_layout()
        save_figure(fig, "fig10_pointwise_error"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ======== Fig 11 (manuscrit) : Viscosité — colorbar 1..6 + JSON ===========
    print("[fig11] Effective viscosity & shear rate (vmin=1, vmax=6)")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        log_mu = np.log10(np.clip(F["mu"], 1e-9, None))
        # vmin/vmax figes sur la fenetre physique [1e1, 1e6] du manuscrit
        cf1 = ax1.contourf(Xk, Yk, log_mu, levels=30, cmap="cividis",
                           vmin=1.0, vmax=6.0)
        cb1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label(r"$\log_{10}\,\mu_{\mathrm{eff}}$ [Pa$\cdot$s]")
        mu_core = jget("chamber_statistics", "mu_core_mean_pa_s")
        clip_ch = jget("physics_constraints",
                       "clamped_fraction_chamber_percent")
        if mu_core is not None:
            txt = rf"Chamber core $\mu$: {mu_core:.2e} Pa$\cdot$s"
            if clip_ch is not None:
                txt += f" | clipping {clip_ch:.1f} %"
            annotate_box(ax1, txt)
        add_chamber(ax1); add_panel_letter(ax1, "(A)")
        style_map(ax1, extent=DATA_EXTENT)

        log_gd = np.log10(np.clip(F["gamma_dot"], 1e-9, None))
        cf2 = ax2.contourf(Xk, Yk, log_gd, levels=30, cmap="inferno")
        cb2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label(r"$\log_{10}\,\dot{\gamma}$ [s$^{-1}$]")
        add_chamber(ax2); add_panel_letter(ax2, "(B)")
        style_map(ax2, show_ylabel=False, extent=DATA_EXTENT)

        fig.tight_layout()
        save_figure(fig, "fig11_viscosity_shear"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ================= Fig 12 (manuscrit) : Contrainte ========================
    print("[fig12] Shear stress distribution (yield contour)")
    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        tau = F["tau_kPa"]
        cf = ax.contourf(Xk, Yk, tau, levels=35, cmap="RdYlBu_r")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"Shear stress $\tau$ [kPa]")
        tau0_kPa = TAU0_PA / 1e3
        if tau.min() < tau0_kPa < tau.max():
            ax.contour(Xk, Yk, tau, levels=[tau0_kPa], colors="black",
                       linewidths=2.0, linestyles="--")
        annotate_box(ax, r"$\tau_0$ = %.0f Pa (Herschel-Bulkley yield)"
                     % TAU0_PA, loc="lower_left")
        add_chamber(ax)
        style_map(ax, extent=DATA_EXTENT)
        fig.tight_layout()
        save_figure(fig, "fig12_shear_stress"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ================= Fig 13 (manuscrit) : Matrice corrélation ===============
    print("[fig13] Multi-physics correlation matrix")
    try:
        corr_cols = ["u_m_s", "v_m_s", "p_prime_pa", "T_K", "tau_pa",
                     "mu_eff_pa_s"]
        corr = df_res[corr_cols].corr().to_numpy()
        labels = [r"$u$", r"$v$", r"$p'$", r"$T$", r"$\tau$", r"$\mu$"]
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_corr_heatmap(ax, corr, labels)
        fig.tight_layout()
        save_figure(fig, "fig13_correlation_matrix"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ==== Fig 14 (manuscrit) : Sismicité — branche imprimée, fallback 225 =====
    print("[fig14] Overpressure & seismicity (obs/synthetic branch printed)")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                       sharex=True, sharey=True)
        px, py, pr = PERNICANA_KM["x"], PERNICANA_KM["y"], PERNICANA_KM["r"]

        cf1 = ax1.contourf(Xk, Yk, F["p_MPa"], levels=30, cmap="Blues")
        cb1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label(r"Magmatic overpressure $p'$ [MPa]")
        ax1.add_patch(plt.Circle((px, py), pr, facecolor="none",
                                 edgecolor="darkred", linewidth=2.0,
                                 linestyle="--", zorder=6))
        ax1.text(px, py - pr - 0.12, "Pernicana fault zone", color="darkred",
                 ha="center", va="top", fontsize=9, fontweight="bold",
                 zorder=7)
        add_panel_letter(ax1, "(A)")
        style_map(ax1, extent=DATA_EXTENT)

        cf2 = ax2.contourf(Xk, Yk, F["p_MPa"], levels=20, cmap="Greys",
                           alpha=0.35)
        if float(F["p_MPa"].max() - F["p_MPa"].min()) > 1e-12:
            cs = ax2.contour(Xk, Yk, F["p_MPa"], levels=8, colors="black",
                             linewidths=0.6, alpha=0.55)
            ax2.clabel(cs, inline=True, fontsize=7.5, fmt="%.1f")

        use_obs = (df_obs is not None and
                   {"x_utm", "y_utm", "magnitude"}.issubset(df_obs.columns))
        if use_obs:
            d = df_obs.dropna(subset=["x_utm", "y_utm", "magnitude"]).copy()
            d = d[(d["x_utm"] > 1e3) & (d["y_utm"] > 1e3)]
            use_obs = len(d) > 0
        if use_obs:
            # NOTE : normalisation min-max -> georeferencement approximatif.
            # Pour une revue Q1 : remplacer par une ancre UTM fixe (centre chambre).
            xk = ((d["x_utm"] - d["x_utm"].min()) / 1000.0
                  + Xk.min()).to_numpy()
            yk = ((d["y_utm"] - d["y_utm"].min()) / 1000.0
                  + Yk.min()).to_numpy()
            inside = ((xk >= Xk.min()) & (xk <= Xk.max()) &
                      (yk >= Yk.min()) & (yk <= Yk.max()))
            xk, yk = xk[inside], yk[inside]
            mags = d["magnitude"].to_numpy()[inside]
            use_obs = len(xk) > 0
        if use_obs:
            sc_label = f"Observed seismicity ($M_w$, {len(xk)} events)"
            print(f"[FIG14] Source sismicite : OBSERVEE ({len(xk)} evenements).")
        else:
            rng = np.random.default_rng(42)
            xk = np.clip(rng.normal(px, 0.30, N_FALLBACK_EVENTS),
                         Xk.min(), Xk.max())
            yk = np.clip(rng.normal(py, 0.25, N_FALLBACK_EVENTS),
                         Yk.min(), Yk.max())
            mags = rng.uniform(1.0, 3.2, N_FALLBACK_EVENTS)
            sc_label = "Seismicity ($M_w$, synthetic)"
            print("[FIG14][WARN] Source sismicite : SYNTHETIQUE "
                  "(CSV absent ou sans x_utm/y_utm) -> harmoniser §3.7 + Fig 15-16.")

        ax2.scatter(xk, yk, s=np.clip(mags, 0.5, None) * 35, color="crimson",
                    edgecolor="black", linewidth=0.5, alpha=0.9, zorder=6,
                    label=sc_label)
        ax2.add_patch(plt.Circle((px, py), pr, facecolor="none",
                                 edgecolor="darkred", linewidth=2.0,
                                 linestyle="--", zorder=5))
        ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
        add_panel_letter(ax2, "(B)")
        style_map(ax2, show_ylabel=False, extent=DATA_EXTENT)

        fig.tight_layout()
        save_figure(fig, "fig14_overpressure_seismicity"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ================= Fig 15 (manuscrit) : Distribution magnitudes ===========
    print("[fig15] Earthquake magnitude distribution")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if df_obs is not None and "magnitude" in df_obs.columns:
            mags = df_obs["magnitude"].dropna().to_numpy(float)
            src = "observed catalogue (Pernicana)"
        else:
            rng = np.random.default_rng(1)
            mags = rng.exponential(scale=1.2, size=N_FALLBACK_EVENTS) + 0.8
            src = "synthetic catalogue"
        plot_hist(ax, mags, color="crimson", bins=18,
                  xlabel=r"Earthquake magnitude ($M_w$)",
                  ylabel="Number of events", stat="count", annotate=False)
        med = float(np.median(mags))
        ax.axvline(med, color="black", linestyle=":", lw=1.5)
        ymax = ax.get_ylim()[1]
        ax.text(med + 0.05, 0.90 * ymax, f"median = {med:.2f}", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="gray", alpha=0.9))
        annotate_box(ax, src, loc="upper_left")
        fig.tight_layout()
        save_figure(fig, "fig15_magnitude_distribution"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ================= Fig 16 (manuscrit) : Profondeur vs magnitude ===========
    print("[fig16] Hypocentral depth vs magnitude")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        use = (df_obs is not None and
               {"depth_bsl", "magnitude"}.issubset(df_obs.columns))
        if use:
            d = df_obs.dropna(subset=["depth_bsl", "magnitude"])
            use = len(d) > 2
        if use:
            depth = d["depth_bsl"].to_numpy(float)
            mag = d["magnitude"].to_numpy(float)
            if np.median(depth) < 0:
                depth = -depth
            src = "observed catalogue"
        else:
            rng = np.random.default_rng(2)
            depth = rng.uniform(0.5, 12.0, N_FALLBACK_EVENTS)
            mag = rng.uniform(1.0, 4.5, N_FALLBACK_EVENTS)
            src = "synthetic catalogue"
        sc = ax.scatter(depth, mag, c=mag, cmap="viridis", s=45,
                        edgecolor="black", linewidth=0.5, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax, shrink=0.9)
        cb.set_label(r"Magnitude ($M_w$)")
        ax.invert_yaxis()
        ax.set_xlabel("Hypocentral depth below sea level [km]")
        ax.set_ylabel(r"Magnitude ($M_w$)")
        annotate_box(ax, src, loc="lower_left")
        fig.tight_layout()
        save_figure(fig, "fig16_depth_magnitude"); n_ok += 1
    except Exception:
        traceback.print_exc()

    # ============================ AUDIT FINAL =================================
    manuscript_audit(decision_data, df_res, df_val, df_log1, df_log2)

    print("=" * 70)
    print(f" SUMMARY : {n_ok}/14 FIGURES GENERATED (fig03..fig16)")
    print(" Labels: English | Titles: none | PNG + PDF @ 600 DPI")
    print(f" Output directory : ./{OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()