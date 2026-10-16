import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
H, G, TAU0, K, N_EXP, M_REG = 1.0, 1000.0, 50.0, 10.0, 0.7, 5.0e2
MU0, EA, T_CONST, T_REF = 100.0, 50e3, 1200.0, 1373.0
R_GAS = 8.314462618
ADAM_EPOCHS, ADAM_LR, CLIP = 6000, 5.0e-4, 1.0
LBFGS_PASSES, LBFGS_ITER = 3, 500
N_BASE, N_INT, WIN = 401, 150, 0.02
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

mu_ratio = float(np.exp((EA / R_GAS) * (1.0 / T_CONST - 1.0 / T_REF)))
tau0_eff, K_eff = mu_ratio * TAU0, mu_ratio * K
yc = tau0_eff / G

def u_exact_np(y):
    """Corrigé : u[mask_rigid] = u_plug (la ligne qui manquait en v3-v7)."""
    y = np.atleast_1d(np.abs(y)).astype(float)
    u = np.zeros_like(y)
    mask_fluid = y > yc
    mask_rigid = ~mask_fluid
    pref = (N_EXP / (N_EXP + 1.0)) * (G / K_eff) ** (1.0 / N_EXP)
    expo = (N_EXP + 1.0) / N_EXP
    u_plug = pref * (H - yc) ** expo
    u[mask_rigid] = u_plug                     
    u[mask_fluid] = pref * ((H - yc) ** expo
                            - (y[mask_fluid] - yc) ** expo)
    return u

def make_net():
    layers = [1, 50, 50, 50, 50, 1]
    mods, acts = [], []
    lin = nn.ModuleList()
    for i in range(len(layers) - 1):
        lin.append(nn.Linear(layers[i], layers[i + 1]).to(DTYPE))
    net = nn.Module(); net.linears = lin
    def forward(x):
        for l in lin[:-1]:
            x = torch.tanh(l(x))
        return lin[-1](x)
    net.forward = forward
    for l in lin:
        nn.init.xavier_normal_(l.weight.to(DTYPE)); nn.init.zeros_(l.bias)
    return net

net = make_net().to(DEVICE)

u_max = float(np.max(u_exact_np(np.linspace(-H, H, 2001))))
u_scale = 1.2 * u_max
print(f"[INFO] mu_ratio = {mu_ratio:.4f} | y_c = {yc:.6f} m | "
      f"u_max = {u_max:.2f} m/s | u_scale = {u_scale:.2f} m/s")

assert u_max > 0.5 * u_scale, "BUG DE RÉFÉRENCE : plug nul détecté !"
print("[GUARD] référence analytique OK (plug > 0).")

y_f = torch.cat([torch.linspace(-H, H, N_BASE, dtype=DTYPE, device=DEVICE),
                 torch.linspace(yc - WIN, yc + WIN, N_INT, dtype=DTYPE, device=DEVICE),
                 torch.linspace(-yc - WIN, -yc + WIN, N_INT, dtype=DTYPE, device=DEVICE)]
                ).reshape(-1, 1)
y_bc = torch.tensor([[-H], [H]], dtype=DTYPE, device=DEVICE)
point_w = torch.ones_like(y_f)

def net_u(y):
    s = y / H
    return (1.0 - s * s) * u_scale * net(s)

def mu_eff(u_y):
    gd = torch.abs(u_y) + 1e-12
    mu_hb = (TAU0 * (1.0 - torch.exp(-M_REG * gd)) / gd
             + K * gd ** (N_EXP - 1.0))
    return mu_hb * mu_ratio

def residual(y):
    y = y.clone().requires_grad_(True)
    u = net_u(y)
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    tau = mu_eff(u_y) * u_y
    return (tau + G * y) / (G * H)            

def loss_fn():
    res = residual(y_f)
    lp = torch.mean(point_w * res ** 2) / torch.mean(point_w)
    lb = torch.mean(net_u(y_bc) ** 2)          
    return lp + 10.0 * lb, res

opt = optim.Adam(net.parameters(), lr=ADAM_LR)
print(f"[INFO] Adam {ADAM_EPOCHS} it. | L-BFGS {LBFGS_PASSES} x {LBFGS_ITER} | "
      f"collocation {y_f.shape[0]} pts")
for epoch in range(ADAM_EPOCHS):
    opt.zero_grad()
    loss, res = loss_fn()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=CLIP)
    opt.step()
    if epoch >= 300 and epoch % 100 == 0:
        with torch.no_grad():
            t = torch.clamp((res.detach() ** 2) / torch.mean(res.detach() ** 2),
                            0.05, 20.0)
            point_w.clamp_(0.1, 50.0)
            point_w.mul_(0.5).add_(0.5 * t).clamp_(0.1, 50.0)
    if epoch % 1000 == 0 or epoch == ADAM_EPOCHS - 1:
        print(f"  Epoch {epoch:5d} | loss {loss.item():.3e}")

for p in range(LBFGS_PASSES):
    optL = optim.LBFGS(net.parameters(), lr=1.0, max_iter=LBFGS_ITER,
                       max_eval=LBFGS_ITER * 2, tolerance_grad=1e-12,
                       tolerance_change=1e-16, history_size=100,
                       line_search_fn="strong_wolfe")
    def closure():
        optL.zero_grad()
        l, _ = loss_fn()
        l.backward()
        return l
    optL.step(closure)
    print(f"  L-BFGS passe {p + 1}/{LBFGS_PASSES} | loss {loss_fn()[0].item():.3e}")

with torch.no_grad():
    y = torch.linspace(-H, H, 500, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    u_pred = net_u(y).cpu().numpy().flatten()
y_np = y.cpu().numpy().flatten()
u_ex = u_exact_np(y_np)
err = np.abs(u_pred - u_ex)
l2 = float(np.linalg.norm(err) / (np.linalg.norm(u_ex) + 1e-12))
r2 = float(1.0 - np.sum(err ** 2) / np.sum((u_ex - u_ex.mean()) ** 2))
rmse, mae = float(np.sqrt(np.mean(err ** 2))), float(np.mean(err))

print("\n[MÉTRIQUES]")
print(f"  L2 = {l2:.4e} | R^2 = {r2:.6f} | RMSE = {rmse:.3f} m/s | "
      f"MAE = {mae:.3f} m/s | max|e| = {err.max():.3f} m/s")

pd.DataFrame({"y_m": y_np, "u_exact_m_s": u_ex, "u_pred_m_s": u_pred,
              "abs_error_m_s": err}).to_csv(
    "step1_validation_data.csv", index=False, float_format="%.8e")
print("[OK] Exporté : step1_validation_data.csv")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(y_np, u_ex, "k-", lw=2.5, label="Analytical solution")
ax1.plot(y_np, u_pred, "r--", lw=2.0, label="PINN prediction")
ax1.axvspan(-yc, yc, color="gray", alpha=0.15, label=f"Rigid plug ($y_c$={yc:.3f} m)")
ax1.set_xlabel(r"Transverse position $y$ [m]")
ax1.set_ylabel(r"Axial velocity $u(y)$ [m/s]")
ax1.legend(fontsize=9); ax1.grid(True, ls="--", alpha=0.5)
ax2.semilogy(y_np, np.clip(err, 1e-8, None), "g-", lw=1.6)
ax2.axvspan(-yc, yc, color="gray", alpha=0.15)
ax2.set_xlabel(r"Transverse position $y$ [m]")
ax2.set_ylabel(r"Absolute error $|u_{PINN}-u_{exact}|$ [m/s]")
ax2.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(f"step1_validation_profile.{fmt}", dpi=300,
                bbox_inches="tight", format=fmt)
plt.close(fig)
print("[OK] Figure : step1_validation_profile.png / .pdf")
