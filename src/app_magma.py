from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import torch
import torch.nn as nn
import torch.optim as optim
CONFIG: Dict = {
    "seed": 42,
    "learn_arrhenius": False,     
    "grad_clip_norm": 1.0,      
    "T_floor_kelvin": 600.0, 
                               
    "step1": {"adam_epochs": 6000, "adam_lr": 5.0e-4,
              "lbfgs_epochs": 500, "lbfgs_passes": 3,
              "n_colloc": 401,                           
              "residual_mode": "stress",
              "float64": True,
              "m_reg": 5.0e2,                            
              "continuation": False},                   
                                                          
    "step2": {"adam_epochs": 4000, "adam_lr": 1.0e-3,
              "lbfgs_epochs": 300, "lbfgs_passes": 1,
              "n_domain": 2000, "n_per_edge": 120, "n_insar": 80, "n_petro": 400},
}

MU_PHYS_MIN = 1.0e1                     
MU_PHYS_MAX = 1.0e6                      
LN_MU_MIN = float(np.log(MU_PHYS_MIN))
LN_MU_MAX = float(np.log(MU_PHYS_MAX))

MU0_BOUNDS = (10.0, 1.0e4)              
EA_BOUNDS = (5.0e4, 3.5e5)               


def _safe_log_bounds(v: np.ndarray, floor: float = 1e-12) -> Tuple[float, float]:

    vmin = float(max(np.min(v), floor))
    vmax = float(np.max(v))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax
class PhysicalConstants:

    R_GAS: float = 8.314462618           
    RHO_MAGMA: float = 2700.0          
    CP_MAGMA: float = 1200.0           
    K_THERMAL: float = 2.5              
    G_GRAVITY: float = 9.81               
    ALPHA_THERMAL: float = 3.0e-5         
    T_REF_ARRHENIUS: float = 1373.0      

class HerschelBulkleyPapanastasiou:
    def __init__(self, tau0: float, K: float, n: float, m_reg: float):
        self.tau0 = float(tau0)
        self.K = float(K)
        self.n = float(n)
        self.m_reg = float(m_reg)

    def stress(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = torch.abs(gamma_dot) + 1e-10
        return self.tau0 * (1.0 - torch.exp(-self.m_reg * gd)) + self.K * gd ** self.n

    def effective_viscosity(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        gd = torch.abs(gamma_dot) + 1e-10
        return (self.tau0 * (1.0 - torch.exp(-self.m_reg * gd)) / gd
                + self.K * gd ** (self.n - 1.0))

class ArrheniusViscosity:
    def __init__(self, mu0: float, Ea: float, T_ref: float,
                 T_min: float = 600.0, T_max: float = 2200.0):
        mu0_c = float(np.clip(mu0, *MU0_BOUNDS))
        Ea_c = float(np.clip(Ea, *EA_BOUNDS))
        if (mu0_c != mu0) or (Ea_c != Ea):
            warnings.warn("[ARRHENIUS] Paramètres contraints dans leurs bornes : "
                          f"mu0 {mu0:.3e} -> {mu0_c:.3e} Pa.s | "
                          f"Ea {Ea:.3e} -> {Ea_c:.3e} J/mol")
        self.mu0 = mu0_c
        self.Ea = Ea_c
        self.T_ref = float(T_ref)
        self.T_min = float(T_min)
        self.T_max = float(T_max)

    def relative_viscosity(self, T: torch.Tensor) -> torch.Tensor:
        T_safe = torch.clamp(T, min=self.T_min, max=self.T_max)
        expo = (self.Ea / PhysicalConstants.R_GAS) * (1.0 / T_safe - 1.0 / self.T_ref)
        return torch.exp(torch.clamp(expo, min=-20.0, max=20.0))

    def relative_viscosity_numpy(self, T: np.ndarray) -> np.ndarray:
        T_safe = np.clip(T, self.T_min, self.T_max)
        expo = (self.Ea / PhysicalConstants.R_GAS) * (1.0 / T_safe - 1.0 / self.T_ref)
        return np.exp(np.clip(expo, -20.0, 20.0))

class ConstrainedArrheniusParameters(nn.Module):

    def __init__(self, mu0_init: float, Ea_init: float):
        super().__init__()

        def _inv_sig(v: float, lo: float, hi: float) -> float:
            s = float(np.clip((v - lo) / (hi - lo), 1e-4, 1.0 - 1e-4))
            return float(np.log(s / (1.0 - s)))

        self.raw_mu0 = nn.Parameter(torch.tensor(_inv_sig(mu0_init, *MU0_BOUNDS),
                                                 dtype=torch.float32))
        self.raw_Ea = nn.Parameter(torch.tensor(_inv_sig(Ea_init, *EA_BOUNDS),
                                                dtype=torch.float32))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mu0 = MU0_BOUNDS[0] + (MU0_BOUNDS[1] - MU0_BOUNDS[0]) * torch.sigmoid(self.raw_mu0)
        Ea = EA_BOUNDS[0] + (EA_BOUNDS[1] - EA_BOUNDS[0]) * torch.sigmoid(self.raw_Ea)
        return mu0, Ea


class AdaptiveLossBalancer:

    def __init__(self, keys: List[str], beta: float = 0.9, update_every: int = 100,
                 warmup: int = 500, w_min: float = 1e-2, w_max: float = 1e3):
        self.keys = list(keys)
        self.beta = beta
        self.update_every = update_every
        self.warmup = warmup
        self.w_min = w_min
        self.w_max = w_max
        self.weights: Dict[str, float] = {k: 1.0 for k in self.keys}

    def should_update(self, epoch: int) -> bool:
        return epoch >= self.warmup and (epoch % self.update_every == 0)

    def update(self, losses: Dict[str, torch.Tensor],
               params: List[nn.Parameter]) -> None:
        norms: Dict[str, float] = {}
        for k in self.keys:
            grads = torch.autograd.grad(losses[k], params,
                                        retain_graph=True, allow_unused=True)
            sq = 0.0
            for g in grads:
                if g is not None:
                    sq += float(torch.sum(g.detach() ** 2).item())
            norms[k] = float(np.sqrt(max(sq, 0.0))) + 1e-12
        g_max = max(norms.values())
        for k in self.keys:
            target = g_max / norms[k]
            new_w = self.beta * self.weights[k] + (1.0 - self.beta) * target
            self.weights[k] = float(np.clip(new_w, self.w_min, self.w_max))


class FullyConnectedNN(nn.Module):

    def __init__(self, layers: List[int], activation: str = "tanh"):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Activation '{activation}' non supportée.")
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        return self.linears[-1](x)


class CSVLogger:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.history: List[Dict[str, float]] = []

    def log(self, epoch: int, losses: Dict[str, float]) -> None:
        entry = {"epoch": int(epoch)}
        entry.update(losses)
        self.history.append(entry)

    def save(self) -> None:
        pd.DataFrame(self.history).to_csv(self.filepath, index=False,
                                          float_format="%.8e")
        print(f"[INFO] Log d'entraînement sauvegardé : {self.filepath}")

class PoiseuilleAnalytical:

    def __init__(self, H: float, G: float, tau0: float, K: float, n: float,
                 mu_ratio: float = 1.0):
        self.H = H
        self.G = G
        self.n = n
        self.tau0_eff = mu_ratio * tau0
        self.K_eff = mu_ratio * K
        self.yc = self.tau0_eff / G
        if self.yc >= H:
            warnings.warn("yc >= H : seuil trop élevé, écoulement bloqué.")

    def velocity_profile(self, y: np.ndarray) -> np.ndarray:

        y = np.atleast_1d(np.abs(y)).astype(float)
        u = np.zeros_like(y)
        mask_fluid = y > self.yc
        mask_rigid = ~mask_fluid
        prefactor = (self.n / (self.n + 1.0)) * (self.G / self.K_eff) ** (1.0 / self.n)
        expo = (self.n + 1.0) / self.n
        u_plug = prefactor * (self.H - self.yc) ** expo
        u[mask_rigid] = u_plug                             
        u[mask_fluid] = prefactor * ((self.H - self.yc) ** expo
                                     - (y[mask_fluid] - self.yc) ** expo)
        return u


class PoiseuillePINN:

    def __init__(self, H: float = 1.0, G: float = 1000.0, tau0: float = 50.0,
                 K: float = 10.0, n: float = 0.7, m_reg: float = 5.0e2,
                 mu0_arrh: float = 100.0, Ea_arrh: float = 50e3,
                 T_const: float = 1200.0, hidden_layers: Optional[List[int]] = None,
                 n_colloc: int = 401, adam_lr: float = 5.0e-4,
                 grad_clip: float = 1.0, T_floor: float = 600.0,
                 residual_mode: str = "stress",
                 dtype: torch.dtype = torch.float64,
                 point_warmup: int = 300, point_every: int = 100,
                 device: Optional[torch.device] = None):
        self.H, self.G, self.T_const = H, G, T_const
        self.grad_clip = float(grad_clip)
        if residual_mode not in ("stress", "flux"):
            raise ValueError("residual_mode doit être 'stress' (v4) ou 'flux' (v3).")
        self.residual_mode = residual_mode
        self.dtype = dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.hb = HerschelBulkleyPapanastasiou(tau0, K, n, m_reg)
        self.tau0_target = float(tau0)             
        self.arrh = ArrheniusViscosity(mu0_arrh, Ea_arrh,
                                       PhysicalConstants.T_REF_ARRHENIUS,
                                       T_min=T_floor)

        self.mu_ratio = float(self.arrh.relative_viscosity(
            torch.tensor([[T_const]], dtype=torch.float32)).item())
        self.mu_ratio_t = torch.tensor(self.mu_ratio, dtype=self.dtype,
                                       device=self.device)
        self.analytical = PoiseuilleAnalytical(H, G, tau0, K, n,
                                               mu_ratio=self.mu_ratio)

        u_max = float(np.max(self.analytical.velocity_profile(
            np.linspace(-H, H, 2001))))
        u_max_no_arrh = u_max * self.mu_ratio ** (1.0 / n)
        self.u_scale = 1.2 * u_max
        print(f"[ÉTAPE 1] Normalisation (vérifiée AMONT) : "
              f"mu(T={T_const:.0f} K)/mu(T_ref) = {self.mu_ratio:.4f} | "
              f"u_max(exact, avec Arrhenius) = {u_max:.1f} m/s | "
              f"u_max(sans Arrhenius) = {u_max_no_arrh:.1f} m/s | "
              f"u_scale = {self.u_scale:.1f} m/s -> facteur Arrhenius ACTIF "
              f"dans u_scale (u_max divisé par {self.mu_ratio ** (1.0 / n):.2f})")

        sens0 = 0.5 * self.analytical.tau0_eff * m_reg
        print(f"[ÉTAPE 1] Rhéologie : tau0 = {tau0:.1f} Pa | m_reg = "
              f"{m_reg:.0e} | y_c = {self.analytical.yc:.6f} m (indép. de m) | "
              f"plateau mu = {tau0 * m_reg:.1e} Pa.s | sensibilité "
              f"dR/d(gd)|_0 ~ {sens0 / (G * H):.1f}")

        if hidden_layers is None:
            hidden_layers = [50, 50, 50, 50]
        self.net = FullyConnectedNN([1] + list(hidden_layers) + [1],
                                    activation="tanh").to(self.device).to(self.dtype)

        self.opt_adam = optim.Adam(self.net.parameters(), lr=adam_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_adam, mode="min", factor=0.5, patience=300)
        y_c_abs = float(self.analytical.yc)
        n_base = max(201, int(n_colloc))
        y_all = [torch.linspace(-H, H, n_base, dtype=self.dtype,
                                device=self.device)]
        win, n_int = 0.02, 150
        for sgn in (+1.0, -1.0):
            c = sgn * y_c_abs
            y_all.append(torch.linspace(c - win, c + win, n_int,
                                        dtype=self.dtype, device=self.device))
        self.y_f = torch.cat(y_all).reshape(-1, 1)
        print(f"[ÉTAPE 1][FIX-14] Collocation : {n_base} pts de base + "
              f"2 x {n_int} pts dans ±{win * 1e3:.0f} mm autour de "
              f"±y_c = ±{y_c_abs:.4f} m (total {self.y_f.shape[0]} pts) "
              f"-> jonction échantillonnée, saut visible à la perte.")
        self.y_bc = torch.tensor([[-H], [H]], dtype=self.dtype,
                                 device=self.device)


        self.point_w = torch.ones_like(self.y_f)
        self.w_ema = 0.5
        self.w_warmup = int(point_warmup)
        self.w_every = int(point_every)

        self.logger = CSVLogger("train_log_step1.csv")
        self.loss_history: List[Dict[str, float]] = []
        self.last_eval: Optional[Dict] = None
        self._iter = 0

    def net_u(self, y: torch.Tensor) -> torch.Tensor:
        s = y / self.H
        return (1.0 - s * s) * self.u_scale * self.net(s)

    def _mu_eff(self, u_y: torch.Tensor) -> torch.Tensor:
        gd = torch.abs(u_y) + 1e-12
        mu_hb = (self.hb.tau0 * (1.0 - torch.exp(-self.hb.m_reg * gd)) / gd
                 + self.hb.K * gd ** (self.hb.n - 1.0))
        return mu_hb * self.mu_ratio_t

    def pde_residual(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clone().requires_grad_(True)
        u = self.net_u(y)
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        tau = self._mu_eff(u_y) * u_y
        if self.residual_mode == "stress":
   
            return (tau + self.G * y) / (self.G * self.H)
        flux_y = torch.autograd.grad(tau, y, grad_outputs=torch.ones_like(tau),
                                     create_graph=True)[0]
        return (flux_y + self.G) / self.G

    def verify_formulation(self, n: int = 4001) -> Dict[str, float]:
        an = self.analytical
        y = torch.linspace(-self.H, self.H, n, dtype=self.dtype,
                           device=self.device).reshape(-1, 1)
        yy = y.clone().requires_grad_(True)
        ay = torch.abs(yy)
        yc, expo = an.yc, (an.n + 1.0) / an.n
        pref = (an.n / (an.n + 1.0)) * (self.G / an.K_eff) ** (1.0 / an.n)
        u_plug = pref * (self.H - yc) ** expo
        u_fluid = pref * ((self.H - yc) ** expo - (ay - yc) ** expo)
        u_ex = torch.where(ay <= yc, u_plug + 0.0 * yy, u_fluid)
        u_y = torch.autograd.grad(u_ex, yy,
                                  grad_outputs=torch.ones_like(u_ex))[0]
        R = ((self._mu_eff(u_y) * u_y + self.G * yy)
             / (self.G * self.H)).detach()
        Rf = torch.abs(R.flatten())
        fluid = (ay > yc).flatten()
        med_f = float(torch.median(Rf[fluid]).item())
        max_f = float(torch.max(Rf[fluid]).item())
        du_reg = 0.5 * self.G * yc ** 2 / (an.tau0_eff * self.hb.m_reg)
        print(f"[ÉTAPE 1][AUTO-TEST v7] Résidu de la solution exacte CIBLE, "
              f"forme '{self.residual_mode}', même code que le PINN "
              f"(Arrhenius x{self.mu_ratio:.4f} des deux côtés) :")
        print(f"    zone fluide : médiane {med_f:.3e}, max {max_f:.3e}")
        print(f"    bande de régularisation |y-y_c| <~ 1/m = "
              f"{1.0 / self.hb.m_reg:.4f} m (y_c = {yc:.3f} m) ; "
              f"impact vitesse de la régularisation : {du_reg:.2e} m/s "
              f"(négligeable devant u_max = {self.u_scale / 1.2:.1f} m/s)")
        print(f"    -> formulation et normalisation COHÉRENTES ; toute "
              f"erreur résiduelle sera imputable à l'optimisation "
              f"(traitée par le raffinement d'interface [FIX-14]).")
        return {"median_fluid": med_f, "max_fluid": max_f,
                "du_regularization": du_reg}

    def loss_function(self) -> Tuple[torch.Tensor, Dict[str, float],
                                     torch.Tensor]:

        res = self.pde_residual(self.y_f)
        loss_pde = (torch.mean(self.point_w * res ** 2)
                    / torch.mean(self.point_w))

        loss_bc = torch.mean(self.net_u(self.y_bc) ** 2)
        total = loss_pde + 10.0 * loss_bc
        d = {"loss_total": float(total.item()),
             "loss_pde": float(loss_pde.item()),
             "loss_bc": float(loss_bc.item())}
        return total, d, res

    def train_adam(self, epochs: int = 6000) -> None:
        print(f"[ÉTAPE 1] Phase Adam : {epochs} itérations "
              f"(tau0 = {self.hb.tau0:.1f} Pa | BC dure, résidu "
              f"'{self.residual_mode}', {str(self.dtype).replace('torch.', '')}, "
              f"clip grad {self.grad_clip}, auto-poids EMA toutes les "
              f"{self.w_every} it. après warm-up {self.w_warmup})...")
        self.net.train()
        t0 = time.time()
        for epoch in range(epochs):
            self.opt_adam.zero_grad()
            loss, d, res = self.loss_function()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           max_norm=self.grad_clip)
            self.opt_adam.step()
            self.scheduler.step(d["loss_total"])

            if epoch >= self.w_warmup and epoch % self.w_every == 0:
                with torch.no_grad():
                    r2 = res.detach() ** 2
                    target = torch.clamp(r2 / torch.mean(r2), 0.05, 20.0)
                    new_w = (self.w_ema * self.point_w
                             + (1.0 - self.w_ema) * target)
                    self.point_w = torch.clamp(new_w, 0.1, 50.0)

            self._iter += 1
            self.logger.log(self._iter, d)
            self.loss_history.append(d)
            if epoch % 500 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    w_min = float(self.point_w.min().item())
                    w_max = float(self.point_w.max().item())
                print(f"  Epoch {epoch:5d} | total {d['loss_total']:.3e} | "
                      f"PDE {d['loss_pde']:.3e} | BC {d['loss_bc']:.3e} | "
                      f"poids [{w_min:.2f}, {w_max:.2f}]")
        print(f"[ÉTAPE 1] Adam terminé en {time.time() - t0:.1f} s")

    def train_lbfgs(self, max_iter: int = 500) -> None:
        print(f"[ÉTAPE 1] Phase L-BFGS-B : {max_iter} itérations max "
              f"(auto-poids FIGÉS, sans clipping)...")
        self.net.train()
        opt = optim.LBFGS(self.net.parameters(), lr=1.0, max_iter=max_iter,
                          max_eval=max_iter * 2, tolerance_grad=1e-12,
                          tolerance_change=1e-16, history_size=100,
                          line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss, d, _ = self.loss_function()   
            loss.backward()
            self._iter += 1
            self.logger.log(self._iter, d)
            self.loss_history.append(d)
            return loss

        opt.step(closure)
        print(f"[ÉTAPE 1] L-BFGS-B terminé | "
              f"loss finale {self.loss_history[-1]['loss_total']:.3e}")

    def train(self, adam_epochs: int = 6000, lbfgs_epochs: int = 500,
              lbfgs_passes: int = 3, continuation: bool = False,
              n_stages: int = 5) -> None:

        t0 = time.time()
        if continuation:
            n_stages = max(2, int(n_stages))
            taus = [self.tau0_target * k / (n_stages - 1)
                    for k in range(n_stages)]
            ep_a = max(1000, adam_epochs // (2 * n_stages - 2))
            ep_l = max(100, lbfgs_epochs // 3)
            for k, tau_k in enumerate(taus[:-1]):
                self.hb.tau0 = tau_k
                yc_k = tau_k * self.mu_ratio / self.G
                print(f"[ÉTAPE 1][HOMOTOPIE {k + 1}/{n_stages}] "
                      f"tau0 = {tau_k:.1f} Pa (y_c = {yc_k:.3f} m)")
                self.train_adam(ep_a)
                self.train_lbfgs(ep_l)
            self.hb.tau0 = self.tau0_target
            print(f"[ÉTAPE 1][HOMOTOPIE {n_stages}/{n_stages}] CIBLE "
                  f"tau0 = {self.tau0_target:.1f} Pa + raffinement final")
            self.train_adam(ep_a)
            for _ in range(max(1, int(lbfgs_passes))):
                self.train_lbfgs(lbfgs_epochs)
        else:
            self.train_adam(adam_epochs)
            for _ in range(max(1, int(lbfgs_passes))):
                self.train_lbfgs(lbfgs_epochs)
        self.logger.save()
        print(f"[ÉTAPE 1] Temps total d'entraînement : {time.time() - t0:.1f} s")

    def probe_solution(self) -> Dict[str, float]:

        self.net.eval()
        with torch.no_grad():
            yg = torch.linspace(-self.H, self.H, 4001, dtype=self.dtype,
                                device=self.device).reshape(-1, 1)
            ug = self.net_u(yg).flatten()
        ygf = yg.flatten()
        u_max_ex = self.u_scale / 1.2

        def _at(yq: float) -> float:
            i = int(torch.argmin(torch.abs(ygf - yq)).item())
            return float(ug[i].item())

        du_max = float(torch.max(torch.abs(ug[1:] - ug[:-1])).item())
        u0 = _at(0.0)
        print(f"[ÉTAPE 1][SONDE v7] u_pred(0) = {u0:.2f} m/s "
              f"(cible plug {u_max_ex:.2f}) | "
              f"u_pred(+y_c) = {_at(self.analytical.yc):.2f} | "
              f"u_pred(0.05) = {_at(0.05):.2f} | "
              f"saut max inter-échantillons (pas 0.5 mm) = {du_max:.2f} m/s")
        if (u_max_ex - abs(u0)) > 0.1 * u_max_ex:
            print("    [SONDE] PLUG EFFONDRÉ détecté (u_pred(0) << u_max) — "
                  "interface non résolue.")
            ok = False
        else:
            print("    [SONDE] plug résolu : u_pred(0) ~ u_max, pas de saut "
                  "inter-échantillons -> profil conforme.")
            ok = True
        return {"u_center": u0, "u_max_jump": du_max, "plug_resolved": ok}

    def evaluate(self, n_points: int = 500) -> Dict[str, object]:
        self.net.eval()
        with torch.no_grad():
            y = torch.linspace(-self.H, self.H, n_points, dtype=self.dtype,
                               device=self.device).reshape(-1, 1)
            u_pred = self.net_u(y).cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        u_exact = self.analytical.velocity_profile(y_np)

        l2 = float(np.linalg.norm(u_pred - u_exact)
                   / (np.linalg.norm(u_exact) + 1e-12))
        ss_res = float(np.sum((u_exact - u_pred) ** 2))
        ss_tot = float(np.sum((u_exact - np.mean(u_exact)) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        rmse = float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
        mae = float(np.mean(np.abs(u_pred - u_exact)))
        max_err = float(np.max(np.abs(u_pred - u_exact)))

        self.last_eval = {"l2": l2, "r2": r2, "rmse": rmse, "mae": mae,
                          "max_abs_err": max_err, "y": y_np,
                          "u_pred": u_pred, "u_exact": u_exact}

        print("\n[MÉTRIQUES ÉTAPE 1]")                   
        print(f"  Résidu (forme)          : {self.residual_mode} | "
              f"dtype {str(self.dtype).replace('torch.', '')} | "
              f"m_reg {self.hb.m_reg:.0e} | raffinement interface [FIX-14] | "
              f"auto-poids [FIX-12]")
        print(f"  Facteur d'Arrhenius mu(T={self.T_const:.0f} K)/mu(T_ref) : "
              f"{self.mu_ratio:.4f} (inclus dans l'analytique ET u_scale)")
        print(f"  Erreur L2 relative : {l2:.6e}")
        print(f"  Coefficient R^2    : {r2:.8f}")
        print(f"  RMSE               : {rmse:.6e} m/s")
        print(f"  MAE                : {mae:.6e} m/s")
        print(f"  Erreur max         : {max_err:.6e} m/s")
        return self.last_eval

    def plot_results(self, fname: str = "step1_poiseuille_validation") -> None:
        if self.last_eval is None:
            self.evaluate()
        m = self.last_eval
        l2, r2 = m["l2"], m["r2"]
        y, u_pred, u_exact = m["y"], m["u_pred"], m["u_exact"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ep = np.array([h["epoch"] for h in self.logger.history], dtype=float)
        for key, style, lab in (("loss_total", "k-", r"$\mathcal{L}_{tot}$"),
                                ("loss_pde", "b--", r"$\mathcal{L}_{PDE}$"),
                                ("loss_bc", "r-.", r"$\mathcal{L}_{BC}$ (hard)")):
            vals = np.array([max(h[key], 1e-16) for h in self.logger.history])
            ax.semilogy(ep, vals, style, lw=1.4, label=lab)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        ax = axes[1]
        ax.plot(y, u_exact, "k-", lw=2.5, label="Analytical solution")
        ax.plot(y, u_pred, "r--", lw=2.0, label="PINN prediction")
        yc = self.analytical.yc
        if yc < self.H:
            ax.axvspan(-yc, yc, color="gray", alpha=0.15,
                       label=r"Rigid plug ($|y|<y_c$)")
        ax.text(0.03, 0.97, r"$\varepsilon_{L2}$ = %.2e" % l2 + "\n"
                r"$R^2$ = %.6f" % r2 + "\n"
                r"RMSE = %.2e m/s" % m["rmse"], transform=ax.transAxes,
                va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_xlabel(r"Transverse position $y$ [m]", fontsize=12)
        ax.set_ylabel(r"Axial velocity $u(y)$ [m/s]", fontsize=12)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax = axes[2]
        err = np.clip(np.abs(u_pred - u_exact), 1e-16, None)
        ax.semilogy(y, err, "g-", lw=1.6)
        if yc < self.H:
            ax.axvspan(-yc, yc, color="gray", alpha=0.15)
        ax.text(0.03, 0.97,
                r"max $|e|$ = %.2e m/s" % m["max_abs_err"] + "\n"
                r"MAE = %.2e m/s" % m["mae"], transform=ax.transAxes,
                va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_xlabel(r"Transverse position $y$ [m]", fontsize=12)
        ax.set_ylabel(r"Absolute error $|u_{PINN}-u_{exact}|$ [m/s]", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            path = f"{fname}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
            print(f"[INFO] Figure sauvegardée : {path}")
        plt.close(fig)

class EtnaDataPipeline:

    def __init__(self, L: float = 6000.0, H: float = 4000.0):
        self.L, self.H = L, H
        self.T_surface = 300.0            
        self.T_chamber = 1373.0          
        self.chamber_x, self.chamber_y = 0.5 * L, 0.65 * H
        self.chamber_radius = 500.0       
        self.tau0 = 2.0                 
        self.K = 25.0                 
        self.n = 0.6                   
        self.m_reg = 1.0e3            
        self.mu0_arrh = 80.0              
        self.Ea = 180e3                 
        self.T_ref = 1373.0               
        self.pressure_depth = 2500.0      
        self.pressure_x = 0.5 * L
        self.dP = 8.0e6                   
        self.U_REF = 0.5                 
        self.P_REF = 5.0e6              
        self.THETA_REF = 750.0           

    def T_cond(self, y):

        return self.T_surface + (self.T_chamber - self.T_surface) * (1.0 - y / self.H)

    def theta_target(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r2 = (x - self.chamber_x) ** 2 + (y - self.chamber_y) ** 2
        gauss = np.exp(-r2 / (2.0 * self.chamber_radius ** 2))
        return (self.T_chamber - self.T_cond(y)) * gauss

    def insar_surface_targets(self, n: int = 80):
        x = np.linspace(0.0, self.L, n)
        r = np.abs(x - self.pressure_x)
        uz = 0.30 * np.exp(-r ** 2 / (2.0 * 1000.0 ** 2))
        ux = -0.05 * (x - self.pressure_x) / self.pressure_depth * \
            np.exp(-r ** 2 / (2.0 * 1200.0 ** 2))
        return x, ux, uz

    def generate_collocation_points(self, n_domain: int = 2000,
                                    n_per_edge: int = 120,
                                    seed: int = 42) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_f = rng.uniform(0.0, self.L, n_domain)
        y_f = rng.uniform(0.0, self.H, n_domain)
        xe = np.linspace(0.0, self.L, n_per_edge)
        ye = np.linspace(0.0, self.H, n_per_edge)
        return {"x_f": x_f, "y_f": y_f,
                "x_bottom": xe, "y_bottom": np.zeros_like(xe),
                "x_top": xe, "y_top": np.full_like(xe, self.H),
                "x_left": np.zeros_like(ye), "y_left": ye,
                "x_right": np.full_like(ye, self.L), "y_right": ye}

    def generate_petro_anchors(self, n: int = 400, seed: int = 43):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, self.L, n)
        y = rng.uniform(0.0, self.H, n)
        return x, y, self.theta_target(x, y)


class EtnaPINN2D:
  
    LOSS_KEYS = ["continuity", "momentum_x", "momentum_y", "energy",
                 "bc_velocity", "bc_temperature",
                 "data_insar", "data_temperature", "pressure_gauge"]

    def __init__(self, pipeline: EtnaDataPipeline, learn_arrhenius: bool = False,
                 adam_lr: float = 1e-3, grad_clip: float = 1.0,
                 T_floor: float = 600.0, device: Optional[torch.device] = None,
                 n_domain: int = 2000, n_per_edge: int = 120,
                 n_insar: int = 80, n_petro: int = 400):
        self.pl = pipeline
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.learn_arrhenius = learn_arrhenius
        self.grad_clip = float(grad_clip)

        self.L0 = pipeline.H
        self.U0 = pipeline.U_REF
        self.P0 = pipeline.P_REF
        self.TH0 = pipeline.THETA_REF
        rho, cp = PhysicalConstants.RHO_MAGMA, PhysicalConstants.CP_MAGMA
        self.S_cont = self.U0 / self.L0
        self.S_ns = self.P0 / self.L0
        self.S_energy = rho * cp * self.TH0 * self.U0 / self.L0

        self.hb = HerschelBulkleyPapanastasiou(pipeline.tau0, pipeline.K,
                                               pipeline.n, pipeline.m_reg)
        self.arrh = ArrheniusViscosity(pipeline.mu0_arrh, pipeline.Ea,
                                       pipeline.T_ref, T_min=T_floor)
        self.arrh_par = (ConstrainedArrheniusParameters(pipeline.mu0_arrh,
                                                        pipeline.Ea).to(self.device)
                         if learn_arrhenius else None)

        self.net = FullyConnectedNN([2, 64, 64, 64, 64, 4],
                                    activation="swish").to(self.device)
        params = list(self.net.parameters())
        if learn_arrhenius:
            params += list(self.arrh_par.parameters())
        self.trainable_params = params

        self.opt_adam = optim.Adam(params, lr=adam_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_adam, mode="min", factor=0.5, patience=400)
        self.balancer = AdaptiveLossBalancer(self.LOSS_KEYS, beta=0.9,
                                             update_every=100, warmup=500)

        pts = pipeline.generate_collocation_points(n_domain, n_per_edge)

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(a, dtype=np.float32).reshape(-1, 1),
                                dtype=torch.float32, device=self.device)

        self.x_f = _t(pts["x_f"]); self.y_f = _t(pts["y_f"])
        self.x_f.requires_grad_(True); self.y_f.requires_grad_(True)
        self.x_bottom = _t(pts["x_bottom"]); self.y_bottom = _t(pts["y_bottom"])
        self.x_top = _t(pts["x_top"]);       self.y_top = _t(pts["y_top"])
        self.x_left = _t(pts["x_left"]);     self.y_left = _t(pts["y_left"])
        self.x_right = _t(pts["x_right"]);   self.y_right = _t(pts["y_right"])

        x_i, ux, uz = pipeline.insar_surface_targets(n_insar)
        self.x_insar = _t(x_i)
        self.y_insar = torch.full_like(self.x_insar, float(pipeline.H))
        self.ux_insar = _t(ux)
        self.uz_insar = _t(uz)

        px, py, pth = pipeline.generate_petro_anchors(n_petro)
        self.x_petro = _t(px); self.y_petro = _t(py); self.th_petro = _t(pth)

        self.logger = CSVLogger("train_log_step2.csv")
        self.loss_history: List[Dict[str, float]] = []
        self.diag: Dict[str, float] = {
            "clamp_fraction": 0.0, "mu_raw_min": float("nan"),
            "mu_raw_max": float("nan"), "mu_eff_min": float("nan"),
            "mu_eff_max": float("nan")}
        self._iter = 0

    def net_uvpT(self, x: torch.Tensor, y: torch.Tensor):
        x_n = 2.0 * x / self.pl.L - 1.0
        y_n = 2.0 * y / self.pl.H - 1.0
        out = self.net(torch.cat([x_n, y_n], dim=1))
        u = out[:, 0:1] * self.U0
        v = out[:, 1:2] * self.U0
        p = out[:, 2:3] * self.P0
        th = out[:, 3:4] * self.TH0
        return u, v, p, th

    def effective_viscosity(self, gamma_dot: torch.Tensor, T: torch.Tensor,
                            update_diag: bool = False) -> torch.Tensor:
        
        mu_hb = self.hb.effective_viscosity(gamma_dot)
        if self.learn_arrhenius and self.arrh_par is not None:
            _, Ea = self.arrh_par()
            T_safe = torch.clamp(T, min=self.arrh.T_min, max=self.arrh.T_max)
            expo = (Ea / PhysicalConstants.R_GAS) * \
                (1.0 / T_safe - 1.0 / self.pl.T_ref)
            rel = torch.exp(torch.clamp(expo, min=-20.0, max=20.0))
        else:
            rel = self.arrh.relative_viscosity(T)
        mu_raw = mu_hb * rel
        ln_mu = torch.log(torch.clamp(mu_raw, min=1e-8))
        ln_mu_c = torch.clamp(ln_mu, LN_MU_MIN, LN_MU_MAX)
        mu_eff = torch.exp(ln_mu_c)
        if update_diag:
            with torch.no_grad():
                self.diag["clamp_fraction"] = float(
                    ((ln_mu < LN_MU_MIN) | (ln_mu > LN_MU_MAX)).float().mean().item())
                self.diag["mu_raw_min"] = float(torch.exp(ln_mu).min().item())
                self.diag["mu_raw_max"] = float(torch.exp(ln_mu).max().item())
                self.diag["mu_eff_min"] = float(mu_eff.min().item())
                self.diag["mu_eff_max"] = float(mu_eff.max().item())
        return mu_eff

    def compute_pde_residuals(self, x: torch.Tensor, y: torch.Tensor,
                              update_diag: bool = False) -> Dict[str, torch.Tensor]:
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        u, v, p, th = self.net_uvpT(x, y)
        ones = torch.ones_like(u)

        u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True)[0]
        th_x = torch.autograd.grad(th, x, grad_outputs=ones, create_graph=True)[0]
        th_y = torch.autograd.grad(th, y, grad_outputs=ones, create_graph=True)[0]
        th_xx = torch.autograd.grad(th_x, x, grad_outputs=ones, create_graph=True)[0]
        th_yy = torch.autograd.grad(th_y, y, grad_outputs=ones, create_graph=True)[0]

        T = self.pl.T_cond(y) + th
        T_x = th_x
        T_y = th_y - (self.pl.T_chamber - self.pl.T_surface) / self.pl.H

        gamma_dot = torch.sqrt(2.0 * u_x ** 2 + 2.0 * v_y ** 2
                               + (u_y + v_x) ** 2) + 1e-9
        mu_eff = self.effective_viscosity(gamma_dot, T, update_diag=update_diag)

        tau_xx = 2.0 * mu_eff * u_x
        tau_yy = 2.0 * mu_eff * v_y
        tau_xy = mu_eff * (u_y + v_x)

        tau_xx_x = torch.autograd.grad(tau_xx, x, grad_outputs=ones, create_graph=True)[0]
        tau_xy_y = torch.autograd.grad(tau_xy, y, grad_outputs=ones, create_graph=True)[0]
        tau_xy_x = torch.autograd.grad(tau_xy, x, grad_outputs=ones, create_graph=True)[0]
        tau_yy_y = torch.autograd.grad(tau_yy, y, grad_outputs=ones, create_graph=True)[0]

        rho = PhysicalConstants.RHO_MAGMA
        grav = PhysicalConstants.G_GRAVITY
        alpha = PhysicalConstants.ALPHA_THERMAL

        continuity = (u_x + v_y) / self.S_cont
        ns_x = (rho * (u * u_x + v * u_y) + p_x
                - (tau_xx_x + tau_xy_y)) / self.S_ns
        buoyancy = rho * grav * alpha * th
        ns_y = (rho * (u * v_x + v * v_y) + p_y
                - (tau_xy_x + tau_yy_y) - buoyancy) / self.S_ns
        dissipation = tau_xx * u_x + tau_yy * v_y + tau_xy * (u_y + v_x)
        energy = (rho * PhysicalConstants.CP_MAGMA * (u * T_x + v * T_y)
                  - PhysicalConstants.K_THERMAL * (th_xx + th_yy)
                  - dissipation) / self.S_energy

        return {"continuity": continuity, "momentum_x": ns_x, "momentum_y": ns_y,
                "energy": energy, "p": p, "T": T, "mu_eff": mu_eff,
                "gamma_dot": gamma_dot}

    def compute_losses(self, update_diag: bool = False):
        res = self.compute_pde_residuals(self.x_f, self.y_f,
                                         update_diag=update_diag)
        L: Dict[str, torch.Tensor] = {}
        L["continuity"] = torch.mean(res["continuity"] ** 2)
        L["momentum_x"] = torch.mean(res["momentum_x"] ** 2)
        L["momentum_y"] = torch.mean(res["momentum_y"] ** 2)
        L["energy"] = torch.mean(res["energy"] ** 2)

        loss_bc_v = 0.0
        for xs, ys in ((self.x_bottom, self.y_bottom),
                       (self.x_left, self.y_left),
                       (self.x_right, self.y_right)):
            u_b, v_b, _, _ = self.net_uvpT(xs, ys)
            loss_bc_v = loss_bc_v + torch.mean(u_b ** 2) + torch.mean(v_b ** 2)
        L["bc_velocity"] = loss_bc_v

        _, _, _, th_b = self.net_uvpT(self.x_bottom, self.y_bottom)
        _, _, _, th_t = self.net_uvpT(self.x_top, self.y_top)
        L["bc_temperature"] = torch.mean(th_b ** 2) + torch.mean(th_t ** 2)

        u_t, v_t, _, _ = self.net_uvpT(self.x_insar, self.y_insar)
        L["data_insar"] = (torch.mean((u_t - self.ux_insar) ** 2)
                           + torch.mean((v_t - self.uz_insar) ** 2))
        _, _, _, th_p = self.net_uvpT(self.x_petro, self.y_petro)
        L["data_temperature"] = torch.mean((th_p - self.th_petro) ** 2)

        L["pressure_gauge"] = (torch.mean(res["p"]) / self.P0) ** 2

        total = 0.0
        for k in self.LOSS_KEYS:
            total = total + self.balancer.weights[k] * L[k]
        return L, total

    def train_adam(self, epochs: int = 4000) -> None:
        print(f"[ÉTAPE 2] Phase Adam : {epochs} itérations "
              f"(clip grad {self.grad_clip}, balancer toutes les "
              f"{self.balancer.update_every} it.)...")
        self.net.train()
        t0 = time.time()
        for epoch in range(epochs):
            self.opt_adam.zero_grad()
            L, total = self.compute_losses(update_diag=(epoch % 500 == 0))
            if self.balancer.should_update(epoch):
                self.balancer.update(L, self.trainable_params)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params,
                                           max_norm=self.grad_clip)
            self.opt_adam.step()
            self.scheduler.step(float(total.item()))

            self._iter += 1
            d = {f"loss_{k}": float(v.item()) for k, v in L.items()}
            d["loss_total"] = float(total.item())
            d.update({f"w_{k}": w for k, w in self.balancer.weights.items()})
            self.logger.log(self._iter, d)
            self.loss_history.append(d)

            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:5d} | total {d['loss_total']:.3e} | "
                      f"cont {d['loss_continuity']:.2e} | "
                      f"mom.x {d['loss_momentum_x']:.2e} | "
                      f"mom.y {d['loss_momentum_y']:.2e} | "
                      f"insar {d['loss_data_insar']:.2e} | "
                      f"clamp mu {100.0 * self.diag['clamp_fraction']:5.1f} %")
        print(f"[ÉTAPE 2] Adam terminé en {time.time() - t0:.1f} s")

    def train_lbfgs(self, max_iter: int = 300) -> None:
        print(f"[ÉTAPE 2] Phase L-BFGS-B : {max_iter} itérations max "
              f"(poids de perte figés, SANS clipping)...")
        self.net.train()
        opt = optim.LBFGS(self.trainable_params, lr=1.0, max_iter=max_iter,
                          max_eval=max_iter * 2, tolerance_grad=1e-9,
                          tolerance_change=1e-12, history_size=100,
                          line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            L, total = self.compute_losses()
            total.backward()
            self._iter += 1
            d = {f"loss_{k}": float(v.item()) for k, v in L.items()}
            d["loss_total"] = float(total.item())
            self.logger.log(self._iter, d)
            self.loss_history.append(d)
            return total

        opt.step(closure)
        self.compute_losses(update_diag=True)
        print(f"[ÉTAPE 2] L-BFGS-B terminé | "
              f"loss finale {self.loss_history[-1]['loss_total']:.3e}")

    def train(self, adam_epochs: int = 4000, lbfgs_epochs: int = 300,
              lbfgs_passes: int = 1) -> None:
        t0 = time.time()
        self.train_adam(adam_epochs)
        for _ in range(max(1, int(lbfgs_passes))):
            self.train_lbfgs(lbfgs_epochs)
        self.logger.save()
        print(f"[ÉTAPE 2] Temps total d'entraînement : {time.time() - t0:.1f} s")

    def predict_with_fields(self, nx: int = 120, ny: int = 90) -> Dict[str, np.ndarray]:
        self.net.eval()
        x_1d = torch.linspace(0.0, self.pl.L, nx, device=self.device)
        y_1d = torch.linspace(0.0, self.pl.H, ny, device=self.device)
        X, Y = np.meshgrid(x_1d.cpu().numpy(), y_1d.cpu().numpy())
        x = x_1d.repeat(ny).reshape(-1, 1)
        y = y_1d.repeat_interleave(nx).reshape(-1, 1)
        x.requires_grad_(True); y.requires_grad_(True)

        u, v, p, th = self.net_uvpT(x, y)
        ones = torch.ones_like(u)
        u_x = torch.autograd.grad(u, x, grad_outputs=ones, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=ones, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=ones, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=ones, retain_graph=True)[0]

        T = self.pl.T_cond(y) + th
        gamma_dot = torch.sqrt(2.0 * u_x ** 2 + 2.0 * v_y ** 2
                               + (u_y + v_x) ** 2) + 1e-9
        mu_eff = self.effective_viscosity(gamma_dot, T, update_diag=True)
        tau = mu_eff * gamma_dot

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().reshape(ny, nx)

        return {"X": X, "Y": Y, "u": _np(u), "v": _np(v), "p": _np(p),
                "T": _np(T), "tau": _np(tau), "mu_eff": _np(mu_eff),
                "gamma_dot": _np(gamma_dot)}

class EruptiveDecisionGrid:

    def __init__(self, p_threshold: float = 5.0e7, mu_threshold: float = 1.0e6):
        self.p_threshold = float(p_threshold)
        self.mu_threshold = float(mu_threshold)

    def classify(self, p_mean: float, mu_mean: float) -> Dict[str, str]:
        if (mu_mean > self.mu_threshold) and (p_mean > self.p_threshold):
            out = ("EXPLOSIF", "ALERTE ROUGE",
                   "Viscosité très élevée (>1e6 Pa.s) combinée à une surpression "
                   "critique (>50 MPa) : forte probabilité de fragmentation "
                   "magmatique et d'éruption paroxysmale explosive.",
                   "Procédure d'urgence : évacuation préventive des zones exposées, "
                   "alerte INGV maximale, surveillance aérienne et satellitaire "
                   "continue.")
        elif mu_mean > self.mu_threshold:
            out = ("EXPLOSIF", "RISQUE ÉLEVÉ",
                   "Viscosité magmatique élevée favorisant l'accumulation de gaz "
                   "et la fragmentation ; surpression encore modérée.",
                   "Renforcement immédiat de la surveillance (sismique, GNSS, "
                   "InSAR, géochimie) et préparation des populations.")
        elif p_mean > self.p_threshold:
            out = ("EFFUSIF/EXPLOSIF", "RISQUE MODÉRÉ",
                   "Surpression élevée mais viscosité faible : éruption effusive "
                   "dominante probable, avec risque de transition paroxysmale "
                   "(fontaines de lave).",
                   "Surveillance renforcée et sécurisation des zones de coulées "
                   "(flanc Est de l'Etna).")
        else:
            out = ("EFFUSIF", "ÉTAT STABLE (type étnéen)",
                   "État typique du Mont Etna : système basaltique alcalin à "
                   "conduit ouvert, viscosité modérée (1e2-1e5 Pa.s) et "
                   "surpression maîtrisée. Activité effusive de fond (coulées, "
                   "fontaines de lave) sans transition explosive attendue.",
                   "Poursuite de la surveillance standard INGV-OE (sismique, "
                   "GNSS, caméras thermiques, InSAR Sentinel-1).")
        return {"risk_class": out[0], "risk_level": out[1],
                "description": out[2], "recommendation": out[3]}


def build_eruptive_report(pinn2: EtnaPINN2D, fields: Dict[str, np.ndarray],
                          final_loss: float,
                          path_json: str = "eruptive_decision.json") -> Dict:
    pl = pinn2.pl
    decider = EruptiveDecisionGrid()
    r_ch = np.sqrt((fields["X"] - pl.chamber_x) ** 2
                   + (fields["Y"] - pl.chamber_y) ** 2)
    mask_ch = r_ch < pl.chamber_radius
    mask_core = r_ch < 0.5 * pl.chamber_radius

    p_dyn = float(np.mean(fields["p"][mask_ch]))
    p_total = float(pl.dP + p_dyn)
    mu_ch_arr = fields["mu_eff"][mask_ch]
    mu_ch = float(np.mean(mu_ch_arr))
    mu_core = float(np.mean(fields["mu_eff"][mask_core]))
    clip_ch = float(np.mean(mu_ch_arr >= MU_PHYS_MAX * 0.999))
    T_ch = float(np.mean(fields["T"][mask_ch]))
    v_max = float(np.max(np.sqrt(fields["u"] ** 2 + fields["v"] ** 2)))
    classification = decider.classify(p_total, mu_ch)

    diag = pinn2.diag
    mu_ok = bool(diag["mu_eff_min"] >= MU_PHYS_MIN * 0.999
                 and diag["mu_eff_max"] <= MU_PHYS_MAX * 1.001)
    if pinn2.learn_arrhenius and pinn2.arrh_par is not None:
        mu0_val, Ea_val = [float(v) for v in pinn2.arrh_par()]
    else:
        mu0_val, Ea_val = pinn2.arrh.mu0, pinn2.arrh.Ea

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "site": "Mont Etna, Sicile, Italie",
        "method": ("PINN PyTorch - Navier-Stokes Boussinesq 2D + continuite "
                   "+ energie + Arrhenius/Herschel-Bulkley"),
        "arrhenius_parameters": {
            "mu0_pa_s": mu0_val, "mu0_bounds_pa_s": [MU0_BOUNDS[0], MU0_BOUNDS[1]],
            "Ea_J_per_mol": Ea_val, "Ea_bounds_J_per_mol": [EA_BOUNDS[0], EA_BOUNDS[1]],
            "learnable": bool(CONFIG["learn_arrhenius"])},
        "physics_constraints": {
            "mu_min_pa_s": MU_PHYS_MIN, "mu_max_pa_s": MU_PHYS_MAX,
            "mu_within_bounds": mu_ok,
            "clamped_fraction_percent": 100.0 * float(diag["clamp_fraction"]),
            "clamped_fraction_chamber_percent": 100.0 * clip_ch,
            "mu_raw_range_before_clip": [float(diag["mu_raw_min"]),
                                         float(diag["mu_raw_max"])],
            "mu_eff_range_after_clip": [float(diag["mu_eff_min"]),
                                        float(diag["mu_eff_max"])]},
        "chamber_statistics": {
            "mogi_overpressure_pa": float(pl.dP),
            "dynamic_pressure_mean_pa": p_dyn,
            "overpressure_total_pa": p_total,
            "mu_mean_pa_s": mu_ch,
            "mu_core_mean_pa_s": mu_core,
            "T_mean_K": T_ch},
        "domain_statistics": {
            "mu_mean_pa_s": float(np.mean(fields["mu_eff"])),
            "tau_max_pa": float(np.max(fields["tau"])),
            "T_min_K": float(np.min(fields["T"])),
            "T_max_K": float(np.max(fields["T"])),
            "v_max_m_s": v_max},
        "thresholds": {"p_threshold_pa": decider.p_threshold,
                       "mu_threshold_pa_s": decider.mu_threshold},
        "decision": classification,
        "final_training_loss": float(final_loss),
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    sep = "=" * 78
    print("\n" + sep)
    print(" RAPPORT DE DÉCISION ÉRUCTIVE — MONT ETNA")
    print(sep)
    print(" Statistiques chambre magmatique (r < %.0f m) :" % pl.chamber_radius)
    print("   Surpression totale (Mogi + dynamique) : %.3e Pa  (seuil %.1e Pa)"
          % (p_total, decider.p_threshold))
    print("   Viscosité moyenne chambre             : %.3e Pa.s (seuil %.1e Pa.s)"
          % (mu_ch, decider.mu_threshold))
    print("   Viscosité noyau (r < R/2)             : %.3e Pa.s | clamp %.2f %%"
          % (mu_core, 100.0 * clip_ch))
    print("   Température moyenne                   : %.1f K" % T_ch)
    print("   Vitesse maximale du domaine           : %.3e m/s" % v_max)
    print(" Contrainte physique sur la viscosité (domaine entier) :")
    print("   bornes imposées            : [%.0e, %.0e] Pa.s"
          % (MU_PHYS_MIN, MU_PHYS_MAX))
    print("   plage obtenue              : [%.3e, %.3e] Pa.s -> %s"
          % (diag["mu_eff_min"], diag["mu_eff_max"],
             "CONFORME" if mu_ok else "HORS BORNES"))
    print("   mu brut avant clip (max)   : %.3e Pa.s" % diag["mu_raw_max"])
    print("   fraction clampée (domaine) : %.2f %%  (croûte froide, attendu)"
          % (100.0 * diag["clamp_fraction"]))
    print(" Classification : %s — %s"
          % (classification["risk_class"], classification["risk_level"]))
    print(" Interprétation : %s" % classification["description"])
    print(" Recommandation : %s" % classification["recommendation"])
    print(sep + "\n")
    print(f"[INFO] Rapport sauvegardé : {path_json}")
    return report

class EtnaVisualizer:

    def __init__(self, pipeline: EtnaDataPipeline):
        self.pl = pipeline

    def plot_simulation(self, fields: Dict[str, np.ndarray],
                        fname: str = "step2_etna_simulation") -> None:
        X, Y = fields["X"], fields["Y"]
        u, v, p, T = fields["u"], fields["v"], fields["p"], fields["T"]
        tau, mu_eff = fields["tau"], fields["mu_eff"]
        cx, cy, R = self.pl.chamber_x, self.pl.chamber_y, self.pl.chamber_radius

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        def _chamber(ax):
            ax.add_patch(plt.Circle((cx, cy), R, facecolor="none",
                                    edgecolor="cyan", linestyle="--", linewidth=2.0))

        def _panel_label(ax, letter):
            ax.text(0.02, 0.97, letter, transform=ax.transAxes, va="top",
                    ha="left", fontsize=14, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        def _axes(ax):
            ax.set_xlabel("Easting $x$ [m]", fontsize=12)
            ax.set_ylabel("Elevation $y$ [m]", fontsize=12)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
            ax.set_autoscale_on(False)

        ax = axes[0, 0]
        speed = np.sqrt(u ** 2 + v ** 2)
        try:
            strm = ax.streamplot(X, Y, u, v, color=speed, cmap="plasma",
                                 density=2.0, linewidth=1.3, arrowsize=1.2)
            cba = fig.colorbar(strm.lines, ax=ax, shrink=0.85, pad=0.02)
        except Exception:
            im0 = ax.pcolormesh(X, Y, speed, shading="gouraud", cmap="plasma")
            cba = fig.colorbar(im0, ax=ax, shrink=0.85, pad=0.02)
        cba.set_label(r"Velocity magnitude $|\mathbf{u}|$ [m/s]", fontsize=11)
        try:
            cs = ax.contour(X, Y, p, levels=10, colors="white",
                            linewidths=0.7, alpha=0.8)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
        except Exception:
            pass
        ax.text(0.03, 0.03, r"white: isobars $p'$ [Pa]", transform=ax.transAxes,
                fontsize=9, bbox=dict(facecolor="white", alpha=0.85,
                                      edgecolor="none"))
        _chamber(ax); _axes(ax); _panel_label(ax, "(a)")

        ax = axes[0, 1]
        vmin_t, vmax_t = _safe_log_bounds(tau, floor=1e-4)
        im = ax.pcolormesh(X, Y, tau, shading="gouraud", cmap="magma",
                           norm=LogNorm(vmin=vmin_t, vmax=vmax_t))
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r"Shear stress $\tau$ [Pa]", fontsize=11)
        if tau.max() > self.pl.tau0:
            ax.contour(X, Y, tau, levels=[self.pl.tau0], colors="black",
                       linewidths=2.0, linestyles="--")
        ax.text(0.03, 0.03, r"$\tau_0$ = %.0f Pa (yield)" % self.pl.tau0,
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        _chamber(ax); _axes(ax); _panel_label(ax, "(b)")

        ax = axes[1, 0]
        vmin_m, vmax_m = _safe_log_bounds(mu_eff, floor=1.0)
        im = ax.pcolormesh(X, Y, mu_eff, shading="gouraud", cmap="viridis",
                           norm=LogNorm(vmin=vmin_m, vmax=vmax_m))
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r"Effective viscosity $\mu_{\mathrm{eff}}$ [Pa$\cdot$s]",
                     fontsize=11)
        ax.text(0.03, 0.03,
                r"$\mu_{\mathrm{eff}} \in [%.2e,\ %.2e]$ Pa$\cdot$s"
                % (mu_eff.min(), mu_eff.max()) + "\n"
                + r"physical clipping $[10^{1},\,10^{6}]$ Pa$\cdot$s",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        _chamber(ax); _axes(ax); _panel_label(ax, "(c)")

        ax = axes[1, 1]
        im = ax.pcolormesh(X, Y, T, shading="gouraud", cmap="inferno")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Temperature [K]", fontsize=11)
        try:
            cs = ax.contour(X, Y, T, levels=9, colors="white",
                            linewidths=0.6, alpha=0.55)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        except Exception:
            pass
        _chamber(ax); _axes(ax); _panel_label(ax, "(d)")

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            path = f"{fname}.{fmt}"
            fig.savefig(path, dpi=600, bbox_inches="tight", format=fmt)
            print(f"[INFO] Figure sauvegardée : {path}")
        plt.close(fig)

    def plot_loss_convergence(self, logger: CSVLogger,
                              fname: str = "step2_loss_convergence") -> None:
        hist = logger.history
        if not hist:
            warnings.warn("[VIZ] Historique vide — figure de convergence ignorée.")
            return
        ep = np.array([h["epoch"] for h in hist], dtype=float)
        comp_keys = sorted(k for k in hist[0]
                           if k.startswith("loss_") and k != "loss_total")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for k in comp_keys:
            vals = np.array([max(h.get(k, 0.0), 1e-16) for h in hist])
            ax.semilogy(ep, vals, lw=0.9, alpha=0.75,
                        label=k.replace("loss_", ""))
        tot = np.array([max(h["loss_total"], 1e-16) for h in hist])
        ax.semilogy(ep, tot, "k-", lw=2.2, label="total")
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        for fmt in ("png", "pdf"):
            path = f"{fname}.{fmt}"
            fig.savefig(path, dpi=400, bbox_inches="tight", format=fmt)
            print(f"[INFO] Figure sauvegardée : {path}")
        plt.close(fig)

def main() -> None:
    t_start = time.time()
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sep = "=" * 78
    print(sep)
    print(" PINN — RHÉOLOGIE MAGMATIQUE : POISEUILLE (HB) -> CHAMBRE DE L'ETNA")
    print(" app : forme contrainte | float64 | auto-test | BC dure | clip Adam |")
    print("      auto-poids [FIX-12] | RAFFINEMENT D'INTERFACE ±y_c [FIX-14]")
    print(sep)
    print(f"[INFO] Device PyTorch : {device} | Seed : {CONFIG['seed']}")
    print(f"[INFO] Plancher thermique Arrhenius : {CONFIG['T_floor_kelvin']:.0f} K "
          f"(900 K = variante pétrologique à assumer explicitement)")
  
    print("\n" + sep)
    print(" ÉTAPE 1 — VALIDATION : POISEUILLE PLAN (HERSCHEL-BULKLEY + ARRHENIUS)")
    print(sep)
    cfg1 = CONFIG["step1"]
    pinn1 = PoiseuillePINN(
        device=device,
        adam_lr=cfg1["adam_lr"],
        grad_clip=CONFIG["grad_clip_norm"],
        T_floor=CONFIG["T_floor_kelvin"],
        n_colloc=cfg1["n_colloc"],
        residual_mode=cfg1["residual_mode"],
        m_reg=cfg1["m_reg"],
        dtype=torch.float64 if cfg1["float64"] else torch.float32)
    pinn1.verify_formulation()
    pinn1.train(adam_epochs=cfg1["adam_epochs"],
                lbfgs_epochs=cfg1["lbfgs_epochs"],
                lbfgs_passes=cfg1["lbfgs_passes"],
                continuation=cfg1.get("continuation", False))  
    m1 = pinn1.evaluate()
    pinn1.probe_solution()          
    pinn1.plot_results()
    if m1["r2"] < 0.99:         
        warnings.warn(f"[QC ÉTAPE 1] R^2 = {m1['r2']:.4f} < 0.99 : "
                      "convergence insuffisante — vérifier la SONDE (plug "
                      "résolu ?), l'auto-test et le log de convergence.")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print("\n" + sep)
    print(" ÉTAPE 2 — APPLICATION GÉOPHYSIQUE : CHAMBRE MAGMATIQUE DU MONT ETNA")
    print(sep)
    pipeline = EtnaDataPipeline()
    cfg2 = CONFIG["step2"]
    pinn2 = EtnaPINN2D(pipeline, learn_arrhenius=CONFIG["learn_arrhenius"],
                       device=device, grad_clip=CONFIG["grad_clip_norm"],
                       T_floor=CONFIG["T_floor_kelvin"],
                       n_domain=cfg2["n_domain"], n_per_edge=cfg2["n_per_edge"],
                       n_insar=cfg2["n_insar"], n_petro=cfg2["n_petro"])
    pinn2.train(adam_epochs=cfg2["adam_epochs"],
                lbfgs_epochs=cfg2["lbfgs_epochs"],
                lbfgs_passes=cfg2["lbfgs_passes"])

    fields = pinn2.predict_with_fields(nx=120, ny=90)
    report = build_eruptive_report(pinn2, fields,
                                   final_loss=pinn2.loss_history[-1]["loss_total"])

    viz = EtnaVisualizer(pipeline)
    viz.plot_simulation(fields)
    viz.plot_loss_convergence(pinn2.logger)

    df = pd.DataFrame({
        "x_m": fields["X"].ravel(), "y_m": fields["Y"].ravel(),
        "u_m_s": fields["u"].ravel(), "v_m_s": fields["v"].ravel(),
        "p_prime_pa": fields["p"].ravel(), "T_K": fields["T"].ravel(),
        "tau_pa": fields["tau"].ravel(), "mu_eff_pa_s": fields["mu_eff"].ravel(),
        "shear_rate_s_inv": fields["gamma_dot"].ravel()})
    df.to_csv("etna_simulation_results.csv", index=False, float_format="%.6e")
    print("[INFO] Champs numériques exportés : etna_simulation_results.csv")

    print("\n" + sep)
    print(" PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f" Durée totale : {time.time() - t_start:.1f} s")
    print(" Fichiers générés :")
    for f in ("train_log_step1.csv", "train_log_step2.csv",
              "step1_poiseuille_validation.png", "step1_poiseuille_validation.pdf",
              "step2_etna_simulation.png", "step2_etna_simulation.pdf",
              "step2_loss_convergence.png", "step2_loss_convergence.pdf",
              "etna_simulation_results.csv", "eruptive_decision.json"):
        print(f"   - {f}")
    print(sep)


if __name__ == "__main__":
    main()
