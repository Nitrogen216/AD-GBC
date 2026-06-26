# EP-GBC — Math & Naming Specification (A000, FROZEN 2026-06-25)

Authoritative formula/symbol/naming freeze for the AD-GBC follow-up. **No EP-GBC
variant may be implemented until this is agreed.** Derived from the 2026-06-25
advisor reviews (see `../../notes/advisor_meetings/`). All code must run on
mps + cuda + cpu.

## 1. Name (frozen)
- **Method = EP-GBC (Evidence-Posterior Granular Balls)** under **Plan A**: the
  Gaussian energy (§4) is the *core* assignment and a full MAP derivation (§6) is
  given. The name "posterior/Evidence-Posterior" is licensed only because of this.
- **Fallback (Plan B), if Gaussian energy destabilizes training:** keep the
  Mahalanobis energy and **rename to "Mass-Adaptive Consensus GBC" / "Evidence-
  Weighted Shrinkage GBC"** — and drop "posterior" language. Decide at end of M1.

## 2. Symbols (frozen — resolves the λ collision)
| Symbol | Meaning |
|---|---|
| `N = H·W` | pixels per image (per GBC placement, at that feature resolution) |
| `K` | number of granular balls |
| `D` | feature dim at the GBC placement |
| `z_i ∈ R^D` | pixel feature i |
| `c_k ∈ R^D` | global learned ball center (prior mean) |
| `σ_k² ∈ R^D_{>0}` | global learned diagonal scale (prior variance); `σ_k = softplus(log_sigma)` |
| `π_k` | mixture weight — **v1: fixed `π_k = 1/K`** (so `−2 log π_k` is constant, dropped). Learned/usage π is deferred. |
| `α_{ik}` | soft assignment of pixel i to ball k, `Σ_k α_{ik}=1` |
| `m_k = Σ_i α_{ik}` | soft ball mass (coverage) |
| `κ` | resolution-normalized prior fraction (scalar; `learn_kappa` optional) |
| `ν = κ·N` | **pseudo-count** (prior strength in evidence units) |
| `λ_W` | Wasserstein/diversity loss weight (was AD-GBC's `λ`) |
| `λ_S` | scale-consistency loss weight |
| `τ` | assignment temperature |
The bare symbol `λ` is **retired**.

## 3. Reduction convention
Distance and log-volume terms use the **same reduction over D**. Default
`distance_reduction = sum`. Dividing the energy by `D` (mean) is an allowed
alternative that makes `τ` comparable across S/M/L; pick one in config and keep it
fixed within an experiment.

## 4. Assignment energy (core = Gaussian)
```
E_{ik} = ½ [ Σ_d (z_{id} − c_{kd})² / (σ_{kd}² + ε)        # Mahalanobis
           + Σ_d log(σ_{kd}² + ε)                          # log-volume
           − 2 log π_k ]                                   # = const for π_k=1/K (dropped)
α_{ik} = softmax_k( − E_{ik} / τ_eff )                     # τ_eff = self.tau if tau is None
```
`energy_mode ∈ {mahalanobis, gaussian}`: `mahalanobis` drops the log-volume term
(volume_weight γ=0); `gaussian` keeps it (γ=1). The M0 **energy-scale test (R008)**
must confirm the Mahalanobis and log-volume terms are commensurate (at D=128/256
the log-volume term can dominate) before trusting `gaussian`.

## 5. Sufficient statistics (per image b)
```
m_{bk}  = Σ_i α_{bik}                       ∈ R^{K}
S¹_{bk} = Σ_i α_{bik} z_{bi}                ∈ R^{K×D}
S²_{bk} = Σ_i α_{bik} z_{bi}²              ∈ R^{K×D}   (elementwise square)
```

## 6. Posterior center (MAP) — the core update
Model (per ball, conditionally): `z_i | μ_k ~ N(μ_k, Σ_k)`,
prior `μ_k ~ N(c_k, Σ_k/ν)`, with `α_{ik}` as fractional responsibilities and
assignment = Gaussian responsibility (§4). Then the posterior mean of `μ_k` is the
precision-weighted average:
```
c̃_{bk} = (S¹_{bk} + ν·c_k) / (m_{bk} + ν)
       = (1 − ρ_{bk})·c_k + ρ_{bk}·μ_{bk},   μ_{bk}=S¹_{bk}/m_{bk}
ρ_{bk} = m_{bk} / (m_{bk} + ν),              ν = κN
```
`ρ` = reliability/evidence ratio. **`ρ` is NOT a success signal**: uniform
assignment `α=1/K` gives `m_k=N/K` and (with ν=N/K, i.e. κ=1/K) `ρ=0.5`
automatically. Reliability must be shown via synthetic calibration (R007) +
assignment mutual information (§8), not by ρ sitting near 0.5.

`consensus_mode ∈ {static, paper_sum(diag), normalized_mean, observed_gmm,
free_gate, evidence_posterior}`. Limits (unit-tested R004): `ν→∞ ⇒ static`,
`ν→0 ⇒ normalized_mean`.

## 7. Two distinct variances (MUST not be conflated)
```
posterior_center_var_{bk} = Σ_k / (m_{bk} + ν)      # uncertainty OF THE ESTIMATE c̃
fused_component_var_{bk}  = q̃_{bk} − c̃_{bk}²,
   q̃_{bk} = (S²_{bk} + ν·(c_k² + σ_k²)) / (m_{bk} + ν)   # predictive dispersion of members
```
Clamp `fused_component_var ≥ posterior_var_floor`. The scale-consistency loss
targets the **member dispersion** (`fused_component_var`), not the center-estimate
uncertainty. Never label `fused_component_var` as "posterior variance".

## 8. Diagnostics (frozen definitions)
```
usage p_k = (1/B) Σ_b m_{bk}/N ;  effective_K = exp(H(p))
assignment_entropy = mean_i H(α_i)
assignment_mutual_information  I(Z;K) = H(K) − H(K|Z),
   H(K)=H(p),  H(K|Z)=mean_i H(α_i)        # distinguishes sharp vs 1/K assignment
dead_ball_ratio = fraction of balls with usage < threshold
posterior_displacement δ_{bk} = ‖c̃_{bk} − c_k‖ / (‖σ_k‖ + ε)
center spectrum: singular values / effective rank / participation ratio of {c_k}
```

## 9. Config schema (frozen keys)
`consensus_mode`, `energy_mode`, `distance_reduction`, `tau`, `use_diag_cov`,
`gbc_num_balls(K)`, `gbc_proj_dim(D)`, `kappa`, `learn_kappa`, `posterior_steps∈{0,1}`,
`posterior_var_floor`, `detach_evidence_gate`, `wdiv_mode∈{legacy,paper,rank_aware}`,
`scale_target_center∈{static,region}`, `preprocess_mode∈{legacy,corrected}`,
`split_seed`, `train_seed`, `deterministic`. Loss weights: `λ_W (div_weight)`,
`λ_S (scale_weight)`. `val_GBC.py` must rebuild the model from ALL of these.

## 10. Rank-aware W-div (frozen)
`X = (C − C̄)/√K`, singular values `s_j`, `r = min(D, K−1)`:
```
L_rank-W = ‖C̄‖² + Σ_{j=1}^r (s_j − 1/√D)²        # = paper Eq.8 + r/D (grad-identical), ≥0
```
SVD in float32. No outer `clamp(min=0)`.
