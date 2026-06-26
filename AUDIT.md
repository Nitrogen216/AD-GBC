# AD-GBC Corrected-Code Audit (Prompt 1)

Date: 2026-06-26. Implements the "corrected code paths" pass (M0). Acceptance =
**all unit tests pass on CPU**; `gbc_mode=static`, `tau=1.0` reproduces the
released math; new consensus + W-div modes run forward/backward.

## Discrepancies found in the released code (now fixed)
1. **τ ignored.** `GranularBall.forward(self, x, tau=1.0)` used the local `tau`;
   `self.tau` was stored but never used, and the model called `self.gbc(t3)` /
   `self.gbc(out)` with no `tau`, so `--tau` never took effect. → `forward(self,
   x, tau=None)` with `tau_eff = self.tau if tau is None else tau`.
2. **`--use_diag_cov` could not select isotropic.** `add_argument` had no
   `type=str2bool`, so CLI `"False"` was a truthy string. → added `type=str2bool`.
3. **Optimizer high-LR group too broad.** Any parameter whose name contained
   `gbc` (incl. GBC refine conv/BN and projection conv/BN) went to the `gbc_lr`
   group. → geometry group restricted to `centers`/`log_sigma`/`log_radius`.
4. **`cudnn.benchmark = True` overrode determinism.** `seed_torch()` set
   `benchmark=False, deterministic=True`, then `cudnn.benchmark = True` undid it.
   → both now respect `--deterministic`; `seed_torch(seed, deterministic)`.
5. **No dynamic consensus.** Only `att @ centers`. → added `gbc_mode ∈ {static,
   paper_sum, mean}` (static = legacy; mean = normalized Set→Ball→Set; paper_sum
   = paper Eq.4 literal, un-normalized, diagnostic — **no hidden LayerNorm**).
6. **W-div coefficient + dead-gradient clamp.** Released loss used
   `clamp(||μ||²+tr(Σ)−2tr(√Σ), min=0)` (coefficient ≠ paper; clamp can zero the
   gradient at init). → `wdiv_mode ∈ {legacy, paper, rank_aware}`; rank_aware =
   `||C̄||²+Σ(s_j−1/√D)²` (= paper + r/D, grad-identical, ≥0, stable).
7. **val rebuild.** `val_GBC.py` already rebuilt K/proj/cov/τ from the saved
   config; added `gbc_mode` so non-default models reload correctly.

## Files changed / added
- `archs_GBC.py`: GranularBall τ fix + `gbc_mode`; threaded `gbc_mode` through
  `GBC_Rolling_Unet_S/M/L`; `.view`→`.reshape` after permute/transpose (mps).
- `losses.py`: `wasserstein_diversity_loss(centers, mode=...)`;
  `BCEDiceWithGeometryLoss(..., wdiv_mode=...)`.
- `train_GBC.py`: `--use_diag_cov` str2bool; `--gbc_mode`, `--wdiv_mode`,
  `--train_seed`, `--deterministic`; geometry-only optimizer group; deterministic
  cudnn; thread `gbc_mode`/`wdiv_mode`.
- `val_GBC.py`: rebuild with `gbc_mode`.
- `gbc_diagnostics.py` (new): usage, effective-K, dead-ball, assignment entropy,
  **assignment mutual information**, center spectrum, W-div grad norm.
- `tests/reference_legacy_gbc.py`, `tests/test_gbc.py` (new): 9 tests.
- `metrics_percase.py`, `tests/test_metrics.py`, `scripts/audit_data.py`,
  `docs/EP_GBC_SPEC.md` (earlier M−1/A000/A001 work).

## Verification
- `tests/test_metrics.py`: **9/9 pass** (CPU).
- `tests/test_gbc.py`: **9/9 pass** (CPU) — incl. static-τ1 == legacy reference,
  τ→entropy, self.tau honoured, isotropic scalar scale, all modes fwd/bwd,
  rank_aware == paper + r/D with non-zero grad at init, uniform-α ⇒ MI = 0.
- Full `GBC_Rolling_Unet_S` forward+backward on **CPU**: OK for static/mean/paper_sum.
- **mps forward (eval): OK.**

## Known limitation (out of Prompt 1 scope)
- **mps backward (training) fails in the upstream Rolling-UNet backbone**:
  BatchNorm backward receives non-contiguous gradients on the mps backend
  (`view size is not compatible ... use reshape`). This is a pre-existing
  PyTorch-mps quirk in the backbone, NOT in the GBC module (GBC forward/backward
  is mps-safe; CPU training works fully). Verified training path = CPU/CUDA; mps
  is forward/val-only until a dedicated backbone mps-compat pass (likely
  `.contiguous()` before decoder BatchNorms). Tracked as a follow-up.

## Not in this prompt (later)
EP-GBC `evidence_posterior`/`observed_gmm`/`free_gate`, Gaussian energy,
posterior variance, scale_target_center=region — Prompt 2 per `EP_GBC_SPEC.md`.
