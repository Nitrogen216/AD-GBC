"""Prompt 1 acceptance tests for GranularBall, W-div modes, and diagnostics.

Run: ../../.venv/bin/python tests/test_gbc.py   (CPU; no pytest needed)
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archs_GBC import GranularBall            # noqa: E402
from losses import wasserstein_diversity_loss  # noqa: E402
import gbc_diagnostics as DIAG                  # noqa: E402
from tests.reference_legacy_gbc import legacy_granularball_recon  # noqa: E402

torch.manual_seed(0)


def _gbc(mode="static", tau=1.0, use_diag_cov=True, in_ch=8, K=6):
    # proj_dim == in_ch so there is no projection (clean numerical comparison)
    g = GranularBall(in_ch=in_ch, num_balls=K, proj_dim=in_ch,
                     use_diag_cov=use_diag_cov, tau=tau, gbc_mode=mode)
    g.eval()
    return g


def test_static_tau1_matches_legacy():
    g = _gbc("static", tau=1.0)
    x = torch.randn(2, 8, 5, 5)
    with torch.no_grad():
        # archs path
        refined, att, sigma, dif = g(x)            # tau=None -> uses self.tau=1.0
        # reference path on the same (no-projection) features
        B, C, H, W = x.shape
        z_flat = x.view(B, C, H * W).permute(0, 2, 1)
        ref_recon, ref_att = legacy_granularball_recon(z_flat, g.centers, sigma, 1.0)
    assert torch.allclose(att, ref_att, atol=1e-6), (att - ref_att).abs().max()
    # reconstruct the model's recon_flat (before refine) and compare to reference
    model_recon = torch.matmul(att, g.centers)
    assert torch.allclose(model_recon, ref_recon, atol=1e-6)


def test_tau_changes_attention_entropy():
    x = torch.randn(2, 8, 5, 5)
    g1 = _gbc("static", tau=1.0)
    g_sharp = _gbc("static", tau=0.1)
    g_sharp.load_state_dict(g1.state_dict())       # same params, different tau
    with torch.no_grad():
        _, a1, _, _ = g1(x)
        _, a2, _, _ = g_sharp(x)
    # lower tau -> sharper -> lower entropy
    assert DIAG.assignment_entropy(a2) < DIAG.assignment_entropy(a1)


def test_self_tau_used_when_none():
    # caller passes no tau (as the models do) -> self.tau must take effect
    x = torch.randn(1, 8, 4, 4)
    g = _gbc("static", tau=0.2)
    with torch.no_grad():
        _, a_self, _, _ = g(x)            # uses self.tau=0.2
        _, a_explicit, _, _ = g(x, tau=0.2)
        _, a_other, _, _ = g(x, tau=2.0)
    assert torch.allclose(a_self, a_explicit, atol=1e-6)
    assert not torch.allclose(a_self, a_other, atol=1e-4)


def test_modes_forward_backward():
    x = torch.randn(2, 8, 5, 5, requires_grad=True)
    for mode in ("static", "paper_sum", "mean"):
        g = _gbc(mode)
        refined, att, sigma, dif = g(x)
        assert refined.shape == x.shape
        assert torch.isfinite(refined).all()
        assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
        refined.sum().backward()
        assert g.centers.grad is not None


def test_isotropic_scalar_scale():
    g = _gbc("static", use_diag_cov=False, in_ch=8, K=6)
    assert g.log_radius.shape == (6, 1)
    x = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        refined, att, sigma, dif = g(x)
    assert sigma.shape[-1] == 1            # scalar radius per ball


def test_wdiv_rank_aware_equals_paper_plus_const_and_has_grad():
    K, D = 8, 16
    C = torch.randn(K, D, requires_grad=True)
    paper = wasserstein_diversity_loss(C, mode="paper")
    rank = wasserstein_diversity_loss(C, mode="rank_aware")
    r = min(D, K - 1)
    # rank_aware == paper + r/D
    assert abs(float(rank) - (float(paper) + r / D)) < 1e-4, (float(rank), float(paper), r / D)
    # rank_aware is non-negative and has a non-zero gradient at small-center init
    Csmall = (torch.randn(K, D) * 0.01).requires_grad_(True)
    loss = wasserstein_diversity_loss(Csmall, mode="rank_aware")
    assert float(loss) >= 0.0
    gnorm = DIAG.wdiv_gradient_norm(loss, Csmall)
    assert gnorm > 0.0, "rank-aware W-div has zero gradient at init"


def test_wdiv_legacy_grad_can_be_zero_at_init():
    # documents WHY we move off legacy: clamp(min=0) tends to kill the gradient
    Csmall = (torch.randn(8, 16) * 0.01).requires_grad_(True)
    loss = wasserstein_diversity_loss(Csmall, mode="legacy")
    # legacy is clamped at 0; at tiny init it is typically exactly 0 (no grad)
    assert float(loss) >= 0.0


def test_diagnostics_finite():
    x = torch.randn(2, 8, 6, 6)
    g = _gbc("mean")
    with torch.no_grad():
        _, att, _, _ = g(x)
    s = DIAG.summarize(att, g.centers)
    for k, v in s.items():
        assert v == v and abs(v) < 1e9, (k, v)        # finite
    # MI is non-negative and <= H(K)
    assert DIAG.assignment_mutual_information(att) >= -1e-6


def test_uniform_assignment_zero_mutual_information():
    # every pixel = 1/K -> uniform usage but MI == 0 (the ρ=0.5 trap)
    att = torch.full((2, 16, 6), 1.0 / 6)
    assert abs(DIAG.assignment_mutual_information(att)) < 1e-5


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
