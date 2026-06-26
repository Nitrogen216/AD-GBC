"""Granular-ball diagnostics (Prompt 1 Part F; A000 spec section 8).

All functions take torch tensors and return python floats / small tensors. They
are backend-agnostic (cpu/mps/cuda). `att` is the soft assignment [B, N, K] with
rows summing to 1 over K.
"""
import torch

EPS = 1e-9


def _entropy(p, dim=-1):
    return -(p.clamp_min(EPS) * p.clamp_min(EPS).log()).sum(dim)


def normalized_ball_usage(att):
    """Mean assignment mass per ball, normalized to a distribution -> [K]."""
    usage = att.mean(dim=(0, 1))            # mean over batch and pixels
    return usage / usage.sum().clamp_min(EPS)


def assignment_entropy(att):
    """Mean per-pixel assignment entropy (nats)."""
    return float(_entropy(att, dim=-1).mean())


def effective_k(usage):
    """exp(H(usage)) — effective number of balls in use."""
    return float(torch.exp(_entropy(usage, dim=-1)))


def dead_ball_ratio(usage, threshold=1e-4):
    return float((usage < threshold).float().mean())


def assignment_mutual_information(att):
    """I(Z;K) = H(K) - H(K|Z).

    H(K)   = entropy of the marginal ball usage.
    H(K|Z) = mean per-pixel assignment entropy.
    Distinguishes 'uniform usage + sharp per-pixel assignment' (high MI) from
    'uniform usage because every pixel is 1/K' (MI = 0). The latter is the ρ=0.5
    trap; effective-K alone cannot tell them apart.
    """
    usage = normalized_ball_usage(att)
    h_k = float(_entropy(usage, dim=-1))
    h_k_given_z = assignment_entropy(att)
    return h_k - h_k_given_z


def center_spectrum(centers, rank_thresholds=(1e-3, 1e-2)):
    """Singular-value based stats of the (K, D) center matrix (centered)."""
    c = centers.detach().float()
    c = c - c.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(c)
    s_max = float(s.max()) if s.numel() else 0.0
    out = {
        "singular_values": s.tolist(),
        "stable_rank": float((s.pow(2).sum() / s.max().clamp_min(EPS) ** 2)) if s.numel() else 0.0,
        # participation ratio of the squared spectrum
        "participation_ratio": float(s.pow(2).sum() ** 2 / s.pow(4).sum().clamp_min(EPS)) if s.numel() else 0.0,
    }
    for t in rank_thresholds:
        out[f"rank@{t}"] = int((s > t * (s_max + EPS)).sum()) if s_max > 0 else 0
    return out


def wdiv_gradient_norm(loss_value, centers):
    """L2 norm of d(loss)/d(centers) — to confirm the W-div term is not dead."""
    g = torch.autograd.grad(loss_value, centers, retain_graph=True, allow_unused=True)[0]
    if g is None:
        return 0.0
    return float(g.norm())


def summarize(att, centers=None):
    usage = normalized_ball_usage(att)
    out = {
        "assignment_entropy": assignment_entropy(att),
        "effective_k": effective_k(usage),
        "dead_ball_ratio": dead_ball_ratio(usage),
        "assignment_mutual_information": assignment_mutual_information(att),
    }
    if centers is not None:
        spec = center_spectrum(centers)
        out["center_stable_rank"] = spec["stable_rank"]
        out["center_participation_ratio"] = spec["participation_ratio"]
    return out
