"""Frozen numerical reference for the ORIGINAL released GranularBall core
(Prompt 1 Part A). Used only to prove that the corrected implementation in
`gbc_mode='static'`, tau=1.0 is mathematically identical to the released code.
The intentional bugs (tau ignored, static-only) live here, NOT in archs_GBC.py.
"""
import torch
import torch.nn.functional as F


def legacy_granularball_recon(z_flat, centers, sigma, tau_argument=1.0):
    """Replicates the released forward core on projected features.

    z_flat : (B, N, d)
    centers: (K, d)
    sigma  : (1, 1, K, d) or (1, 1, K, 1)  (already softplus-positive)
    Returns recon_flat (B, N, d) and att (B, N, K), exactly as the released
    code computed them: att = softmax(-dist2/max(1e-6,tau)); recon = att @ centers.
    """
    dif = z_flat.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)   # (B,N,K,d)
    dif_scaled = dif / sigma
    dist2 = (dif_scaled ** 2).sum(-1)                              # (B,N,K)
    att = F.softmax(-dist2 / max(1e-6, tau_argument), dim=-1)
    recon_flat = torch.matmul(att, centers)                       # (B,N,d)
    return recon_flat, att
