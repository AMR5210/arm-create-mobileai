"""Pure-Python (numpy) Q2_K encoder for GGUF export.

Why this exists
---------------
The QAT fine-tune simulates 2-bit weights with a simple per-group affine
min/max scheme (qat/fake_quant.py). The deployment format, GGUF ``Q2_K``, uses
a different, richer layout (256-element super-blocks split into 16 sub-blocks
of 16, with the per-sub-block scale and min themselves 4-bit-quantized under an
fp16 super-block scale). The old export path piped the QAT weights through
``llama-quantize``, which *re-derives* its own Q2_K scheme from the raw fp16
weights -- so what ran on device was never exactly what training optimized.

This module lets us pack tensors into Q2_K ourselves, in-process, so the export
step is transparent and controllable (e.g. force specific tensors to stay F16 --
the mixed-precision skip-layers). The Q2_K math here is a faithful port of
ggml's reference quantiser (``quantize_row_q2_K_ref`` + ``make_qkx2_quants`` in
ggml-quants.c), so output quality matches llama.cpp's own Q2_K.

Correctness is checkable without a C toolchain: ``roundtrip_error`` below
re-decodes our packed blocks with gguf-py's *trusted* ``Q2_K.dequantize_blocks``
(the numpy mirror of ggml's ``dequantize_row_q2_K``). If our byte layout were
wrong, that decode would not reconstruct the input -- so a small round-trip
error validates the packing exactly, and the quality of the quantiser
separately.

Note on granularity: Q2_K sub-blocks are 16 elements, so training with
``--group-size 16`` (see qat/apply_qat.py) aligns the QAT grouping with Q2_K's
and is the intended companion to this encoder. Packing a checkpoint trained at
group_size 32 still works, but the trained values do not lie exactly on the
Q2_K grid and are re-quantised here (an approximation), so the largest quality
win comes from a group_size-16 retrain.
"""
from __future__ import annotations

import numpy as np

QK_K = 256          # Q2_K super-block size (elements)
SUBBLOCK = 16       # elements per sub-block
N_SUB = QK_K // SUBBLOCK   # 16 sub-blocks per super-block
Q2K_BLOCK_BYTES = 84       # 16 (scales) + 64 (qs) + 2 (d) + 2 (dmin)


def _nearest_int(x: np.ndarray) -> np.ndarray:
    """Round half to even, matching ggml's nearest_int() magic-number trick."""
    return np.rint(x)


def _make_qkx2_quants(
    X: np.ndarray,
    W: np.ndarray,
    nmax: int = 3,
    rmin: float = -0.5,
    rdelta: float = 0.1,
    nstep: int = 15,
    use_mad: bool = True,
):
    """Vectorised port of ggml's make_qkx2_quants.

    X, W: (M, 16) float32 -- one sub-block per row, W = |X| weights.
    Returns (scale, the_min) each (M,) float32. ``the_min`` is ``-min`` (the
    non-negative offset stored downstream), matching the C ``*the_min`` output.
    Per-block integer codes are intentionally NOT returned: the reference
    recomputes them in a second pass after the scale/min are 4-bit-quantised.
    """
    X = X.astype(np.float32, copy=False)
    W = W.astype(np.float32, copy=False)

    xmin = np.minimum(X.min(axis=1), 0.0)   # force min <= 0
    xmax = X.max(axis=1)
    sum_w = W.sum(axis=1)
    sum_x = (W * X).sum(axis=1)

    rng = xmax - xmin
    degenerate = rng == 0.0
    rng_safe = np.where(degenerate, 1.0, rng)

    # Initial guess: uniform quantiser over [min, max].
    iscale = nmax / rng_safe
    scale = 1.0 / iscale
    L0 = np.clip(_nearest_int(iscale[:, None] * (X - xmin[:, None])), 0, nmax)
    diff0 = scale[:, None] * L0 + xmin[:, None] - X
    diff0 = np.abs(diff0) if use_mad else diff0 * diff0
    best_mad = (W * diff0).sum(axis=1)

    cur_scale = scale.copy()
    cur_min = xmin.copy()

    for is_ in range(nstep + 1):
        denom = xmax - cur_min
        denom_safe = np.where(denom == 0.0, 1.0, denom)
        iscale = (rmin + rdelta * is_ + nmax) / denom_safe

        Laux = np.clip(_nearest_int(iscale[:, None] * (X - cur_min[:, None])), 0, nmax)
        sum_l = (W * Laux).sum(axis=1)
        sum_l2 = (W * Laux * Laux).sum(axis=1)
        sum_xl = (W * Laux * X).sum(axis=1)

        D = sum_w * sum_l2 - sum_l * sum_l
        goodD = D > 0.0
        D_safe = np.where(goodD, D, 1.0)

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D_safe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D_safe

        # If the fitted min is positive, clamp it to 0 and refit scale.
        posmin = this_min > 0.0
        sl2_safe = np.where(sum_l2 == 0.0, 1.0, sum_l2)
        this_scale = np.where(posmin, sum_xl / sl2_safe, this_scale)
        this_min = np.where(posmin, 0.0, this_min)

        recon = this_scale[:, None] * Laux + this_min[:, None] - X
        recon = np.abs(recon) if use_mad else recon * recon
        mad = (W * recon).sum(axis=1)

        improve = goodD & (mad < best_mad)
        best_mad = np.where(improve, mad, best_mad)
        cur_scale = np.where(improve, this_scale, cur_scale)
        cur_min = np.where(improve, this_min, cur_min)

    scale_out = np.where(degenerate, 0.0, cur_scale).astype(np.float32)
    the_min = np.where(degenerate, -xmin, -cur_min).astype(np.float32)
    return scale_out, the_min


# Byte b in [0,64) holds 4 quants at bit-shifts 0/2/4/6, drawn from linear
# element indices n = G*128 + S*32 + E with G=b//32, E=b%32, S the shift index.
# (Same mapping ggml uses when packing y.qs; the inverse of gguf-py's decode.)
_B = np.arange(64)
_QS_IDX = ((_B // 32) * 128 + (_B % 32))[:, None] + (np.arange(4) * 32)[None, :]  # (64, 4)


def quantize_q2_k(weight: np.ndarray) -> np.ndarray:
    """Encode a 2-D weight (out_features, in_features) into Q2_K blocks.

    ``in_features`` must be divisible by 256. Returns a uint8 array of shape
    (out_features, n_superblocks * 84) laid out exactly as GGUF stores Q2_K:
    per output row, consecutive 84-byte super-blocks over the input dimension.
    """
    w = np.ascontiguousarray(weight, dtype=np.float32)
    R, C = w.shape
    if C % QK_K != 0:
        raise ValueError(f"in_features {C} not divisible by {QK_K} (Q2_K super-block)")
    n_sb = C // QK_K

    # Sub-blocks of 16 across the whole tensor: (M, 16).
    sub = w.reshape(R * (C // SUBBLOCK), SUBBLOCK)
    scales, mins = _make_qkx2_quants(sub, np.abs(sub))

    scales = scales.reshape(R, n_sb, N_SUB)   # per super-block, 16 sub-block scales
    mins = mins.reshape(R, n_sb, N_SUB)

    # Quantise the 16 sub-block scales/mins to 4 bits under an fp16 super scale.
    q4 = 15.0
    max_scale = scales.max(axis=2)            # (R, n_sb)
    max_min = mins.max(axis=2)
    has_s = max_scale > 0.0
    has_m = max_min > 0.0

    isc = np.where(has_s, q4 / np.where(has_s, max_scale, 1.0), 0.0)
    imn = np.where(has_m, q4 / np.where(has_m, max_min, 1.0), 0.0)
    scale_nib = np.clip(_nearest_int(isc[..., None] * scales), 0, 15).astype(np.uint8)
    min_nib = np.clip(_nearest_int(imn[..., None] * mins), 0, 15).astype(np.uint8)
    scales_byte = (scale_nib | (min_nib << 4)).astype(np.uint8)   # (R, n_sb, 16)

    d = np.where(has_s, max_scale / q4, 0.0).astype(np.float16)   # (R, n_sb)
    dmin = np.where(has_m, max_min / q4, 0.0).astype(np.float16)

    # Second pass: recompute 2-bit codes against the stored (fp16 + 4-bit) grid.
    d_f = d.astype(np.float32)
    dmin_f = dmin.astype(np.float32)
    d_local = d_f[..., None] * scale_nib.astype(np.float32)       # (R, n_sb, 16)
    dm_local = dmin_f[..., None] * min_nib.astype(np.float32)

    wsb = w.reshape(R, n_sb, N_SUB, SUBBLOCK)
    dl_safe = np.where(d_local == 0.0, 1.0, d_local)[..., None]
    L = np.clip(_nearest_int((wsb + dm_local[..., None]) / dl_safe), 0, 3).astype(np.uint8)
    L = np.where((d_local == 0.0)[..., None], np.uint8(0), L)      # (R, n_sb, 16, 16)

    # Pack codes: linear index n = sub*16 + elem within a super-block.
    L_lin = L.reshape(R, n_sb, QK_K)
    gathered = L_lin[:, :, _QS_IDX]                                # (R, n_sb, 64, 4)
    shifts = (np.arange(4) * 2).astype(np.uint8)
    qs = np.zeros((R, n_sb, 64), dtype=np.uint8)
    for s in range(4):
        qs |= (gathered[:, :, :, s] << shifts[s]).astype(np.uint8)

    # Assemble 84-byte blocks: [scales(16) | qs(64) | d(2) | dmin(2)].
    d_bytes = d.view(np.uint8).reshape(R, n_sb, 2)
    dmin_bytes = dmin.view(np.uint8).reshape(R, n_sb, 2)
    blocks = np.concatenate([scales_byte, qs, d_bytes, dmin_bytes], axis=2)  # (R, n_sb, 84)
    assert blocks.shape[2] == Q2K_BLOCK_BYTES
    return blocks.reshape(R, n_sb * Q2K_BLOCK_BYTES)


def dequantize_q2_k(blocks_2d: np.ndarray, in_features: int) -> np.ndarray:
    """Decode Q2_K bytes back to float32 (out, in) using gguf-py's trusted
    reference decoder -- used only for the round-trip self-check."""
    from gguf.quants import Q2_K

    R = blocks_2d.shape[0]
    n_sb = in_features // QK_K
    blk = blocks_2d.reshape(R * n_sb, Q2K_BLOCK_BYTES)
    out = Q2_K.dequantize_blocks(blk)          # (R*n_sb, 256)
    return out.reshape(R, in_features)


def roundtrip_error(weight: np.ndarray) -> dict:
    """Encode then decode a weight; report reconstruction error. A small error
    proves both the byte layout (decoded by the trusted reference) and the
    quantiser quality."""
    w = np.ascontiguousarray(weight, dtype=np.float32)
    packed = quantize_q2_k(w)
    recon = dequantize_q2_k(packed, w.shape[1])
    err = recon - w
    denom = float(np.abs(w).mean()) or 1.0
    return {
        "mean_abs_err": float(np.abs(err).mean()),
        "max_abs_err": float(np.abs(err).max()),
        "rel_mean_abs_err": float(np.abs(err).mean() / denom),
        "packed_bytes": int(packed.nbytes),
    }
