"""
KeSSie HIP-Compatible Kernels for MI250X (gfx90a)
====================================================
PROPRIETARY & CONFIDENTIAL

Triton kernels targeting AMD MI250X via ROCm:

1. kessie_fused_fog_attention  - Flash Attention with FUSED fog decay
   (eliminates separate V *= fog_mask pass — saves one full V read/write per layer)
2. kessie_windowed_attention   - Flash Attention with window bounds (original)
3. kessie_paged_kv_manager     - Paged KV cache with fog-aware eviction/insertion
4. kessie_rope_remap           - RoPE position re-rotation for probed KV
5. kessie_index_query          - GPU cosine similarity over conversation index

REQUIREMENTS:
  - ROCm 6.1+ with Triton 3.x
  - PyTorch 2.4+ (ROCm build)
  - Target: gfx90a (MI250X), also works on gfx942 (MI300X)

USAGE:
  python kessie_hip_kernels.py --test all
"""

import torch
import math
import time
import argparse
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger("KeSSie.HIP")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    logger.info(f"Triton {triton.__version__} loaded")
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not available. Install: pip install triton")


# =============================================================================
# Kernel 1: Fused Fog+Attention (NEW — Proposal 8)
# =============================================================================
# The original pipeline:
#   1. fog_mask = build_fog_decay(window_len, fog_alpha, fog_start)  [kernel launch]
#   2. V_fogged = V * fog_mask[None, None, :, None]                 [full V read+write]
#   3. O = flash_attention(Q, K, V_fogged)                          [full V read again]
#
# This kernel fuses step 2 into step 3: during the attention reduction,
# each V row is scaled by its fog weight before accumulation.
# Saves one full V tensor read/write per layer (2 * B*H*L*D * sizeof(fp16) bytes).
#
# On MI250X with 8B model (32 layers, 8 heads, 128 dim, 4096 window):
#   Savings = 32 * 2 * 1 * 8 * 4096 * 128 * 2 bytes = 512 MB bandwidth per forward
#   At 1.6 TB/s HBM bandwidth, that's ~0.3ms saved per forward pass.

if HAS_TRITON:
    @triton.jit
    def _kessie_fused_fog_attn_kernel(
        Q, K, V, Out,
        fog_weights,        # (N_CTX,) float32 — per-position fog multiplier
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX, HEAD_DIM: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Flash Attention with fused fog decay on V.

        Identical to _kessie_attn_fwd_kernel EXCEPT:
        - Each V[n] is multiplied by fog_weights[n] during accumulation
        - fog_weights is a 1D tensor: 1.0 for clear tokens, <1.0 for fogged
        - No separate fog pass needed — V is never modified in-place
        """
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)
        pid_h = tl.program_id(2)

        qkv_offset = pid_z * stride_qz + pid_h * stride_qh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)

        # Load Q block
        q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # KV iteration bounds
        q_pos_max = (pid_m + 1) * BLOCK_M

        if WINDOW_SIZE > 0:
            kv_start = tl.maximum(0, q_pos_max - WINDOW_SIZE)
            kv_start_block = kv_start // BLOCK_N
        else:
            kv_start_block = 0

        if IS_CAUSAL:
            kv_end_block = (q_pos_max + BLOCK_N - 1) // BLOCK_N
        else:
            kv_end_block = (N_CTX + BLOCK_N - 1) // BLOCK_N

        for block_n in range(kv_start_block, kv_end_block):
            kv_offs_n = block_n * BLOCK_N + offs_n

            # Load K block
            k_ptrs = K + qkv_offset + kv_offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
            k = tl.load(k_ptrs, mask=kv_offs_n[None, :] < N_CTX, other=0.0)

            # QK^T
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k) * sm_scale

            # Causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= kv_offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Window mask
            if WINDOW_SIZE > 0:
                window_mask = (offs_m[:, None] - kv_offs_n[None, :]) < WINDOW_SIZE
                window_mask = window_mask & (kv_offs_n[None, :] >= 0)
                qk = tl.where(window_mask, qk, float("-inf"))

            bounds_mask = kv_offs_n[None, :] < N_CTX
            qk = tl.where(bounds_mask, qk, float("-inf"))

            # Online softmax
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)

            # Load V block
            v_ptrs = V + qkv_offset + kv_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=kv_offs_n[:, None] < N_CTX, other=0.0)

            # === FUSED FOG DECAY ===
            # Load fog weights for this KV block and scale V in-register
            fog = tl.load(fog_weights + kv_offs_n, mask=kv_offs_n < N_CTX, other=0.0)
            v = v * fog[:, None]
            # === END FUSED FOG ===

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        acc = acc / l_i[:, None]

        out_ptrs = Out + qkv_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


# =============================================================================
# Kernel 1b: Original Windowed Attention (no fog fusion)
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _kessie_attn_fwd_kernel(
        Q, K, V, Out,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX, HEAD_DIM: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """KeSSie Windowed Flash Attention Forward Kernel (no fog fusion)."""
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)
        pid_h = tl.program_id(2)

        qkv_offset = pid_z * stride_qz + pid_h * stride_qh
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)

        q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q_pos_max = (pid_m + 1) * BLOCK_M
        if WINDOW_SIZE > 0:
            kv_start_block = tl.maximum(0, q_pos_max - WINDOW_SIZE) // BLOCK_N
        else:
            kv_start_block = 0
        if IS_CAUSAL:
            kv_end_block = (q_pos_max + BLOCK_N - 1) // BLOCK_N
        else:
            kv_end_block = (N_CTX + BLOCK_N - 1) // BLOCK_N

        for block_n in range(kv_start_block, kv_end_block):
            kv_offs_n = block_n * BLOCK_N + offs_n
            k_ptrs = K + qkv_offset + kv_offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
            k = tl.load(k_ptrs, mask=kv_offs_n[None, :] < N_CTX, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k) * sm_scale

            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kv_offs_n[None, :], qk, float("-inf"))
            if WINDOW_SIZE > 0:
                w_mask = ((offs_m[:, None] - kv_offs_n[None, :]) < WINDOW_SIZE) & (kv_offs_n[None, :] >= 0)
                qk = tl.where(w_mask, qk, float("-inf"))
            qk = tl.where(kv_offs_n[None, :] < N_CTX, qk, float("-inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)

            v_ptrs = V + qkv_offset + kv_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=kv_offs_n[:, None] < N_CTX, other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        acc = acc / l_i[:, None]
        out_ptrs = Out + qkv_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


# =============================================================================
# Kernel 2: Paged KV Cache
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _kessie_page_evict_kernel(
        KV_cache, page_table, page_positions, page_valid,
        evict_before_pos,
        NUM_PAGES: tl.constexpr, PAGE_SIZE: tl.constexpr,
    ):
        """Evict pages with position < evict_before_pos."""
        pid = tl.program_id(0)
        if pid < NUM_PAGES:
            valid = tl.load(page_valid + pid)
            pos = tl.load(page_positions + pid)
            if (valid == 1) & (pos < evict_before_pos):
                tl.store(page_valid + pid, tl.zeros([], dtype=tl.int8))

    @triton.jit
    def _kessie_page_insert_kernel(
        KV_cache, new_keys, new_values,
        free_page_ids, insert_position,
        page_positions, page_valid,
        stride_kv_page, stride_kv_tok, stride_kv_kv, stride_kv_h, stride_kv_d,
        NUM_TOKENS: tl.constexpr, NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr, PAGE_SIZE: tl.constexpr,
    ):
        """Insert probe KV into free pages (un-fog operation)."""
        pid = tl.program_id(0)
        head_id = tl.program_id(1)
        if pid < NUM_TOKENS:
            page_idx = pid // PAGE_SIZE
            tok_in_page = pid % PAGE_SIZE
            phys_page = tl.load(free_page_ids + page_idx)
            offs_d = tl.arange(0, HEAD_DIM)
            k = tl.load(new_keys + pid * NUM_HEADS * HEAD_DIM + head_id * HEAD_DIM + offs_d, mask=offs_d < HEAD_DIM)
            v = tl.load(new_values + pid * NUM_HEADS * HEAD_DIM + head_id * HEAD_DIM + offs_d, mask=offs_d < HEAD_DIM)
            k_dst = KV_cache + phys_page * stride_kv_page + tok_in_page * stride_kv_tok + 0 * stride_kv_kv + head_id * stride_kv_h + offs_d * stride_kv_d
            v_dst = KV_cache + phys_page * stride_kv_page + tok_in_page * stride_kv_tok + 1 * stride_kv_kv + head_id * stride_kv_h + offs_d * stride_kv_d
            tl.store(k_dst, k, mask=offs_d < HEAD_DIM)
            tl.store(v_dst, v, mask=offs_d < HEAD_DIM)
            if tok_in_page == 0:
                tl.store(page_valid + phys_page, tl.full([], 1, dtype=tl.int8))
                tl.store(page_positions + phys_page, insert_position + pid)


# =============================================================================
# Kernel 3: RoPE Position Remapping (NEW — Proposal 4)
# =============================================================================
# When probed KV from position P is injected into a window at position W,
# the RoPE frequencies in K encode position P. This kernel re-rotates K
# from original to target positions in-place on GPU.

if HAS_TRITON:
    @triton.jit
    def _kessie_rope_remap_kernel(
        K_data,             # (num_tokens, num_heads, head_dim) — MODIFIED IN PLACE
        orig_positions,     # (num_tokens,) int32
        target_positions,   # (num_tokens,) int32
        rope_theta: tl.constexpr,
        NUM_TOKENS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HALF_DIM: tl.constexpr,
    ):
        """Re-rotate K cache from original to target RoPE positions."""
        pid = tl.program_id(0)
        d = pid % HALF_DIM
        head = (pid // HALF_DIM) % NUM_HEADS
        token = pid // (HALF_DIM * NUM_HEADS)

        if token >= NUM_TOKENS:
            return

        freq = 1.0 / tl.math.pow(rope_theta, tl.cast(d, tl.float32) / tl.cast(HALF_DIM, tl.float32))
        orig_pos = tl.load(orig_positions + token)
        tgt_pos = tl.load(target_positions + token)

        delta = tl.cast(tgt_pos - orig_pos, tl.float32) * freq
        cos_d = tl.cos(delta)
        sin_d = tl.sin(delta)

        HEAD_DIM: tl.constexpr = 2 * HALF_DIM
        base_ptr = K_data + token * NUM_HEADS * HEAD_DIM + head * HEAD_DIM
        k1 = tl.load(base_ptr + d).to(tl.float32)
        k2 = tl.load(base_ptr + HALF_DIM + d).to(tl.float32)

        tl.store(base_ptr + d, (k1 * cos_d - k2 * sin_d).to(K_data.dtype.element_ty))
        tl.store(base_ptr + HALF_DIM + d, (k1 * sin_d + k2 * cos_d).to(K_data.dtype.element_ty))


# =============================================================================
# Kernel 4: Index Query (cosine similarity)
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _kessie_index_query_kernel(
        query_embed, index_embeds, similarities,
        NUM_CHUNKS: tl.constexpr, EMBED_DIM: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        """Cosine similarity between query and all index embeddings."""
        pid = tl.program_id(0)
        offs_d = tl.arange(0, EMBED_DIM)
        q = tl.load(query_embed + offs_d)
        q_norm = tl.sqrt(tl.sum(q * q) + 1e-8)
        q = q / q_norm
        for c in range(BLOCK_C):
            chunk_idx = pid * BLOCK_C + c
            if chunk_idx < NUM_CHUNKS:
                idx = tl.load(index_embeds + chunk_idx * EMBED_DIM + offs_d, mask=offs_d < EMBED_DIM, other=0.0)
                idx_norm = tl.sqrt(tl.sum(idx * idx) + 1e-8)
                sim = tl.sum(q * (idx / idx_norm))
                tl.store(similarities + chunk_idx, sim)


# =============================================================================
# Python Wrappers
# =============================================================================

def kessie_fused_fog_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    fog_weights: torch.Tensor,
    window_size: int = 0, causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Flash attention with fused fog decay on V.

    Args:
        q, k, v: (batch, heads, seq_len, head_dim) FP16
        fog_weights: (seq_len,) float32 — 1.0=clear, <1.0=fogged
        window_size: 0 = full attention
        causal: Apply causal mask
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton required")
    Z, H, N_CTX, HEAD_DIM = q.shape
    assert fog_weights.shape == (N_CTX,)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = ((N_CTX + BLOCK_M - 1) // BLOCK_M, Z, H)
    _kessie_fused_fog_attn_kernel[grid](
        q, k, v, out, fog_weights, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM, WINDOW_SIZE=window_size, IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def kessie_build_fog_weights(
    seq_len: int, fog_alpha: float = 0.5, fog_start: float = 0.5,
    device: torch.device = None,
) -> torch.Tensor:
    """Build fog decay weight vector. 1.0=clear, <1.0=fogged."""
    weights = torch.ones(seq_len, dtype=torch.float32, device=device)
    clear_start = int(seq_len * (1.0 - fog_start))
    if fog_alpha > 0 and clear_start > 0:
        t = torch.arange(clear_start, dtype=torch.float32, device=device) / max(clear_start - 1, 1)
        weights[:clear_start] = t.pow(fog_alpha)
    return weights


def kessie_windowed_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window_size: int = 0, causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """KeSSie windowed flash attention (no fog fusion)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton required")
    Z, H, N_CTX, HEAD_DIM = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = ((N_CTX + BLOCK_M - 1) // BLOCK_M, Z, H)
    _kessie_attn_fwd_kernel[grid](
        q, k, v, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM, WINDOW_SIZE=window_size, IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def kessie_rope_remap(
    k_tensor: torch.Tensor,
    orig_positions: torch.Tensor,
    target_positions: torch.Tensor,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Re-rotate K cache from original RoPE positions to target positions.

    Fixes positional discontinuity when injecting recalled KV from distant
    conversation positions into the active window.

    Args:
        k_tensor: (num_tokens, num_heads, head_dim) — MODIFIED IN PLACE
        orig_positions: (num_tokens,) int32 — original absolute positions
        target_positions: (num_tokens,) int32 — desired virtual positions
        rope_theta: RoPE base frequency (check model config.rope_theta)
    """
    if not HAS_TRITON:
        return _rope_remap_ref(k_tensor, orig_positions, target_positions, rope_theta)

    num_tokens, num_heads, head_dim = k_tensor.shape
    half_dim = head_dim // 2
    total_threads = num_tokens * num_heads * half_dim
    BLOCK = 256
    grid = ((total_threads + BLOCK - 1) // BLOCK,)

    k_tensor = k_tensor.contiguous()
    _kessie_rope_remap_kernel[grid](
        k_tensor,
        orig_positions.contiguous().to(torch.int32),
        target_positions.contiguous().to(torch.int32),
        rope_theta=rope_theta,
        NUM_TOKENS=num_tokens, NUM_HEADS=num_heads, HALF_DIM=half_dim,
    )
    return k_tensor


# =============================================================================
# KeSSiePagedKVCache
# =============================================================================

class KeSSiePagedKVCache:
    """Paged KV cache with KeSSie fog-of-war semantics."""

    def __init__(self, num_pages, page_size, num_layers, num_heads, head_dim,
                 device, dtype=torch.float16):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.kv_data = torch.zeros(
            num_layers, num_pages, page_size, 2, num_heads, head_dim,
            device=device, dtype=dtype)
        self.page_positions = torch.zeros(num_pages, dtype=torch.int32, device=device)
        self.page_valid = torch.zeros(num_pages, dtype=torch.int8, device=device)
        self.free_pages = list(range(num_pages))
        self.total_evictions = 0
        self.total_insertions = 0
        vram_gb = self.kv_data.nelement() * self.kv_data.element_size() / 1e9
        logger.info(f"KeSSiePagedKVCache: {num_pages}p x {page_size}t, "
                     f"{num_layers}L x {num_heads}H x {head_dim}D, VRAM={vram_gb:.2f} GB")

    @property
    def vram_bytes(self):
        return self.kv_data.nelement() * self.kv_data.element_size()

    @property
    def active_pages(self):
        return self.num_pages - len(self.free_pages)

    @property
    def active_tokens(self):
        return self.active_pages * self.page_size

    def allocate_pages(self, n):
        if len(self.free_pages) < n:
            return None
        return torch.tensor([self.free_pages.pop() for _ in range(n)],
                            dtype=torch.int32, device=self.device)

    def evict_before(self, position):
        if HAS_TRITON:
            grid = ((self.num_pages + 255) // 256,)
            _kessie_page_evict_kernel[grid](
                self.kv_data.data_ptr(),
                torch.zeros(1, dtype=torch.int32, device=self.device),
                self.page_positions, self.page_valid, position,
                NUM_PAGES=self.num_pages, PAGE_SIZE=self.page_size)
        evicted = 0
        for i in range(self.num_pages):
            if self.page_valid[i].item() == 0 and i not in self.free_pages:
                self.free_pages.append(i)
                evicted += 1
        self.total_evictions += evicted
        return evicted

    def insert_probe_kv(self, layer_idx, keys, values, conv_position):
        num_tokens = keys.shape[0]
        pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        page_ids = self.allocate_pages(pages_needed)
        if page_ids is None:
            return False
        for p in range(pages_needed):
            pid = page_ids[p].item()
            s, e = p * self.page_size, min((p + 1) * self.page_size, num_tokens)
            self.kv_data[layer_idx, pid, :e-s, 0] = keys[s:e]
            self.kv_data[layer_idx, pid, :e-s, 1] = values[s:e]
            self.page_valid[pid] = 1
            self.page_positions[pid] = conv_position + s
        self.total_insertions += num_tokens
        return True

    def get_layer_kv(self, layer_idx):
        valid = self.page_valid.bool().nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            empty = torch.empty(0, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
            return empty, empty.clone()
        kv = self.kv_data[layer_idx, valid].reshape(-1, 2, self.num_heads, self.head_dim)
        return kv[:, 0], kv[:, 1]

    def get_stats(self):
        return {"total_pages": self.num_pages, "active_pages": self.active_pages,
                "free_pages": len(self.free_pages), "active_tokens": self.active_tokens,
                "vram_gb": self.vram_bytes / 1e9,
                "total_evictions": self.total_evictions, "total_insertions": self.total_insertions}


# =============================================================================
# Reference Implementations (no Triton)
# =============================================================================

def kessie_windowed_attention_ref(q, k, v, window_size=0, causal=True, sm_scale=None):
    Z, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        scores = scores.masked_fill(~torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool)), float("-inf"))
    if window_size > 0:
        pos = torch.arange(N, device=q.device)
        dist = pos[:, None] - pos[None, :]
        scores = scores.masked_fill(~((dist >= 0) & (dist < window_size)), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def kessie_fused_fog_attention_ref(q, k, v, fog_weights, window_size=0, causal=True, sm_scale=None):
    return kessie_windowed_attention_ref(q, k, v * fog_weights[None, None, :, None],
                                         window_size, causal, sm_scale)


def _rope_remap_ref(k_tensor, orig_positions, target_positions, rope_theta=10000.0):
    """Reference RoPE remapping using PyTorch ops."""
    num_tokens, num_heads, head_dim = k_tensor.shape
    half_dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (torch.arange(half_dim, dtype=torch.float32, device=k_tensor.device) / half_dim))
    delta = torch.outer((target_positions - orig_positions).float(), freqs)
    cos_d, sin_d = delta.cos()[:, None, :], delta.sin()[:, None, :]
    k1, k2 = k_tensor[:, :, :half_dim].float(), k_tensor[:, :, half_dim:].float()
    k_tensor[:, :, :half_dim] = (k1 * cos_d - k2 * sin_d).to(k_tensor.dtype)
    k_tensor[:, :, half_dim:] = (k1 * sin_d + k2 * cos_d).to(k_tensor.dtype)
    return k_tensor


# =============================================================================
# Test Suite
# =============================================================================

class KeSSieKernelTests:
    def __init__(self, device="cuda"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def test_windowed_attention_correctness(self):
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Windowed Attention Correctness")
        logger.info("=" * 60)
        configs = [
            (1, 4, 128, 64, 0, True), (1, 4, 128, 64, 64, True),
            (1, 8, 256, 128, 128, True), (2, 8, 512, 128, 256, True),
        ]
        ok = True
        for Z, H, N, D, W, causal in configs:
            q = torch.randn(Z, H, N, D, device=self.device, dtype=torch.float16)
            k = torch.randn(Z, H, N, D, device=self.device, dtype=torch.float16)
            v = torch.randn(Z, H, N, D, device=self.device, dtype=torch.float16)
            ref = kessie_windowed_attention_ref(q.float(), k.float(), v.float(), W, causal).half()
            if HAS_TRITON and self.device.type != "cpu":
                tri = kessie_windowed_attention(q, k, v, W, causal)
                md = (ref - tri).abs().max().item()
                pf = "PASS" if md < 0.05 else "FAIL"
                if pf == "FAIL": ok = False
                logger.info(f"  Z={Z} H={H} N={N} D={D} W={W} | max_diff={md:.6f} [{pf}]")
            else:
                logger.info(f"  Z={Z} H={H} N={N} D={D} W={W} | SKIP")
        return ok

    def test_fused_fog_attention(self):
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Fused Fog+Attention")
        logger.info("=" * 60)
        Z, H, N, D = 1, 8, 512, 128
        q = torch.randn(Z, H, N, D, device=self.device, dtype=torch.float16)
        k = torch.randn(Z, H, N, D, device=self.device, dtype=torch.float16)
        v = torch.randn(Z, H, N, D, device=self.device, dtype=torch.float16)
        fog = kessie_build_fog_weights(N, 0.5, 0.5, self.device)
        ref = kessie_fused_fog_attention_ref(q.float(), k.float(), v.float(), fog).half()
        if HAS_TRITON and self.device.type != "cpu":
            tri = kessie_fused_fog_attention(q, k, v, fog)
            md = (ref - tri).abs().max().item()
            pf = "PASS" if md < 0.1 else "FAIL"
            logger.info(f"  N={N} fog_alpha=0.5 | max_diff={md:.6f} [{pf}]")
            return pf == "PASS"
        logger.info("  SKIP")
        return True

    def test_rope_remap(self):
        logger.info("\n" + "=" * 60)
        logger.info("TEST: RoPE Remapping")
        logger.info("=" * 60)
        L, H, D = 64, 8, 128
        k = torch.randn(L, H, D, device=self.device, dtype=torch.float16)
        k_orig = k.clone()
        orig = torch.arange(100, 100 + L, dtype=torch.int32, device=self.device)
        tgt = torch.arange(5000, 5000 + L, dtype=torch.int32, device=self.device)
        kessie_rope_remap(k, orig, tgt)
        k_back = k.clone()
        kessie_rope_remap(k_back, tgt, orig)
        md = (k_orig.float() - k_back.float()).abs().max().item()
        pf = "PASS" if md < 0.05 else "FAIL"
        logger.info(f"  Round-trip: max_diff={md:.6f} [{pf}]")
        k2 = torch.randn(L, H, D, device=self.device, dtype=torch.float16)
        k2o = k2.clone()
        kessie_rope_remap(k2, orig, orig)
        md2 = (k2o.float() - k2.float()).abs().max().item()
        pf2 = "PASS" if md2 < 1e-5 else "FAIL"
        logger.info(f"  Identity:   max_diff={md2:.6f} [{pf2}]")
        return pf == "PASS" and pf2 == "PASS"

    def test_paged_kv_cache(self):
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Paged KV Cache")
        logger.info("=" * 60)
        cache = KeSSiePagedKVCache(64, 128, 2, 8, 64, self.device)
        keys = torch.randn(512, 8, 64, device=self.device, dtype=torch.float16)
        vals = torch.randn(512, 8, 64, device=self.device, dtype=torch.float16)
        ok = cache.insert_probe_kv(0, keys, vals, 0)
        logger.info(f"  Insert 512 at pos 0: {'OK' if ok else 'FAIL'}, active={cache.active_pages}")
        ev = cache.evict_before(5000)
        logger.info(f"  Evict <5000: {ev} freed, active={cache.active_pages}")
        return True

    def bench_fused_vs_separate(self, seq_len=4096):
        logger.info(f"\nBENCH: Fused vs Separate Fog (N={seq_len})")
        if not HAS_TRITON or self.device.type == "cpu":
            logger.info("  SKIP"); return
        Z, H, D = 1, 8, 128
        q = torch.randn(Z, H, seq_len, D, device=self.device, dtype=torch.float16)
        k = torch.randn(Z, H, seq_len, D, device=self.device, dtype=torch.float16)
        v = torch.randn(Z, H, seq_len, D, device=self.device, dtype=torch.float16)
        fog = kessie_build_fog_weights(seq_len, 0.5, 0.5, self.device)
        for _ in range(3):
            kessie_fused_fog_attention(q, k, v, fog); kessie_windowed_attention(q, k, v)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            kessie_windowed_attention(q, k, v * fog[None, None, :, None])
        torch.cuda.synchronize(); t_sep = (time.perf_counter() - t0) / 20 * 1000
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            kessie_fused_fog_attention(q, k, v, fog)
        torch.cuda.synchronize(); t_fused = (time.perf_counter() - t0) / 20 * 1000
        logger.info(f"  Separate: {t_sep:.2f}ms | Fused: {t_fused:.2f}ms | Speedup: {t_sep/t_fused:.2f}x")

    def run_all(self):
        logger.info("#" * 60)
        logger.info("# KeSSie HIP Kernel Test Suite")
        logger.info(f"# Triton: {'Yes' if HAS_TRITON else 'No'}")
        logger.info("#" * 60)
        self.test_windowed_attention_correctness()
        self.test_fused_fog_attention()
        self.test_rope_remap()
        self.test_paged_kv_cache()
        if HAS_TRITON and self.device.type != "cpu":
            self.bench_fused_vs_separate()
        logger.info("\n# Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KeSSie HIP Kernel Tests")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--test", default="all",
                        choices=["all", "correctness", "fused-fog", "rope-remap",
                                 "bench-fog", "bench-paged"])
    parser.add_argument("--seq-len", type=int, default=4096)
    args = parser.parse_args()
    tests = KeSSieKernelTests(device=args.device)
    if args.test == "all": tests.run_all()
    elif args.test == "correctness": tests.test_windowed_attention_correctness()
    elif args.test == "fused-fog": tests.test_fused_fog_attention()
    elif args.test == "rope-remap": tests.test_rope_remap()
    elif args.test == "bench-fog": tests.bench_fused_vs_separate(args.seq_len)
    elif args.test == "bench-paged": tests.test_paged_kv_cache()
