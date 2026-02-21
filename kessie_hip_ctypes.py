"""
KeSSie HIP Kernels - ctypes Python Interface
==============================================
PROPRIETARY & CONFIDENTIAL

Loads compiled libkessie_hip.so via ctypes and provides Python-callable
functions that accept PyTorch tensors.

Functions:
  windowed_attention    — Flash attention with sliding window
  fused_fog_attention   — Flash attention with fused fog decay on V (NEW)
  rope_remap            — Re-rotate K cache positions for probe injection (NEW)
  page_evict            — Fog pages below conversation position
  page_insert           — Un-fog: insert probe KV into free pages

Async support:
  All operations accept an optional `stream` parameter. When provided,
  the kernel runs on that stream without synchronization, enabling
  overlap with decode. Use `create_stream()` and `sync_stream()`.

Build:
  make lib   (or: hipcc -O3 --offload-arch=gfx90a -shared -fPIC
                   -o build/libkessie_hip.so kessie_attn_kernel.hip)

Usage:
  from kessie_hip_ctypes import KeSSieKernels
  kernels = KeSSieKernels()
  output = kernels.windowed_attention(q, k, v, window_size=4096, causal=True)
  output = kernels.fused_fog_attention(q, k, v, fog_weights, causal=True)
  kernels.rope_remap(k_data, orig_pos, tgt_pos, rope_theta=10000.0)
"""

import ctypes
import os
import math
import torch
from typing import Optional


class KeSSieKernels:
    """Python interface to KeSSie HIP kernels via ctypes."""

    def __init__(self, lib_path: str = None):
        if lib_path is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), "build", "libkessie_hip.so"),
                os.path.join(os.path.dirname(__file__), "libkessie_hip.so"),
                "/usr/local/lib/libkessie_hip.so",
            ]
            for p in candidates:
                if os.path.exists(p):
                    lib_path = p
                    break

        if lib_path is None or not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libkessie_hip.so not found. Build with: make lib\n"
                f"Searched: {candidates if lib_path is None else [lib_path]}"
            )

        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self._streams = {}
        print(f"KeSSie HIP kernels loaded from {lib_path}")

    def _setup_signatures(self):
        """Define C function signatures for ctypes."""

        # windowed_attention
        self.lib.launch_kessie_windowed_attn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.launch_kessie_windowed_attn.restype = None

        # fused_fog_attention
        self.lib.launch_kessie_fused_fog_attn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,  # fog_weights
            ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.launch_kessie_fused_fog_attn.restype = None

        # page_evict
        self.lib.launch_kessie_page_evict.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.launch_kessie_page_evict.restype = None

        # page_insert
        self.lib.launch_kessie_page_insert.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.launch_kessie_page_insert.restype = None

        # rope_remap
        self.lib.launch_kessie_rope_remap.argtypes = [
            ctypes.c_void_p,  # k_data
            ctypes.c_void_p,  # orig_positions
            ctypes.c_void_p,  # target_positions
            ctypes.c_float,   # rope_theta
            ctypes.c_int,     # num_tokens
            ctypes.c_int,     # num_heads
            ctypes.c_int,     # half_dim
            ctypes.c_void_p,  # stream
        ]
        self.lib.launch_kessie_rope_remap.restype = None

    # =========================================================================
    # Stream management for async operations (Proposal 2)
    # =========================================================================

    def create_stream(self, name: str = "probe") -> torch.cuda.Stream:
        """Create a named CUDA stream for async kernel launches.

        Usage:
            stream = kernels.create_stream("probe")
            kernels.rope_remap(k, orig, tgt, stream=stream)  # non-blocking
            # ... do other work ...
            kernels.sync_stream("probe")  # wait for completion
        """
        s = torch.cuda.Stream()
        self._streams[name] = s
        return s

    def sync_stream(self, name: str = "probe"):
        """Synchronize a named stream (wait for all its kernels)."""
        if name in self._streams:
            self._streams[name].synchronize()

    def _get_stream_ptr(self, stream=None) -> ctypes.c_void_p:
        """Get HIP stream pointer. None = sync on default stream."""
        if stream is None:
            return ctypes.c_void_p(0)
        if isinstance(stream, str):
            stream = self._streams.get(stream)
        if isinstance(stream, torch.cuda.Stream):
            return ctypes.c_void_p(stream.cuda_stream)
        return ctypes.c_void_p(0)

    # =========================================================================
    # Tensor helpers
    # =========================================================================

    @staticmethod
    def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
        """Get raw data pointer from a PyTorch CUDA tensor."""
        assert tensor.is_cuda, f"Tensor must be on GPU, got {tensor.device}"
        assert tensor.is_contiguous(), "Tensor must be contiguous"
        return ctypes.c_void_p(tensor.data_ptr())

    # =========================================================================
    # Attention kernels
    # =========================================================================

    def windowed_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        window_size: int = 0,
        causal: bool = True,
        sm_scale: Optional[float] = None,
        stream=None,
    ) -> torch.Tensor:
        """KeSSie windowed flash attention.

        Args:
            Q, K, V: [B, H, N, D] FP16 on GPU
            window_size: 0 = full attention
            stream: Optional CUDA stream for async execution
        """
        assert Q.dtype == torch.float16
        assert Q.shape == K.shape == V.shape
        B, H, N, D = Q.shape
        assert D in (64, 128)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        O = torch.empty_like(Q)
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()

        self.lib.launch_kessie_windowed_attn(
            self._ptr(Q), self._ptr(K), self._ptr(V), self._ptr(O),
            ctypes.c_float(sm_scale),
            B, H, N, D, window_size, 1 if causal else 0,
            self._get_stream_ptr(stream),
        )

        if stream is None:
            torch.cuda.synchronize()
        return O

    def fused_fog_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        fog_weights: torch.Tensor,
        window_size: int = 0,
        causal: bool = True,
        sm_scale: Optional[float] = None,
        stream=None,
    ) -> torch.Tensor:
        """Flash attention with fused fog decay on V.

        V is NOT modified — fog weights are applied in-register during
        the attention reduction. Saves one full V read+write per layer.

        Args:
            Q, K, V: [B, H, N, D] FP16 on GPU
            fog_weights: [N] FP32 — 1.0=clear, <1.0=fogged
            stream: Optional CUDA stream for async execution
        """
        assert Q.dtype == torch.float16
        assert fog_weights.dtype == torch.float32
        assert Q.shape == K.shape == V.shape
        B, H, N, D = Q.shape
        assert fog_weights.shape[0] == N
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        O = torch.empty_like(Q)
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        fog_weights = fog_weights.contiguous()

        self.lib.launch_kessie_fused_fog_attn(
            self._ptr(Q), self._ptr(K), self._ptr(V), self._ptr(O),
            self._ptr(fog_weights),
            ctypes.c_float(sm_scale),
            B, H, N, D, window_size, 1 if causal else 0,
            self._get_stream_ptr(stream),
        )

        if stream is None:
            torch.cuda.synchronize()
        return O

    # =========================================================================
    # RoPE remapping (Proposal 4)
    # =========================================================================

    def rope_remap(
        self,
        k_data: torch.Tensor,
        orig_positions: torch.Tensor,
        target_positions: torch.Tensor,
        rope_theta: float = 10000.0,
        stream=None,
    ):
        """Re-rotate K cache from original to target RoPE positions.

        Fixes positional discontinuity when injecting probed KV from
        distant positions into the active window.

        Args:
            k_data: [tokens, heads, dim] — MODIFIED IN PLACE
            orig_positions: [tokens] INT32
            target_positions: [tokens] INT32
            rope_theta: Model's RoPE base (check config.rope_theta)
            stream: Optional CUDA stream for async execution
        """
        assert k_data.is_cuda and k_data.is_contiguous()
        num_tokens, num_heads, head_dim = k_data.shape
        assert head_dim % 2 == 0
        half_dim = head_dim // 2

        orig_positions = orig_positions.contiguous().to(torch.int32)
        target_positions = target_positions.contiguous().to(torch.int32)

        self.lib.launch_kessie_rope_remap(
            self._ptr(k_data),
            self._ptr(orig_positions),
            self._ptr(target_positions),
            ctypes.c_float(rope_theta),
            num_tokens, num_heads, half_dim,
            self._get_stream_ptr(stream),
        )

        if stream is None:
            torch.cuda.synchronize()

    # =========================================================================
    # Page management
    # =========================================================================

    def page_evict(
        self,
        page_valid: torch.Tensor,
        page_positions: torch.Tensor,
        evict_before_pos: int,
        stream=None,
    ) -> int:
        """Evict (fog) all KV pages with position below threshold."""
        assert page_valid.dtype == torch.int8
        assert page_positions.dtype == torch.int32
        num_pages = page_valid.shape[0]
        evict_count = torch.zeros(1, dtype=torch.int32, device=page_valid.device)

        self.lib.launch_kessie_page_evict(
            self._ptr(page_valid), self._ptr(page_positions),
            self._ptr(evict_count),
            evict_before_pos, num_pages,
            self._get_stream_ptr(stream),
        )

        if stream is None:
            torch.cuda.synchronize()
        return evict_count.item()

    def page_insert(
        self,
        kv_cache: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        target_page_ids: torch.Tensor,
        page_valid: torch.Tensor,
        page_positions: torch.Tensor,
        insert_conv_position: int,
        stream=None,
    ):
        """Insert probe KV into allocated pages (un-fog operation)."""
        assert kv_cache.dtype == torch.float16
        assert new_keys.dtype == torch.float16
        num_tokens = new_keys.shape[0]
        num_heads = new_keys.shape[1]
        head_dim = new_keys.shape[2]
        page_size = kv_cache.shape[1]
        max_pages = kv_cache.shape[0]

        self.lib.launch_kessie_page_insert(
            self._ptr(kv_cache), self._ptr(new_keys), self._ptr(new_values),
            self._ptr(target_page_ids), self._ptr(page_valid),
            self._ptr(page_positions),
            insert_conv_position,
            num_tokens, num_heads, head_dim, page_size, max_pages,
            self._get_stream_ptr(stream),
        )

        if stream is None:
            torch.cuda.synchronize()


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("KeSSie HIP Kernels - ctypes interface test")

    try:
        k = KeSSieKernels()
    except FileNotFoundError as e:
        print(f"Cannot test: {e}")
        print("Build first: make lib")
        exit(1)

    if not torch.cuda.is_available():
        print("No GPU available, skipping test")
        exit(0)

    device = torch.device("cuda")
    B, H, N, D = 1, 8, 512, 128

    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    kk = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # Full causal
    out = k.windowed_attention(q, kk, v, window_size=0, causal=True)
    print(f"Full causal:      {out.shape}, NaN={out.isnan().sum().item()}, mean={out.abs().mean():.4f}")

    # Windowed
    out_w = k.windowed_attention(q, kk, v, window_size=128, causal=True)
    print(f"Window=128:       {out_w.shape}, NaN={out_w.isnan().sum().item()}, mean={out_w.abs().mean():.4f}")

    # Fused fog
    from kessie_hip_kernels import kessie_build_fog_weights
    fog = kessie_build_fog_weights(N, fog_alpha=0.5, fog_start=0.5, device=device)
    out_fog = k.fused_fog_attention(q, kk, v, fog, window_size=0, causal=True)
    print(f"Fused fog:        {out_fog.shape}, NaN={out_fog.isnan().sum().item()}, mean={out_fog.abs().mean():.4f}")

    # RoPE remap
    k_data = torch.randn(64, H, D, device=device, dtype=torch.float16)
    orig_pos = torch.arange(100, 164, dtype=torch.int32, device=device)
    tgt_pos = torch.arange(5000, 5064, dtype=torch.int32, device=device)
    k.rope_remap(k_data, orig_pos, tgt_pos)
    print(f"RoPE remap:       {k_data.shape}, NaN={k_data.isnan().sum().item()}")

    # Async stream test
    stream = k.create_stream("probe")
    k.rope_remap(k_data, tgt_pos, orig_pos, stream="probe")
    print("Async remap dispatched...")
    k.sync_stream("probe")
    print(f"Async remap done: NaN={k_data.isnan().sum().item()}")

    print("PASS")
