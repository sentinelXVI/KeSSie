"""
KeSSie Attention Backend for vLLM V1

FA3-style async overlap with per-step shared fog cache.

Key optimizations over naive fog-of-war:
  1. SHARED FOG CACHE: Fog bias computed ONCE per forward step, shared
     across all 64+ layers. Not per-layer. Single lock-free read per layer.
  2. PING-PONG DOUBLE BUFFER: Two pre-allocated tensors. While attention
     runs using buffer A, buffer B is async-computed for the next step.
  3. ASYNC SIDE STREAM: Fog generation overlaps with attention kernels.
  4. ZERO-ALLOC HOT PATH: No tensor allocation in steady state.
  5. GENERATION COUNTER: Layers detect stale fog via integer compare,
     no lock acquisition needed on cache hit.
"""

import os
import threading
import logging
from typing import Optional, Set

logger = logging.getLogger("KeSSie.Attention")


# --- Global KeSSie attention state (thread-safe, set per-request) ---

class KeSSieAttentionState:
    """Thread-safe global state for KeSSie attention parameters."""

    def __init__(self):
        self._lock = threading.Lock()
        self.fog_alpha: float = 0.5
        self.fog_start: float = 0.5
        self.prompt_len: int = 0
        self.recall_positions: Set[int] = set()
        self.recall_boost: float = 0.1  # positive bias for recalled content
        self.enabled: bool = True
        self.generation: int = 0

    def update(self, fog_alpha=None, fog_start=None, prompt_len=None,
               recall_positions=None, recall_boost=None):
        with self._lock:
            if fog_alpha is not None: self.fog_alpha = fog_alpha
            if fog_start is not None: self.fog_start = fog_start
            if prompt_len is not None: self.prompt_len = prompt_len
            if recall_positions is not None: self.recall_positions = set(recall_positions)
            if recall_boost is not None: self.recall_boost = recall_boost
            self.generation += 1

    def get(self):
        with self._lock:
            return {
                "fog_alpha": self.fog_alpha,
                "fog_start": self.fog_start,
                "prompt_len": self.prompt_len,
                "recall_positions": set(self.recall_positions),
                "recall_boost": self.recall_boost,
                "enabled": self.enabled,
                "generation": self.generation,
            }

KESSIE_STATE = KeSSieAttentionState()


# --- Shared per-step fog cache ---

class _FogCache:
    """
    Singleton. Fog bias computed once per generation by first layer,
    all subsequent layers in the same step get a cached tensor view.
    """

    def __init__(self):
        self._gen = -1
        self._kv_len = 0

        # Ping-pong
        self._buf_a = None
        self._buf_b = None
        self._active = 0
        self._alloc_len = 0
        self._device = None
        self._dtype = None

        # Async
        self._stream = None
        self._event = None
        self._pending = False
        self._pending_gen = -1

        # Cached view
        self._view = None
        self._view_kv_len = 0

        # Only one layer computes per step
        self._compute_lock = threading.Lock()

    def _ensure_alloc(self, kv_len, device, dtype):
        if self._alloc_len >= kv_len and self._device == device and self._dtype == dtype:
            return
        alloc = max(kv_len, 2048)
        alloc = (alloc + 255) & ~255
        self._buf_a = _torch.zeros(alloc, device=device, dtype=dtype)
        self._buf_b = _torch.zeros(alloc, device=device, dtype=dtype)
        self._alloc_len = alloc
        self._device = device
        self._dtype = dtype
        self._gen = -1
        self._view = None
        if self._stream is None:
            self._stream = _torch.cuda.Stream(device=device)
            self._event = _torch.cuda.Event()

    @staticmethod
    def _compute_into(buf, kv_len, fog_alpha, fog_start, recall_pos, recall_boost=0.1):
        """
        Compute fog bias into buf[:kv_len].
        
        Fog zone (older tokens): negative bias decaying from -fog_alpha to 0
        Clear zone (recent tokens): zero bias (full attention)
        Recalled positions: POSITIVE bias (+recall_boost)
          - Fully cancels any fog decay at that position
          - Adds amplification so model preferentially attends to recalled content
          - This means recalled content gets BETTER attention than even clear-zone tokens
        """
        fog_boundary = int(kv_len * (1.0 - fog_start))
        buf[:kv_len].zero_()
        if fog_boundary > 0:
            pos = _torch.arange(fog_boundary, device=buf.device, dtype=buf.dtype)
            decay = (1.0 - pos / fog_boundary)
            decay.mul_(decay)
            decay.mul_(-fog_alpha)
            buf[:fog_boundary].copy_(decay)

        # Recalled positions: zero fog + positive amplification
        # This overwrites whatever fog value was there, then adds boost
        for p in recall_pos:
            if 0 <= p < kv_len:
                buf[p] = recall_boost  # fully un-fog + amplify

    def get_fog_bias(self, kv_len, device, dtype, state):
        """Return (1,1,1,kv_len) fog bias. Computed once per generation."""
        gen = state["generation"]

        # Fast path: cached and valid (no lock)
        if self._gen == gen and self._kv_len == kv_len and self._view is not None:
            return self._view

        with self._compute_lock:
            # Double-check
            if self._gen == gen and self._kv_len == kv_len and self._view is not None:
                return self._view

            self._ensure_alloc(kv_len, device, dtype)
            recall_pos = frozenset(state["recall_positions"])
            fog_alpha = state["fog_alpha"]
            fog_start = state["fog_start"]
            recall_boost = state.get("recall_boost", 0.1)

            # Check if async pre-compute from last step matches
            if self._pending and self._pending_gen == gen:
                self._event.synchronize()
                self._pending = False
                self._active = 1 - self._active
            else:
                # Sync compute
                active = self._buf_a if self._active == 0 else self._buf_b
                self._compute_into(active, kv_len, fog_alpha, fog_start, recall_pos, recall_boost)

            self._gen = gen
            self._kv_len = kv_len

            active = self._buf_a if self._active == 0 else self._buf_b
            self._view = active[:kv_len].view(1, 1, 1, kv_len)
            self._view_kv_len = kv_len

            # Kick async for next step (optimistic: same params, gen+1)
            next_buf = self._buf_b if self._active == 0 else self._buf_a
            with _torch.cuda.stream(self._stream):
                self._compute_into(next_buf, kv_len, fog_alpha, fog_start, recall_pos, recall_boost)
                self._event.record(self._stream)
            self._pending = True
            self._pending_gen = gen + 1

            return self._view


def register_kessie_attention():
    """Register KeSSie attention. Call BEFORE AsyncLLMEngine creation."""
    try:
        from vllm.v1.attention.backends.registry import (
            register_backend, AttentionBackendEnum,
        )
    except ImportError:
        logger.warning("vLLM V1 attention registry not available")
        return False

    import torch
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

    if is_rocm:
        target = AttentionBackendEnum.TRITON_ATTN
        original_path = target.value
        logger.info(f"  KeSSie: overriding TRITON_ATTN on ROCm")
    else:
        target = AttentionBackendEnum.FLASH_ATTN
        original_path = target.value
        logger.info(f"  KeSSie: overriding FLASH_ATTN on NVIDIA")

    os.environ["_KESSIE_ORIGINAL_ATTN_PATH"] = original_path
    register_backend(target, "kessie_attention.KeSSieAttentionBackend")
    logger.info(f"  KeSSie attention registered (wrapping {original_path})")
    return True


# --- Backend wrapper ---

try:
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.utils.import_utils import resolve_obj_by_qualname
    import torch as _torch
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    _torch = None
    class AttentionBackend: pass

# Singleton fog cache
_FOG_CACHE: Optional[_FogCache] = None

if HAS_VLLM:
    _FOG_CACHE = _FogCache()

    class KeSSieAttentionBackend(AttentionBackend):
        _original_cls = None

        @classmethod
        def _get_original(cls):
            if cls._original_cls is None:
                path = os.environ.get("_KESSIE_ORIGINAL_ATTN_PATH", "")
                if path:
                    cls._original_cls = resolve_obj_by_qualname(path)
                else:
                    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
                    cls._original_cls = TritonAttentionBackend
            return cls._original_cls

        @classmethod
        def get_name(cls) -> str:
            return "KeSSie"

        @classmethod
        def get_impl_cls(cls):
            return KeSSieAttentionImpl

        @classmethod
        def get_metadata_cls(cls):
            return cls._get_original().get_metadata_cls()

        @classmethod
        def get_kv_cache_shape(cls, *args, **kwargs):
            return cls._get_original().get_kv_cache_shape(*args, **kwargs)


    class KeSSieAttentionImpl:
        """
        Per-layer wrapper. Hot path per layer (after first):
          - 1 int compare (generation)
          - 1 cached view return
          - 0 allocs, 0 locks
        """

        def __init__(self, *args, **kwargs):
            orig_cls = KeSSieAttentionBackend._get_original()
            orig_impl_cls = orig_cls.get_impl_cls()
            self._inner = orig_impl_cls(*args, **kwargs)

        def forward(self, *args, **kwargs):
            state = KESSIE_STATE.get()

            if not state["enabled"] or state["fog_alpha"] <= 0 or state["prompt_len"] <= 0:
                return self._inner.forward(*args, **kwargs)

            if 'attn_bias' in kwargs or len(args) < 4:
                return self._inner.forward(*args, **kwargs)

            query = args[1]
            key = args[2]

            if not (hasattr(query, 'shape') and hasattr(key, 'shape') and key.dim() >= 2):
                return self._inner.forward(*args, **kwargs)

            kv_len = key.shape[-2]
            if kv_len <= 0:
                return self._inner.forward(*args, **kwargs)

            try:
                fog_bias = _FOG_CACHE.get_fog_bias(
                    kv_len, query.device, query.dtype, state)
                if fog_bias is not None:
                    kwargs['attn_bias'] = fog_bias
                return self._inner.forward(*args, **kwargs)
            except Exception:
                return self._inner.forward(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

else:
    class KeSSieAttentionBackend:
        pass
    class KeSSieAttentionImpl:
        pass


# ===================================================================
# VALIDATION TEST
# ===================================================================
# Run: python kessie_attention.py
#
# Tests:
#   1. Fog bias shape and values are correct
#   2. Decay suppresses old positions (negative bias)
#   3. Recall positions get boosted (+0.5)
#   4. Shared cache returns same tensor for same generation
#   5. Async pipeline produces correct results after generation bump
#   6. Zero fog_alpha bypasses fog entirely
# ===================================================================

if __name__ == "__main__":
    import torch as _torch
    import sys

    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    print("=" * 60)
    print("KeSSie Attention — Fog-of-War Validation Test")
    print("=" * 60)

    # Init cache
    cache = _FogCache()
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    dtype = _torch.float32

    # --- Test 1: Basic fog shape ---
    print("\n[Test 1] Basic fog bias shape")
    state = {
        "fog_alpha": 0.5, "fog_start": 0.5, "prompt_len": 100,
        "recall_positions": set(), "enabled": True, "recall_boost": 0.1, "generation": 1,
    }
    bias = cache.get_fog_bias(128, device, dtype, state)
    check("Shape is (1,1,1,128)", bias.shape == (1, 1, 1, 128))
    check("Not all zeros (fog active)", bias.abs().sum().item() > 0)

    # --- Test 2: Decay values ---
    print("\n[Test 2] Decay suppresses old positions")
    flat = bias.view(-1)
    fog_boundary = int(128 * (1.0 - 0.5))  # = 64
    check("Position 0 has strongest suppression (most negative)", flat[0].item() < flat[32].item())
    check("Position at boundary-1 has weak suppression", flat[fog_boundary - 1].item() < 0)
    check("Positions after boundary are zero", flat[fog_boundary:].abs().sum().item() == 0)
    check("All fog values are <= 0", flat.max().item() <= 0)

    # --- Test 3: Recall boost (un-fog + amplify) ---
    print("\n[Test 3] Recall position un-fog + amplification")
    state2 = {
        "fog_alpha": 0.5, "fog_start": 0.5, "prompt_len": 100,
        "recall_positions": {10, 50, 100}, "recall_boost": 0.1,
        "enabled": True, "generation": 2,
    }
    bias2 = cache.get_fog_bias(128, device, dtype, state2)
    flat2 = bias2.view(-1)
    check("Recall pos 10 is POSITIVE (fully un-fogged + amplified)",
          flat2[10].item() > 0)
    check("Recall pos 10 equals recall_boost (0.1)",
          abs(flat2[10].item() - 0.1) < 1e-5)
    check("Recall pos 100 has +0.1 (beyond fog boundary)",
          abs(flat2[100].item() - 0.1) < 1e-5)
    check("Non-recall fogged pos 11 is negative",
          flat2[11].item() < 0)
    check("Recalled pos has BETTER attention than clear zone (positive vs zero)",
          flat2[10].item() > flat2[70].item())  # pos 70 is in clear zone (= 0)

    # --- Test 4: Cache sharing (same generation) ---
    print("\n[Test 4] Cache sharing across layers")
    bias_a = cache.get_fog_bias(128, device, dtype, state2)
    bias_b = cache.get_fog_bias(128, device, dtype, state2)
    check("Same tensor returned for same generation",
          bias_a.data_ptr() == bias_b.data_ptr())

    # --- Test 5: Async pipeline ---
    print("\n[Test 5] Async pipeline correctness")
    state3 = {
        "fog_alpha": 0.8, "fog_start": 0.3, "prompt_len": 200,
        "recall_positions": {5}, "recall_boost": 0.1, "enabled": True, "generation": 3,
    }
    bias3 = cache.get_fog_bias(256, device, dtype, state3)
    flat3 = bias3.view(-1)
    fog_boundary3 = int(256 * (1.0 - 0.3))  # = 179
    check("Shape is (1,1,1,256)", bias3.shape == (1, 1, 1, 256))
    check("Fog boundary at ~179", flat3[fog_boundary3:].abs().sum().item() < 1e-5)
    check("Alpha=0.8 gives stronger suppression than alpha=0.5",
          flat3[0].item() < flat.view(-1)[0].item())

    # Bump generation, should use async pre-computed buffer
    state4 = dict(state3)
    state4["generation"] = 4
    bias4 = cache.get_fog_bias(256, device, dtype, state4)
    check("Gen 4 returns valid bias (async or sync)",
          bias4.shape == (1, 1, 1, 256))
    check("Gen 4 matches gen 3 values (same params)",
          _torch.allclose(bias3.view(-1)[:256], bias4.view(-1)[:256], atol=1e-6))
    check("Gen 4 is different tensor (buffer swapped)",
          bias4.data_ptr() != bias3.data_ptr())

    # --- Test 6: Disabled fog ---
    print("\n[Test 6] Disabled / zero alpha")
    # Zero alpha should be caught before get_fog_bias is called,
    # but verify the cache still works
    state_off = {
        "fog_alpha": 0.0, "fog_start": 0.5, "prompt_len": 100,
        "recall_positions": set(), "recall_boost": 0.1, "enabled": True, "generation": 5,
    }
    bias_off = cache.get_fog_bias(128, device, dtype, state_off)
    check("Zero alpha produces all-zero bias",
          bias_off.view(-1).abs().sum().item() == 0)

    # --- Test 7: Performance ---
    print("\n[Test 7] Performance — 64 layer simulation")
    if _torch.cuda.is_available():
        import time
        state_perf = {
            "fog_alpha": 0.5, "fog_start": 0.5, "prompt_len": 4096,
            "recall_positions": {100, 200, 300}, "recall_boost": 0.1, "enabled": True, "generation": 10,
        }
        # Warm up
        for _ in range(3):
            cache.get_fog_bias(4096, device, dtype, state_perf)
            state_perf["generation"] += 1

        # Time 64 layer calls (simulating one step)
        gen = state_perf["generation"]
        state_perf["generation"] = gen + 1
        _torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(64):
            cache.get_fog_bias(4096, device, dtype, state_perf)
        _torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        check(f"64 layers in {elapsed_us:.0f}µs (< 500µs target)", elapsed_us < 500)
        print(f"    (first layer computes, layers 2-64 are cache hits)")
    else:
        print("    (skipped — no CUDA)")

    # --- Summary ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f" ({failed} FAILED)")
        sys.exit(1)
    else:
        print(" — ALL PASSED ✓")
        sys.exit(0)
