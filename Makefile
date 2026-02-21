# KeSSie HIP Kernels - Makefile
# ===============================
# PROPRIETARY & CONFIDENTIAL
#
# Targets:
#   make all      - Build self-test + shared lib
#   make test     - Build and run self-test binary
#   make lib      - Build shared library (for ctypes / Python linking)
#   make pytorch  - Build PyTorch extension via JIT
#   make triton   - Run Triton kernel tests (fused fog, RoPE remap, etc.)
#   make bench    - Build self-test and run benchmarks
#   make clean    - Remove build artifacts
#   make info     - Print build environment
#
# Kernels (kessie_kernels_v2.hip):
#   1. kessie_windowed_attn_mfma  — Flash attention, MFMA-accelerated
#   2. kessie_fused_fog_attn_mfma — Flash attention + fused fog decay on V (NEW)
#   3. kessie_page_evict          — Fog pages below conversation position
#   4. kessie_page_insert         — Un-fog: copy probe KV into free pages
#   5. kessie_rope_remap          — RoPE position re-rotation for probed KV (NEW)
#
# Requirements:
#   - ROCm 6.1+ with hipcc on PATH
#   - Target: gfx90a (MI250X). Change GPU_ARCH for other targets.
#   - For PyTorch extension: PyTorch 2.4+ (ROCm build)
#   - For Triton tests: triton 3.x

GPU_ARCH    ?= gfx90a
HIPCC       ?= hipcc
CXXFLAGS    := -O3 --offload-arch=$(GPU_ARCH) -std=c++17
LDFLAGS     := -lhiprt

BUILD_DIR   := build
SRC_KERNEL  := kessie_kernels_v2.hip
SRC_BIND    := kessie_hip_binding.cpp

.PHONY: all test lib clean bench pytorch triton info

all: test lib

# ---------------------------------------------------------------------------
# HIP C++ targets (native MFMA kernels)
# ---------------------------------------------------------------------------

# Self-test binary
$(BUILD_DIR)/kessie_self_test: $(SRC_KERNEL) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -DKESSIE_SELF_TEST -o $@ $<
	@echo "Built: $@"

test: $(BUILD_DIR)/kessie_self_test
	@echo "\n=== Running KeSSie Self-Test ==="
	./$<

# Shared library (for ctypes / dynamic linking)
$(BUILD_DIR)/libkessie_hip.so: $(SRC_KERNEL) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -shared -fPIC -o $@ $<
	@echo "Built: $@"
	@echo "  Exports: launch_kessie_windowed_attn"
	@echo "           launch_kessie_fused_fog_attn"
	@echo "           launch_kessie_rope_remap"
	@echo "           launch_kessie_page_evict"
	@echo "           launch_kessie_page_insert"

lib: $(BUILD_DIR)/libkessie_hip.so

# Benchmark
bench: $(BUILD_DIR)/kessie_self_test
	@echo "\n=== Running KeSSie Benchmarks ==="
	./$<

# ---------------------------------------------------------------------------
# PyTorch extension (JIT-compiled via torch.utils.cpp_extension)
# ---------------------------------------------------------------------------

pytorch:
	python kessie_hip_build.py

# ---------------------------------------------------------------------------
# Triton kernel tests (Python — tests fused fog, RoPE remap, paged cache)
# ---------------------------------------------------------------------------

triton:
	@echo "\n=== Running Triton Kernel Tests ==="
	python kessie_hip_kernels.py --test all

triton-fog:
	python kessie_hip_kernels.py --test fused-fog

triton-rope:
	python kessie_hip_kernels.py --test rope-remap

triton-bench:
	python kessie_hip_kernels.py --test bench-fog --seq-len 8192

# ---------------------------------------------------------------------------
# ctypes quick test (needs lib target first)
# ---------------------------------------------------------------------------

ctypes-test: lib
	python kessie_hip_ctypes.py

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
	rm -f kessie_self_test
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

info:
	@echo "GPU_ARCH  = $(GPU_ARCH)"
	@echo "HIPCC     = $(HIPCC)"
	@echo "CXXFLAGS  = $(CXXFLAGS)"
	@echo "SRC       = $(SRC_KERNEL)"
	@echo ""
	@echo "--- Exported functions ---"
	@echo "  launch_kessie_windowed_attn     (original flash attention)"
	@echo "  launch_kessie_fused_fog_attn    (NEW: fog decay fused into attention)"
	@echo "  launch_kessie_rope_remap        (NEW: RoPE position re-rotation)"
	@echo "  launch_kessie_page_evict        (fog page eviction)"
	@echo "  launch_kessie_page_insert       (probe KV page insertion)"
	@echo ""
	@which hipcc 2>/dev/null && hipcc --version || echo "hipcc not found"
	@rocminfo 2>/dev/null | head -20 || echo "rocminfo not available"
