"""
KeSSie HIP Kernel Build Script
================================

Compiles kessie_attn_kernel.hip and kessie_hip_binding.cpp into
a PyTorch extension loadable via `import kessie_hip`.

Usage:
  # Build the extension
  python kessie_hip_build.py

  # Build self-test binary (standalone, no PyTorch)
  python kessie_hip_build.py --self-test

  # Then use from Python:
  import kessie_hip
  out = kessie_hip.windowed_attention(q, k, v, window_size=4096, causal=True)

Requirements:
  - ROCm 6.1+ with hipcc
  - PyTorch 2.4+ (ROCm build)
  - Target: gfx90a (MI250X)
"""

import os
import sys
import subprocess
import argparse

KESSIE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_self_test():
    """Build standalone self-test binary (no PyTorch dependency)."""
    hip_file = os.path.join(KESSIE_DIR, "kessie_kernels_v2.hip")
    output = os.path.join(KESSIE_DIR, "kessie_self_test")

    cmd = [
        "hipcc",
        "-O3",
        "--offload-arch=gfx90a",
        "-DKESSIE_SELF_TEST",
        "-std=c++17",
        "-o", output,
        hip_file,
    ]

    print(f"Building self-test: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"BUILD FAILED:\n{result.stderr}")
        sys.exit(1)

    print(f"Self-test binary: {output}")
    print(f"Run: {output}")
    return output


def build_pytorch_extension():
    """Build PyTorch C++ extension with HIP kernels."""
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError:
        print("ERROR: PyTorch not found. Install PyTorch for ROCm first.")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/HIP available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    hip_kernel = os.path.join(KESSIE_DIR, "kessie_kernels_v2.hip")
    cpp_binding = os.path.join(KESSIE_DIR, "kessie_hip_binding.cpp")

    # Check files exist
    for f in [hip_kernel, cpp_binding]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found")
            sys.exit(1)

    print("Building KeSSie HIP extension...")
    print(f"  Kernel: {hip_kernel}")
    print(f"  Binding: {cpp_binding}")

    # JIT compile with torch.utils.cpp_extension.load
    # This handles hipcc invocation, linking, and module creation
    kessie_hip = load(
        name="kessie_hip",
        sources=[cpp_binding, hip_kernel],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--offload-arch=gfx90a",
            "-std=c++17",
        ],
        verbose=True,
        build_directory=os.path.join(KESSIE_DIR, "build"),
    )

    print("\nBuild successful!")
    print("Import with: import kessie_hip")
    print("Functions available:")
    print("  kessie_hip.windowed_attention(Q, K, V, window_size, causal)")
    print("  kessie_hip.page_evict(page_valid, page_positions, evict_before_pos)")
    print("  kessie_hip.page_insert(kv_cache, keys, values, page_ids, ...)")

    return kessie_hip


def run_quick_test(kessie_hip):
    """Quick functional test after build."""
    import torch

    print("\n=== Quick Functional Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("SKIP: No GPU available")
        return

    B, H, N, D = 1, 8, 256, 128
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # Test full causal attention
    out_full = kessie_hip.windowed_attention(q, k, v, 0, True)
    print(f"  Full causal:    shape={list(out_full.shape)}, "
          f"NaN={out_full.isnan().sum().item()}, "
          f"mean={out_full.abs().mean().item():.6f}")

    # Test windowed attention
    out_win = kessie_hip.windowed_attention(q, k, v, 128, True)
    print(f"  Window=128:     shape={list(out_win.shape)}, "
          f"NaN={out_win.isnan().sum().item()}, "
          f"mean={out_win.abs().mean().item():.6f}")

    # Test page eviction
    num_pages = 64
    page_valid = torch.ones(num_pages, dtype=torch.int8, device=device)
    page_positions = torch.arange(0, num_pages * 128, 128,
                                   dtype=torch.int32, device=device)
    evict_count = kessie_hip.page_evict(page_valid, page_positions, 4096)
    evicted = evict_count.item()
    print(f"  Page evict:     evicted={evicted} pages (pos < 4096)")

    # Test page insertion
    kv_cache = torch.zeros(num_pages, 128, 2, H, D, device=device, dtype=torch.float16)
    new_k = torch.randn(128, H, D, device=device, dtype=torch.float16)
    new_v = torch.randn(128, H, D, device=device, dtype=torch.float16)
    target_pages = torch.tensor([0], dtype=torch.int32, device=device)
    kessie_hip.page_insert(kv_cache, new_k, new_v, target_pages,
                            page_valid, page_positions, 50000)
    print(f"  Page insert:    128 tokens at conv pos 50000 [OK]")

    print("\nAll quick tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KeSSie HIP Kernels")
    parser.add_argument("--self-test", action="store_true",
                        help="Build standalone self-test binary only")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip functional test after build")
    args = parser.parse_args()

    if args.self_test:
        build_self_test()
    else:
        ext = build_pytorch_extension()
        if not args.no_test:
            run_quick_test(ext)
