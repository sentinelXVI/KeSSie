/*
 * KeSSie HIP Kernels - PyTorch Extension Binding
 * ================================================
 * PROPRIETARY & CONFIDENTIAL
 *
 * Provides torch-callable functions for KeSSie's three HIP kernels.
 *
 * Build:
 *   cd /home/claude && python kessie_hip_build.py
 *
 * Usage from Python:
 *   import kessie_hip
 *   output = kessie_hip.windowed_attention(q, k, v, window_size=4096, causal=True)
 *   kessie_hip.page_evict(page_valid, page_positions, evict_before_pos)
 *   kessie_hip.page_insert(kv_cache, keys, values, page_ids, page_valid,
 *                          page_positions, conv_position)
 */

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// External launcher declarations (defined in kessie_attn_kernel.hip)
extern "C" void launch_kessie_windowed_attn(
    const void* Q, const void* K, const void* V, void* O,
    float sm_scale,
    int batch_size, int num_heads, int seq_len, int head_dim,
    int window_size, int is_causal,
    hipStream_t stream
);

extern "C" void launch_kessie_fused_fog_attn(
    const void* Q, const void* K, const void* V, void* O,
    const void* fog_weights,
    float sm_scale,
    int batch_size, int num_heads, int seq_len, int head_dim,
    int window_size, int is_causal,
    hipStream_t stream
);

extern "C" void launch_kessie_page_evict(
    void* page_valid, const void* page_positions, void* evict_count,
    int evict_before_pos, int num_pages,
    hipStream_t stream
);

extern "C" void launch_kessie_page_insert(
    void* kv_cache, const void* new_keys, const void* new_values,
    const void* target_page_ids, void* page_valid, void* page_positions,
    int insert_conv_position,
    int num_tokens, int num_heads, int head_dim, int page_size, int max_pages,
    hipStream_t stream
);

extern "C" void launch_kessie_rope_remap(
    void* k_data,
    const void* orig_positions, const void* target_positions,
    float rope_theta,
    int num_tokens, int num_heads, int half_dim,
    hipStream_t stream
);


// =============================================================================
// PyTorch Wrappers
// =============================================================================

torch::Tensor windowed_attention(
    torch::Tensor Q,         // [B, H, N, D] FP16
    torch::Tensor K,         // [B, H, N, D] FP16
    torch::Tensor V,         // [B, H, N, D] FP16
    int window_size,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on GPU");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "Q must be FP16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K shape must match Q");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V shape must match Q");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128");

    auto O = torch::empty_like(Q);
    float sm_scale = 1.0f / sqrtf((float)D);

    // Get current HIP stream from PyTorch
    hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

    launch_kessie_windowed_attn(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        sm_scale, B, H, N, D, window_size, causal ? 1 : 0, stream
    );

    return O;
}


torch::Tensor page_evict(
    torch::Tensor page_valid,      // [num_pages] int8
    torch::Tensor page_positions,  // [num_pages] int32
    int evict_before_pos
) {
    TORCH_CHECK(page_valid.is_cuda(), "page_valid must be on GPU");
    TORCH_CHECK(page_valid.dtype() == torch::kInt8, "page_valid must be int8");

    int num_pages = page_valid.size(0);
    auto evict_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(page_valid.device()));

    hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

    launch_kessie_page_evict(
        page_valid.data_ptr(), page_positions.data_ptr(), evict_count.data_ptr(),
        evict_before_pos, num_pages, stream
    );

    return evict_count;
}


void page_insert(
    torch::Tensor kv_cache,         // [pages, page_size, 2, heads, dim] FP16
    torch::Tensor new_keys,         // [tokens, heads, dim] FP16
    torch::Tensor new_values,       // [tokens, heads, dim] FP16
    torch::Tensor target_page_ids,  // [pages_needed] int32
    torch::Tensor page_valid,       // [max_pages] int8
    torch::Tensor page_positions,   // [max_pages] int32
    int insert_conv_position
) {
    TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on GPU");
    TORCH_CHECK(kv_cache.dtype() == torch::kHalf, "kv_cache must be FP16");

    int num_tokens = new_keys.size(0);
    int num_heads  = new_keys.size(1);
    int head_dim   = new_keys.size(2);
    int page_size  = kv_cache.size(1);
    int max_pages  = kv_cache.size(0);

    hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

    launch_kessie_page_insert(
        kv_cache.data_ptr(), new_keys.data_ptr(), new_values.data_ptr(),
        target_page_ids.data_ptr(), page_valid.data_ptr(), page_positions.data_ptr(),
        insert_conv_position,
        num_tokens, num_heads, head_dim, page_size, max_pages, stream
    );
}


torch::Tensor fused_fog_attention(
    torch::Tensor Q,            // [B, H, N, D] FP16
    torch::Tensor K,            // [B, H, N, D] FP16
    torch::Tensor V,            // [B, H, N, D] FP16
    torch::Tensor fog_weights,  // [N] FP32
    int window_size,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on GPU");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "Q must be FP16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(fog_weights.dtype() == torch::kFloat, "fog_weights must be FP32");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    auto O = torch::empty_like(Q);
    float sm_scale = 1.0f / sqrtf((float)D);

    hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

    launch_kessie_fused_fog_attn(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        fog_weights.data_ptr(),
        sm_scale, B, H, N, D, window_size, causal ? 1 : 0, stream
    );

    return O;
}


void rope_remap(
    torch::Tensor k_data,           // [tokens, heads, dim] FP16 — modified in place
    torch::Tensor orig_positions,    // [tokens] INT32
    torch::Tensor target_positions,  // [tokens] INT32
    float rope_theta
) {
    TORCH_CHECK(k_data.is_cuda(), "k_data must be on GPU");
    TORCH_CHECK(k_data.is_contiguous(), "k_data must be contiguous");
    TORCH_CHECK(orig_positions.dtype() == torch::kInt, "positions must be INT32");

    int num_tokens = k_data.size(0);
    int num_heads  = k_data.size(1);
    int head_dim   = k_data.size(2);
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    int half_dim = head_dim / 2;

    hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

    launch_kessie_rope_remap(
        k_data.data_ptr(),
        orig_positions.data_ptr(), target_positions.data_ptr(),
        rope_theta, num_tokens, num_heads, half_dim, stream
    );
}


// =============================================================================
// Module Registration
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KeSSie HIP Kernels — Attention, Fog, RoPE Remap, Page Management";
    m.def("windowed_attention", &windowed_attention,
          "KeSSie windowed flash attention (FP16, causal, sliding window)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("window_size") = 0, py::arg("causal") = true);
    m.def("fused_fog_attention", &fused_fog_attention,
          "KeSSie fused fog+attention (FP16, fog weights applied to V inline)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("fog_weights"), py::arg("window_size") = 0, py::arg("causal") = true);
    m.def("rope_remap", &rope_remap,
          "Re-rotate K cache from original to target RoPE positions (in-place)",
          py::arg("k_data"), py::arg("orig_positions"),
          py::arg("target_positions"), py::arg("rope_theta") = 10000.0f);
    m.def("page_evict", &page_evict,
          "Evict KV pages below conversation position (fog operation)",
          py::arg("page_valid"), py::arg("page_positions"),
          py::arg("evict_before_pos"));
    m.def("page_insert", &page_insert,
          "Insert probe KV into free pages (un-fog operation)",
          py::arg("kv_cache"), py::arg("new_keys"), py::arg("new_values"),
          py::arg("target_page_ids"), py::arg("page_valid"),
          py::arg("page_positions"), py::arg("insert_conv_position"));
}
