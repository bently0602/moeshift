// Decode attention with grouped query heads (GGML_METAL_FA_GROUP_HEADS) must give the same bits as
// the regular kernel: each head repeats the regular kernel's arithmetic over the same cache chunks.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static float sample(uint32_t & seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return float(int32_t(seed >> 8) - 8388608) / 4194304.0f;
}

static void fill(ggml_tensor * t, uint32_t & seed) {
    std::vector<float> src(ggml_nelements(t));
    for (auto & v : src) { v = sample(seed); }
    std::vector<uint8_t> dst(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(dst.data(), src.data(), dst.size());
    } else {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) dst.data(), ggml_nelements(t));
    }
    ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) {
        std::printf("SKIP no Metal backend\n");
        return 0;
    }
    struct shape { int heads, kv_heads, kv, nt; ggml_type type; };
    std::vector<shape> shapes;
    for (ggml_type type : {GGML_TYPE_F16}) {
        for (int kv : {100, 1000, 3000, 6144, 42389}) {
            for (int nt : {1, 2}) {
                shapes.push_back({16, 2, kv, nt, type}); // Qwen3.6-35B-A3B: 8 query heads per KV head
                shapes.push_back({24, 2, kv, nt, type}); // 12 query heads per KV head
            }
        }
    }
    const int d = 256;
    for (const shape & sh : shapes) {
        std::vector<float> results[2];
        double times[2] = {};
        for (int grouped = 0; grouped < 2; ++grouped) {
            if (grouped) {
                setenv("GGML_METAL_FA_GROUP_HEADS", "1", 1);
            } else {
                unsetenv("GGML_METAL_FA_GROUP_HEADS");
            }
            auto ctx = ggml_init({16*1024*1024, nullptr, true});
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, sh.nt, sh.heads, 1);
            auto k = ggml_new_tensor_4d(ctx, sh.type, d, sh.kv, sh.kv_heads, 1);
            auto v = ggml_new_tensor_4d(ctx, sh.type, d, sh.kv, sh.kv_heads, 1);
            auto m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.kv, sh.nt, 1, 1);
            auto out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt(float(d)), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            uint32_t seed = 11;
            fill(q, seed);
            fill(k, seed);
            fill(v, seed);
            // a causal-looking mask with scattered masked cells and one fully masked 32-cell chunk
            std::vector<ggml_fp16_t> vm(ggml_nelements(m));
            for (int t = 0; t < sh.nt; ++t) {
                for (int c = 0; c < sh.kv; ++c) {
                    const bool masked = c > sh.kv - sh.nt + t || c % 97 == 5 || (c >= 64 && c < 96);
                    vm[t*sh.kv + c] = ggml_fp32_to_fp16(masked ? -INFINITY : 0.0f);
                }
            }
            ggml_backend_tensor_set(m, vm.data(), 0, ggml_nbytes(m));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            results[grouped].resize(ggml_nelements(out));
            ggml_backend_tensor_get(out, results[grouped].data(), 0, ggml_nbytes(out));
            const int iterations = 50;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            }
            ggml_backend_synchronize(backend);
            times[grouped] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        unsetenv("GGML_METAL_FA_GROUP_HEADS");
        if (std::memcmp(results[0].data(), results[1].data(), results[0].size()*sizeof(float)) != 0) {
            for (size_t i = 0; i < results[0].size(); ++i) {
                if (std::memcmp(&results[0][i], &results[1][i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "FAIL %s heads=%d/%d kv=%d tokens=%d i=%zu regular=%.9g grouped=%.9g\n",
                        ggml_type_name(sh.type), sh.heads, sh.kv_heads, sh.kv, sh.nt, i, results[0][i], results[1][i]);
                    break;
                }
            }
            return 3;
        }
        std::printf("PASS exact %s heads=%d/%d kv=%d tokens=%d regular_us=%.1f grouped_us=%.1f\n",
            ggml_type_name(sh.type), sh.heads, sh.kv_heads, sh.kv, sh.nt, times[0], times[1]);
    }
    ggml_backend_free(backend);
    return 0;
}
