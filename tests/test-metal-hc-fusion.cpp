#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static float sample(uint32_t & seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return float(int32_t(seed >> 8) - 8388608) / 4194304.0f;
}

// The end of build_hc_mix: SIGMOID, MUL, the ADD chain over the stream views and SCALE.
// Exposing gated as an output keeps Metal from fusing, which gives the separate kernels' result.
static int check_hc_mix_tail(ggml_backend_t backend) {
    const int streams = 4;
    struct shape { int width, nt; };
    for (shape sh : {shape{2560, 1}, shape{2560, 7}, shape{2560, 1536}, shape{40, 7}, shape{37, 3}}) {
        std::vector<float> results[2];
        double times[2] = {};
        for (int separate = 0; separate < 2; ++separate) {
            auto ctx = ggml_init({4*1024*1024, nullptr, true});
            auto xn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.width*streams, sh.nt);
            auto g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.width*streams, sh.nt);
            auto gated = ggml_mul(ctx, xn, ggml_sigmoid(ctx, g));
            auto g3 = ggml_reshape_3d(ctx, gated, sh.width, streams, sh.nt);
            auto stream = [&](int c) {
                return ggml_view_2d(ctx, g3, sh.width, sh.nt, ggml_row_size(GGML_TYPE_F32, sh.width)*streams,
                    ggml_row_size(GGML_TYPE_F32, sh.width)*c);
            };
            auto mixed = ggml_add(ctx, stream(0), stream(1));
            for (int c = 2; c < streams; ++c) {
                mixed = ggml_add(ctx, mixed, stream(c));
            }
            mixed = ggml_scale(ctx, mixed, 1.0f/streams);
            if (separate) {
                ggml_set_output(gated);
            }
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, mixed);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            uint32_t seed = 7;
            std::vector<float> vx(ggml_nelements(xn)), vg(ggml_nelements(g));
            for (auto & v : vx) { v = sample(seed); }
            for (auto & v : vg) { v = 4.0f*sample(seed); }
            ggml_backend_tensor_set(xn, vx.data(), 0, ggml_nbytes(xn));
            ggml_backend_tensor_set(g, vg.data(), 0, ggml_nbytes(g));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            results[separate].resize(ggml_nelements(mixed));
            ggml_backend_tensor_get(mixed, results[separate].data(), 0, ggml_nbytes(mixed));
            const int iterations = sh.nt == 1536 ? 20 : 100;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            }
            times[separate] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        if (std::memcmp(results[0].data(), results[1].data(), results[0].size()*sizeof(float)) != 0) {
            for (size_t i = 0; i < results[0].size(); ++i) {
                if (std::memcmp(&results[0][i], &results[1][i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "FAIL hc mix tail width=%d tokens=%d i=%zu fused=%.9g separate=%.9g\n",
                        sh.width, sh.nt, i, results[0][i], results[1][i]);
                    break;
                }
            }
            return 3;
        }
        std::printf("PASS exact hc mix tail width=%d tokens=%d mean_us=%.2f separate_us=%.2f\n",
            sh.width, sh.nt, times[0], times[1]);
    }
    return 0;
}

// A one-token gated delta net and the copy of its new state into a cache row, as in
// build_recurrent_attn. Exposing the state view as an output keeps Metal from writing the state
// straight into the cache, which gives the separate copy's result.
static int check_gdn_state_copy(ggml_backend_t backend) {
    const int S = 128, H = 48, H_k = 16;
    std::vector<float> results[2][2];
    for (int separate = 0; separate < 2; ++separate) {
        auto ctx = ggml_init({4*1024*1024, nullptr, true});
        auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, 1, 1);
        auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, 1, 1);
        auto v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
        auto g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, 1);
        auto b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, 1);
        auto s0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S*S*H, 1, 1);
        auto cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H, 2);
        auto result = ggml_gated_delta_net(ctx, q, k, v, g, b, s0);
        auto output = ggml_view_4d(ctx, result, S, H, 1, 1, ggml_row_size(GGML_TYPE_F32, S),
            ggml_row_size(GGML_TYPE_F32, S*H), ggml_row_size(GGML_TYPE_F32, S*H), 0);
        auto state = ggml_view_4d(ctx, result, S, S, H, 1, ggml_row_size(GGML_TYPE_F32, S),
            ggml_row_size(GGML_TYPE_F32, S*S), ggml_row_size(GGML_TYPE_F32, S*S*H), ggml_row_size(GGML_TYPE_F32, S*H));
        if (separate) {
            ggml_set_output(state);
        }
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, state, ggml_view_2d(ctx, cache, S*S*H, 1, cache->nb[1], cache->nb[1])));
        auto out = ggml_scale(ctx, output, 1.0f);
        ggml_build_forward_expand(graph, out);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        uint32_t seed = 11;
        for (auto * t : {q, k, v, g, b, s0}) {
            std::vector<float> data(ggml_nelements(t));
            for (auto & x : data) { x = 0.25f*sample(seed); }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
        std::vector<float> zero(ggml_nelements(cache), 0.0f);
        ggml_backend_tensor_set(cache, zero.data(), 0, ggml_nbytes(cache));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
        results[separate][0].resize(ggml_nelements(cache));
        results[separate][1].resize(ggml_nelements(out));
        ggml_backend_tensor_get(cache, results[separate][0].data(), 0, ggml_nbytes(cache));
        ggml_backend_tensor_get(out, results[separate][1].data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    for (int r = 0; r < 2; ++r) {
        if (std::memcmp(results[0][r].data(), results[1][r].data(), results[0][r].size()*sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL gated delta net %s differs from the separate copy\n", r ? "output" : "state");
            return 3;
        }
    }
    bool written = false;
    for (size_t i = S*S*H; i < results[0][0].size() && !written; ++i) written = results[0][0][i] != 0.0f;
    if (!written) {
        std::fprintf(stderr, "FAIL gated delta net state was not written to the cache row\n");
        return 3;
    }
    std::printf("PASS exact gated delta net state written to the cache\n");
    return 0;
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) {
        std::fprintf(stderr, "FAIL: this test requires Metal\n");
        return 1;
    }
    if (const int rc = check_hc_mix_tail(backend)) {
        return rc;
    }
    if (const int rc = check_gdn_state_copy(backend)) {
        return rc;
    }
    for (int nt : {1, 7, 1536}) {
        for (bool expose_intermediate : {false, true}) {
            const int width = nt == 7 ? 37 : 2560;
            const int streams = 4;
            auto ctx = ggml_init({4*1024*1024, nullptr, true});
            auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, nt);
            auto w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, streams, nt);
            auto r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, streams, nt);
            auto repeated = ggml_repeat(ctx, x, r);
            auto product = ggml_mul(ctx, repeated, w);
            auto out = ggml_add(ctx, r, product);
            if (expose_intermediate) {
                ggml_set_output(product);
            }
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            uint32_t seed = 42;
            std::vector<float> vx(ggml_nelements(x)), vw(ggml_nelements(w)), vr(ggml_nelements(r));
            for (auto & v : vx) { v = sample(seed); }
            for (auto & v : vw) { v = sample(seed); }
            for (auto & v : vr) { v = sample(seed); }
            ggml_backend_tensor_set(x, vx.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(w, vw.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(r, vr.data(), 0, ggml_nbytes(r));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            std::vector<float> got(vr.size()), mid(expose_intermediate ? vr.size() : 0);
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            if (expose_intermediate) { ggml_backend_tensor_get(product, mid.data(), 0, ggml_nbytes(product)); }
            for (size_t i = 0; i < got.size(); ++i) {
                const size_t row = i / width;
                volatile float mul = vx[(row / streams)*width + i % width] * vw[row];
                const float expected = vr[i] + mul;
                if (std::memcmp(&expected, &got[i], sizeof(float)) != 0 ||
                        (expose_intermediate && mid[i] != mul)) {
                    std::fprintf(stderr, "FAIL width=%d tokens=%d exposed=%d i=%zu got=%.9g expected=%.9g\n",
                        width, nt, expose_intermediate, i, got[i], expected);
                    return 3;
                }
            }
            const int iterations = nt == 1536 ? 20 : 100;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
            std::printf("PASS exact width=%d tokens=%d exposed=%d mean_us=%.2f\n", width, nt, expose_intermediate, us);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
