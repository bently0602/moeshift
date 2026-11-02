#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-metal/ggml-metal-impl.h"
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) throw std::runtime_error("Metal is required");
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0})
    for (int rows : {1, 3}) for (int mask = 0; mask < 4; ++mask) {
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        auto * weights = ggml_new_tensor_3d(ctx, type, 256, 4, 2);
        auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, rows);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, rows);
        auto * plain_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, rows);
        auto * plain = ggml_mul_mat_id(ctx, weights, input, plain_ids);
        auto * resident = ggml_mul_mat_id(ctx, weights, input, ids);
        auto * missing = ggml_mul_mat_id(ctx, weights, input, ids);
        resident->op_params[GGML_MOE_PHASE_PARAM] = GGML_MOE_PHASE_RESIDENT;
        missing->op_params[GGML_MOE_PHASE_PARAM] = GGML_MOE_PHASE_MISSING;
        auto * sum = ggml_add(ctx, resident, missing);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum);
        ggml_build_forward_expand(graph, plain);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> w(256*4*2), x(256*rows), result(8*rows), ready(8*rows), late(8*rows), reference(8*rows);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i) * 0.1f);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * 0.2f);
        std::vector<uint8_t> quantized(ggml_nbytes(weights));
        ggml_quantize_chunk(type, w.data(), quantized.data(), 0, 8, 256, nullptr);
        std::vector<int32_t> slots(2*rows);
        for (int r=0; r<rows; ++r) for (int e=0; e<2; ++e)
            slots[2*r+e] = mask & (1<<e) ? -e-1 : e;
        std::vector<int32_t> unsigned_slots(2*rows);
        for (int i = 0; i < 2*rows; ++i) unsigned_slots[i] = i%2;
        ggml_backend_tensor_set(plain_ids, unsigned_slots.data(), 0, ggml_nbytes(plain_ids));
        ggml_backend_tensor_set(weights, quantized.data(), 0, ggml_nbytes(weights));
        ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(ids, slots.data(), 0, ggml_nbytes(ids));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Metal phase computation failed");
        ggml_backend_tensor_get(sum, result.data(), 0, ggml_nbytes(sum));
        ggml_backend_tensor_get(resident, ready.data(), 0, ggml_nbytes(resident));
        ggml_backend_tensor_get(missing, late.data(), 0, ggml_nbytes(missing));
        ggml_backend_tensor_get(plain, reference.data(), 0, ggml_nbytes(plain));
        for (int i=0; i<8*rows; ++i) {
            const float expected = reference[i];
            const bool pending = slots[i/4]<0;
            if (!(std::abs(result[i]-expected)<=1e-4f) ||
                !(std::abs(ready[i]-(pending?0:expected))<=1e-4f) ||
                !(std::abs(late[i]-(pending?expected:0))<=1e-4f))
                throw std::runtime_error("resident/missing phase selection changed the expert result");
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
