#pragma once
#include "llama-model.h"
#include "llama-moe-protected-cache.h"
#include <cstdlib>
#include <map>

// Shared prefill banks and per-layer decode windows, grouped by family and quantization.
class llama_moe_tensor_loader {
public:
    llama_moe_tensor_loader(llama_model_base & model, llama_model_loader & ml);
    ggml_tensor * create(const LLM_TN_IMPL & name, const std::initializer_list<int64_t> & dims, int flags);
    void create_gate_up(llama_layer & layer, int il, int64_t n_ff_exp, int flags);
    void finish();
private:
    llama_model_base & model;
    llama_model_loader & ml;
    const LLM_TN tn;
    int64_t dec_c, k_pf, prefill_cap;
    bool dec_enabled;
    std::vector<int32_t> layer_c;
    std::map<std::pair<ggml_type, size_t>, std::vector<int>> groups[4];
    std::map<std::string, int64_t> group_slots;
    int64_t width_of(int il) const {
        if (!dec_enabled || il < 0 || il >= (int) layer_c.size()) return 0;
        const char * unified = std::getenv("LLAMA_MOE_PREFETCH_UNIFIED");
        const bool protected_cache = llm_arch_supports_les(model.arch) &&
            std::getenv("LLAMA_LES_DECODE_BANK") && layer_c[il] == 12 && k_pf == 3 &&
            !(unified && *unified == '0');
        const int maximum = int(std::min<int64_t>({prefill_cap, model.hparams.n_expert, 1024})) - 15;
        return layer_c[il] + k_pf + (protected_cache ? les_protected_slots(maximum) : 0);
    }
    int64_t slots_for(const LLM_TN_IMPL & name) const {
        const auto it = group_slots.find(name.str());
        return it == group_slots.end() ? prefill_cap : it->second;
    }
};
