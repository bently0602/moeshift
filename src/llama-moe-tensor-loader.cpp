#include "llama-moe-tensor-loader.h"
#include "llama-model-loader.h"
#include "llama-moe-experts.h"
#include "llama-impl.h"
#include <algorithm>

llama_moe_tensor_loader::llama_moe_tensor_loader(llama_model_base & model, llama_model_loader & ml) : model(model), ml(ml), tn(model.arch) {
    dec_c = model.moe_decode_window();
    // 每层解码视图 = 缓存区 c + 预读暂存区 K。分区见 llama-moe-slot.cpp 的
    // install_prefetch_into_window：预读只在暂存区分配，不会淘汰缓存区条目。
    //
    // **按层宽度**：各层的「容量→miss」曲线差别大（layer 0 要 c≈200 才饱和，多数层
    // c≈96 就饱和），均匀分配会让冷层浪费 slot。逐层宽度由 LLAMA_MOE_LAYER_WINDOWS
    // 指定，见 llama_model::moe_layer_windows()。
    k_pf = model.moe_prefetch_slots();
    layer_c = model.moe_layer_windows();   // 下标 = MoE 层序号
    dec_enabled = dec_c > 0 && !layer_c.empty();
    // 某个 MoE 层序号对应的总宽度（c_il + K）
    prefill_cap = model.moe_slot_bank_size();

    const llm_tensor fam_ids[4] = {
        LLM_TENSOR_FFN_GATE_UP_EXPS, LLM_TENSOR_FFN_GATE_EXPS,
        LLM_TENSOR_FFN_UP_EXPS,      LLM_TENSOR_FFN_DOWN_EXPS,
    };
    // family -> (bytes_per_expert -> 层号升序)
    for (int il = 0; il < (int) model.hparams.n_layer; ++il) {
        for (int fi = 0; fi < 4; ++fi) {
            const auto * e = model.moe_experts_entry(tn(fam_ids[fi], "weight", il).str().c_str());
            if (e) {
                groups[fi][{e->type, e->bytes_per_expert}].push_back(il);
            }
        }
    }
    // Qwen's mixed quantizations can share transient prefill storage. Decode
    // windows use byte offsets, so different expert strides never overlap.
    if (llm_arch_supports_les(model.arch) && !std::getenv("LLAMA_QWEN_SEPARATE_PREFILL_BANKS")) {
        for (int fi = 0; fi < 4; ++fi) {
            std::vector<int> members;
            for (const auto & group : groups[fi]) {
                members.insert(members.end(), group.second.begin(), group.second.end());
            }
            if (members.empty()) continue;
            std::sort(members.begin(), members.end());
            const auto * first_entry = model.moe_experts_entry(tn(fam_ids[fi], "weight", members.front()).str().c_str());
            const auto group_key = std::make_pair(first_entry->type, first_entry->bytes_per_expert);
            groups[fi].clear();
            groups[fi][group_key] = std::move(members);
        }
    }
    // 组长（真正分配内存的那一层）-> 分配的 slot 数。非组长按 prefill 容量建，
    // 稍后统一改写成 base 上的 view。
    for (int fi = 0; fi < 4; ++fi) {
        for (const auto & kv : groups[fi]) {
            // 组内各层的宽度可能不同（按层分配），所以要**逐个累加**，不能用
            // n_members * 单层宽度。
            size_t max_stride = kv.first.second;
            for (int il : kv.second) {
                max_stride = std::max(max_stride, model.moe_experts_entry(tn(fam_ids[fi], "weight", il).str().c_str())->bytes_per_expert);
            }
            size_t bytes = prefill_cap * max_stride;
            for (int il : kv.second) {
                const auto * e = model.moe_experts_entry(tn(fam_ids[fi], "weight", il).str().c_str());
                bytes += width_of(il) * e->bytes_per_expert;
            }
            if (llm_arch_supports_les(model.arch) && dec_enabled) {
                size_t stride_sum = 0;
                for (int il : kv.second)
                    stride_sum += model.moe_experts_entry(tn(fam_ids[fi], "weight", il).str().c_str())->bytes_per_expert;
                const size_t required = model.hparams.n_expert_used + k_pf;
                bytes = std::max(bytes, required * stride_sum);
            }
            group_slots[tn(fam_ids[fi], "weight", kv.second.front()).str()] =
                (bytes + kv.first.second - 1) / kv.first.second;
        }
    }
}

ggml_tensor * llama_moe_tensor_loader::create(const LLM_TN_IMPL & tensor_name,
                                    const std::initializer_list<int64_t> & dims,
                                    int flags) {
        // 专家张量的内容来自 experts（运行时 preadv 写入），跟模型文件无关：
        //  - 必须避开 mmap 的文件映射包装（否则启用 mmap 时落进只读映射，写不进去）
        //  - gguf 里也可能根本没有它（用 strip_experts.py 剥离过），此时按 manifest 合成
        //
        // 这里一律先按「非必需」创建，避免 create_tensor 在 gguf 缺项时直接抛；
        // 找不到来源时再按调用方传的 flags 决定是否报错。
        const bool required = !(flags & llama_model_loader::TENSOR_NOT_REQUIRED);
        ggml_tensor * tensor = model.create_tensor_own_alloc(
            ml, tensor_name, dims, flags | llama_model_loader::TENSOR_NOT_REQUIRED);

        const auto * entry = model.moe_experts_entry(tensor_name.str().c_str());

        if (!tensor && entry) {
            tensor = model.create_expert_tensor_from_manifest(ml, tensor_name, dims, entry->type);
        }
        if (!tensor) {
            if (required && model.moe_slot_bank_enabled()) {
                throw std::runtime_error(format(
                    "routed tensor '%s' is in neither the GGUF nor the MoE experts",
                    tensor_name.str().c_str()));
            }
            return tensor;
        }
        if (!model.moe_slot_bank_enabled()) {
            return tensor;
        }
        if (!entry) {
            throw std::runtime_error(format(
                "MoE experts is missing routed tensor '%s'",
                tensor_name.str().c_str()));
        }
        // 核对「dims x 类型」的字节数 == experts 记录的 每专家字节 x 专家数。
        // 模型的 expert_count 或量化档位与 experts 不一致时，这里立刻暴露，
        // 而不是等到运行时读到错位字节（那种错是静默的）。
        if (ggml_nbytes(tensor) != entry->bytes_per_expert * (size_t) model.hparams.n_expert) {
            throw std::runtime_error(format(
                "expert tensor '%s' byte size mismatch: tensor=%zu experts=%zu",
                tensor_name.str().c_str(), ggml_nbytes(tensor),
                entry->bytes_per_expert * (size_t) model.hparams.n_expert));
        }

        ml.virtualize_expert_tensor(
            tensor,
            model.hparams.n_expert,
            slots_for(tensor_name),
            entry->type,
            entry->bytes_per_expert);
        return tensor;
    }

void llama_moe_tensor_loader::create_gate_up(llama_layer & layer, int il, int64_t n_ff_exp, int flags) {
        layer.ffn_gate_up_exps = create(
            tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", il),
            { model.hparams.n_embd, n_ff_exp * 2, model.hparams.n_expert },
            llama_model_loader::TENSOR_NOT_REQUIRED | flags);
        if (!layer.ffn_gate_up_exps) {
            layer.ffn_gate_exps = create(
                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il),
                { model.hparams.n_embd, n_ff_exp, model.hparams.n_expert },
                flags);
            layer.ffn_up_exps = create(
                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il),
                { model.hparams.n_embd, n_ff_exp, model.hparams.n_expert },
                flags);
        }
    }

void llama_moe_tensor_loader::finish() {
    // 每组一个物理 base，组内后层是它的 view（RAM = base，而不是每组一个 base × 组内层数）。
    //
    // 名字很关键：运行时 `bind()` 按张量名查 experts，所以 prefill 视图必须沿用
    // 原始 GGUF 名；decode 视图加 `.dec` 后缀，由 `model.moe_experts_entry()` 剥掉再查。
    if (model.moe_slot_bank_enabled()) {
        struct fam_t {
            ggml_tensor * llama_layer::* pf;
            ggml_tensor * llama_layer::* pf_dec;
        };
        const fam_t fams[4] = {
            { &llama_layer::ffn_gate_up_exps, &llama_layer::ffn_gate_up_exps_dec },
            { &llama_layer::ffn_gate_exps,    &llama_layer::ffn_gate_exps_dec    },
            { &llama_layer::ffn_up_exps,      &llama_layer::ffn_up_exps_dec      },
            { &llama_layer::ffn_down_exps,    &llama_layer::ffn_down_exps_dec    },
        };

        int overlay_width = 0;
        if (llm_arch_supports_les(model.arch) && dec_enabled &&
            !std::getenv("LLAMA_LES_DECODE_BANK") &&
            !std::getenv("LLAMA_QWEN_SEPARATE_PREFILL_BANKS")) {
            overlay_width = 512;
            for (int fi = 0; fi < 4; ++fi) for (const auto & kv : groups[fi]) {
                size_t stride_sum = 0;
                for (int il : kv.second) stride_sum += (model.layers[il].*fams[fi].pf)->nb[2];
                const auto * base = model.layers[kv.second.front()].*fams[fi].pf;
                GGML_ASSERT(stride_sum > 0);
                overlay_width = std::min(overlay_width, int(ggml_nbytes(base) / stride_sum));
            }
            for (int il = 0; il < int(model.hparams.n_layer); ++il)
                GGML_ASSERT(overlay_width >= width_of(il));
            LLAMA_LOG_INFO("moeshift: phase-overlay: %d slots/layer in unchanged allocations\n", overlay_width);
        }
        size_t resident = 0;
        int n_win = 0;
        int n_grp = 0;

        for (int fi = 0; fi < 4; ++fi) {
            for (const auto & kv : groups[fi]) {
                const std::vector<int> & members = kv.second;
                if (members.empty()) {
                    continue;
                }
                const int first = members.front();
                ggml_tensor * base = model.layers[first].*fams[fi].pf;
                if (!base) {
                    continue;
                }
                const auto * e = model.moe_experts_entry(ggml_get_name(base));
                if (!e || e->bytes_per_expert == 0) {
                    throw std::runtime_error(format(
                        "MoE base '%s' has no usable experts entry", ggml_get_name(base)));
                }
                if (e->type != kv.first.first || e->bytes_per_expert != kv.first.second) {
                    throw std::runtime_error(format(
                        "MoE group for '%s' has inconsistent expert bytes", ggml_get_name(base)));
                }
                resident += ggml_nbytes(base);
                n_grp++;
                if (llm_arch_supports_les(model.arch) && !std::getenv("LLAMA_QWEN_SEPARATE_PREFILL_BANKS")) {
                    size_t max_stride = e->bytes_per_expert;
                    for (int il : members) max_stride = std::max(max_stride, (model.layers[il].*fams[fi].pf)->nb[2]);
                    size_t decode_offset = overlay_width ? 0 : prefill_cap * max_stride;
                    const std::string base_name = ggml_get_name(base);
                    ggml_set_name(base, (base_name + ".base").c_str());
                    for (int il : members) {
                        auto & layer = model.layers[il];
                        auto * original = layer.*fams[fi].pf;
                        const std::string name = il == first ? base_name : ggml_get_name(original);
                        const auto * entry = model.moe_experts_entry(name.c_str());
                        if (original != base) ml.alias_expert_slot_tensor(original, base);
                        layer.*fams[fi].pf = ml.typed_expert_slot_view(
                            name.c_str(), base, original, 0, prefill_cap);
                        if (dec_enabled) {
                            layer.*fams[fi].pf_dec = ml.typed_expert_slot_view(
                                (name + ".dec").c_str(), base, original, decode_offset, overlay_width ? overlay_width : width_of(il));
                            decode_offset += (overlay_width ? overlay_width : width_of(il)) * entry->bytes_per_expert;
                        }
                        n_win++;
                    }
                    continue;
                }


                // 组内非组长的张量先转成 base 的 view —— 不这么做它们会真的分配内存，
                // RAM 会变成「每组 base + 组内层数 × prefill 容量」。
                for (size_t j = 1; j < members.size(); ++j) {
                    ggml_tensor * t = model.layers[members[j]].*fams[fi].pf;
                    if (t) {
                        ml.alias_expert_slot_tensor(t, base);
                    }
                }

                // 组长的张量就是 base 本身（ne[2] = 组的容量），必须换成 view 才能进图：
                // 否则 base 的 ne[2] 可能 > 1024，Metal 的 mul_mat_id 会断言崩溃。
                for (size_t j = 0; j < members.size(); ++j) {
                    auto & L = model.layers[members[j]];
                    if (j == 0) {
                        const std::string orig = ggml_get_name(base);
                        ggml_set_name(base, (orig + ".base").c_str());
                        L.*fams[fi].pf = ml.window_expert_slot_tensor(
                            orig.c_str(), base, 0, prefill_cap, e->bytes_per_expert);
                    }
                    n_win++;
                    if (dec_enabled) {
                        const std::string n2 = format("%s.dec", ggml_get_name(L.*fams[fi].pf));
                        // 偏移 = prefill 段之后 + 组内**前面各层**宽度之和
                        // （各层宽度可能不同）。prefill 段固定占 [0, prefill_cap)，
                        // 与 decode 段物理隔离，见上面布局说明。
                        int64_t off = prefill_cap;
                        for (size_t k2 = 0; k2 < j; ++k2) {
                            off += width_of(members[k2]);
                        }
                        const int64_t wj = width_of(members[j]);
                        L.*fams[fi].pf_dec = ml.window_expert_slot_tensor(
                            n2.c_str(), base, off, wj, e->bytes_per_expert);
                    }
                }
            }
        }

        LLAMA_LOG_INFO(
            "%s: MoE bank %d groups (family x quant), %.2f MiB resident; %d prefill windows%s\n",
            __func__, n_grp, resident / 1024.0 / 1024.0, n_win,
            dec_enabled ? format(", %d decode windows (c=%d, 按层宽度 %s)",
                                 n_win, (int) dec_c,
                                 getenv("LLAMA_MOE_LAYER_WINDOWS") ? "on" : "off").c_str()
                        : " (no decode windows)");
    }
}
