#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // pooled indexer keys of layer il, F32 [indexer_head_size, blocks]: row b holds the block of
    // positions [b*ratio, (b+1)*ratio), pooled, normed and rotated as build_qsa_top_k scores it.
    // Each ubatch pools only the blocks its tokens touch, instead of every block on every token.
    // A row is valid once its block's last cell was written; incomplete blocks are masked anyway.
    // nullptr when there is no indexer or more than one sequence
    ggml_tensor * get_pooled(int32_t il) const;

    // true when the next graph must pool every block: the indexer cells changed outside a graph
    bool get_pooled_stale() const { return pooled_stale; }
    void set_pooled_valid() const { pooled_stale = false; }

    // sizes the pooled keys to the indexer cache, which can have grown since the last call
    void pooled_fit();

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // see get_pooled; indexed by layer, nullptr for layers without a sparse indexer
    std::vector<ggml_tensor *> pooled;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> pooled_bufs;

    // per layer: where its rows live and its block size (0: no pooled keys); empty: no pooled cache
    std::vector<ggml_backend_buffer_type_t> pooled_buft;
    std::vector<uint32_t> pooled_ratio;
    int64_t  pooled_head_size = 0;
    uint32_t pooled_cells     = 0; // indexer cells the rows were sized for

    // set by every change to the indexer cells that no graph makes: restores, copies, shifts, clears
    mutable bool pooled_stale = true;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer, and for the update context, which builds no sparse graph
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // see llama_memory_hybrid_idx::get_pooled
    ggml_tensor * get_pooled(int32_t il) const;

    // blocks a graph for ubatch pools into the pooled cache: all n_blocks for a full context, a stale
    // cache or positions that are not contiguous, else at most the blocks contiguous tokens can touch,
    // so that steps of one size share a graph
    int64_t get_n_pool(const llama_ubatch & ubatch, uint32_t ratio, int64_t n_blocks) const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_pool, ns]   cells making up each pooled block
    //   blk_pos   I32 [4*n_pool*ns]        mrope position rows of each pooled block's first token
    //   pool_blk  I32 [n_pool]             the pooled blocks, with the pooled cache only (else nullptr)
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // n_pool (get_n_pool) is n_blocks, every block in order, or the blocks the ubatch's tokens touch,
    // repeated at the end to fill the graph's shape
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * pool_blk, ggml_tensor * bias, const llama_ubatch * ubatch,
                       uint32_t ratio, bool blk_bias) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // the reservation context, whose graph must pool every block to size the worst case
    const bool full = false;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer and this is a batch or full context
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
