#pragma once

#include <csignal>
#include <set>
#include <string>

inline const std::set<std::string> moeshift_model_options{
    "model", "moe-experts", "ctx-size", "ctx-checkpoints", "prefill-max-tokens",
    "moe-prefill-bank", "moe-decode-bank", "n-gpu-layers", "flash-attn",
    "cache-type-k", "cache-type-v", "threads", "no-mmap", "alias"};

inline constexpr int invalid_router_config = 2;

int run_les_router(const char * config_path, const volatile std::sig_atomic_t & stop_requested, int argc = 0, char ** argv = nullptr);
