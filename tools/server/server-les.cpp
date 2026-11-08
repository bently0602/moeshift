#ifdef LLAMA_LES_METAL_KEEPALIVE
#include "gpu-keepalive.h"
#endif

#include "server-context.h"
#include "server-http.h"
#include "server-les-router.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <filesystem>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#include <thread>

// The router owns the worker, including its Metal queues, even after SIGKILL.
class worker_parent_guard {
    std::atomic<bool> stopped{false};
    std::thread watcher;
public:
    worker_parent_guard() : watcher([this, parent = getppid()] {
        while (!stopped.load(std::memory_order_relaxed)) {
            if (parent == 1 || getppid() != parent) {
                std::fputs("moeshift: server exited; stopping orphan worker\n", stderr);
                std::_Exit(EXIT_FAILURE);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }) {}
    ~worker_parent_guard() {
        stopped.store(true, std::memory_order_relaxed);
        watcher.join();
    }
};

static volatile std::sig_atomic_t stop_requested = 0;

static void handle_signal(int) {
    stop_requested = 1;
}

int main(int argc, char ** argv) {
    common_log_set_application(common_log_main(), "moeshift");
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    if (argc < 2 || std::string(argv[1]) != "--worker") {
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            std::cout << "Usage: moeshift [--config FILE] [--model GGUF] [--moe-experts DIR] [--port N]\n"
                      << "Profile options override the configuration file; inference parallelism is always 1.\n";
            return 0;
        }
        try {
            std::filesystem::path config = "moeshift.json";
            int first = 1;
            bool user_file = false;
            if (argc > 1 && std::string(argv[1]) == "--config") {
                if (argc < 3) { std::cerr << "moeshift: --config requires a file\n"; return 1; }
                config = argv[2];
                first = 3;
            } else if (!std::filesystem::exists(std::filesystem::symlink_status(config))) {
                const char * xdg = std::getenv("XDG_CONFIG_HOME");
                const char * home = std::getenv("HOME");
                if (xdg && *xdg) config = std::filesystem::path(xdg) / "moeshift/config.json";
                else if (home && *home) config = std::filesystem::path(home) / ".config/moeshift/config.json";
                else throw std::runtime_error("Set HOME or pass --config FILE.");
                user_file = true;
            }

            bool repair = false;
            if (!user_file || std::filesystem::exists(std::filesystem::symlink_status(config))) {
                const int result = run_les_router(config.c_str(), stop_requested, argc - first, argv + first);
                if (result != invalid_router_config) return result;
                if (!user_file) {
                    std::cerr << "Correct " << config << " and run moeshift again.\n";
                    return 1;
                }
                repair = true;
            }
            if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
                std::cerr << "Run moeshift in a terminal to " << (repair ? "repair" : "create")
                          << " " << config << ", or pass --config FILE.\n";
                return 1;
            }
#ifdef __APPLE__
            uint32_t size = 0;
            _NSGetExecutablePath(nullptr, &size);
            std::vector<char> path(size);
            if (_NSGetExecutablePath(path.data(), &size) != 0) throw std::runtime_error("Cannot locate moeshift.");
            auto binary = std::filesystem::canonical(path.data());
            auto prefix = binary.parent_path().parent_path();
            auto wizard = prefix / "share/moeshift/tools/moe/onboarding.py";
            if (!std::filesystem::is_regular_file(wizard)) throw std::runtime_error("Install the moeshift CMake component to use the setup wizard.");
            auto python = prefix / "libexec/bin/python";
            std::vector<std::string> args{std::filesystem::exists(python) ? python.string() : "python3",
                wizard.string(), config.string(), binary.string(), repair ? "repair" : "new"};
            args.insert(args.end(), argv + first, argv + argc);
            std::vector<char *> command;
            for (auto & arg : args) command.push_back(arg.data());
            command.push_back(nullptr);
            execvp(command.front(), command.data());
            throw std::runtime_error("Cannot start Python for setup.");
#else
            throw std::runtime_error("The setup wizard requires macOS.");
#endif
        } catch (const std::exception & e) {
            std::cerr << "moeshift: " << e.what() << '\n';
            return 1;
        }
    }
    --argc;
    ++argv;
    worker_parent_guard parent_guard;

    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option.size() <= 2 || option.substr(0, 2) != "--" ||
            (!moeshift_model_options.count(option.substr(2)) && option != "--host" && option != "--port")) {
            std::cerr << "moeshift: Unsupported worker option: " << option << "\n";
            return 1;
        }
        if (option != "--no-mmap" && ++i == argc) {
            std::cerr << "moeshift: Missing worker option value\n";
            return 1;
        }
    }

    common_params params;

    common_init();

    if (!common_params_parse(
            argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // LES always serves one inference at a time with the experts page cache enabled.
    params.n_parallel = 1;
    params.hostname = "127.0.0.1";
    setenv("LLAMA_MOE_F_NOCACHE", "0", 1);

    // A speculative check of n drafted tokens is one decode step of n + 1 rows. Wider steps would
    // leave the decode phase, so drafts are capped to the decode rows the expert windows allow.
    {
        const char * rows_env = std::getenv("LLAMA_MOE_DECODE_ROWS");
        const int rows = rows_env && *rows_env ? std::atoi(rows_env) : 1;
        const bool speculative = std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
                [](common_speculative_type t) { return t != COMMON_SPECULATIVE_TYPE_NONE; });
        if (speculative && rows < 2) {
            std::cerr << "moeshift: speculative decoding needs LLAMA_MOE_DECODE_ROWS of 2 or more; disabled\n";
            params.speculative = {};
        } else if (speculative) {
            params.speculative.draft.n_max = std::min(params.speculative.draft.n_max, rows - 1);
            params.speculative.draft.n_min = std::min(params.speculative.draft.n_min, params.speculative.draft.n_max);
            setenv("LLAMA_SPEC_DRAFT_CAP", std::to_string(rows - 1).c_str(), 1);
        }
    }
    if (params.moe_experts.empty() || params.les_decode_bank >= params.les_prefill_bank) {
        std::cerr << "moeshift: adaptive expert streaming requires MoE experts and decode bank < prefill bank\n";
        return 1;
    }
    // Enable growing KV buffers for LES; other clients retain the original allocator.
    setenv("LLAMA_KV_GROW", "1", 1);
    params.n_batch = params.n_ubatch = params.les_prefill_max_tokens;
    params.moe_slot_bank = params.les_prefill_bank;
    setenv("LLAMA_LES_DECODE_BANK", std::to_string(params.les_decode_bank).c_str(), 1);
    const char * moe_quiet = std::getenv("LLAMA_MOE_QUIET");
    if (moe_quiet && std::string(moe_quiet) != "1") {
        std::cerr << "moeshift: adaptive: prefill-max=" << params.n_ubatch
                  << " shared-bank=" << params.les_prefill_bank << " -> " << params.les_decode_bank << "\n";
    }

#ifdef LLAMA_LES_METAL_KEEPALIVE
    gpu_keepalive keepalive;
    if (!keepalive.ready) {
        return 1;
    }
#endif

    llama_backend_init();
    llama_numa_init(params.numa);

    int result = 0;

    {
        server_context ctx;
        server_routes routes(params, ctx);
        server_http_context http;

        if (!http.init(params)) {
            std::cerr << "moeshift: HTTP initialization failed\n";
            result = 1;
        } else {
            // OpenAI-compatible endpoints.
            http.get(
                "/health",
                routes.get_health
            );

            http.get(
                "/v1/models",
                routes.get_models
            );

            http.post(
                "/v1/chat/completions",
                routes.post_chat_completions
            );

            http.post(
                "/v1/completions",
                routes.post_completions_oai
            );

            // Load the model through the existing
            // llama.cpp inference backend.
            if (!ctx.load_model(params)) {
                std::cerr << "moeshift: Model loading failed\n";
                result = 1;
            } else {
                routes.update_meta(ctx);

                if (!http.start()) {
                    std::cerr << "moeshift: HTTP server startup failed\n";
                    result = 1;
                } else {
                    http.is_ready.store(true);

                    std::cout
                        << "moeshift: OpenAI-compatible API listening on "
                        << http.listening_address
                        << "/v1\n";

                    std::thread inference([&ctx]() {
                        ctx.start_loop();
                    });

                    while (!stop_requested) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(200)
                        );
                    }

                    ctx.terminate();
                    inference.join();

                    http.stop();
                    if (http.thread.joinable()) {
                        http.thread.join();
                    }
                }
            }
        }
    }

    llama_backend_free();

    return result;
}
