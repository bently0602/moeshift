#include "server-les-router.h"
#include "server-models.h"

#include <cpp-httplib/httplib.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#ifndef _WIN32
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace {
using namespace std::chrono_literals;

void require(bool valid, const std::string & message) {
    if (!valid) {
        throw std::runtime_error(message);
    }
}

void check_keys(const json & value, const std::set<std::string> & allowed) {
    require(value.is_object(), "Expected a JSON object");
    for (const auto & item : value.items()) {
        require(allowed.count(item.key()), "Unknown configuration key: " + item.key());
    }
}

std::string scalar(const json & value) {
    require(value.is_string() || value.is_number_integer(), "Expected a string or integer");
    auto result = value.is_string() ? value.get<std::string>() : value.dump();
    require(!result.empty() && result.find('\0') == std::string::npos, "Empty value or embedded NUL");
    return result;
}

std::string resolve_path(const std::filesystem::path & base, const json & value) {
    require(value.is_string(), "File paths must be strings");
    return std::filesystem::absolute(base / scalar(value)).lexically_normal().string();
}

server_http_res_ptr reply(int status, const json & body) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = body.dump();
    return res;
}

server_http_res_ptr error(int status, const std::string & message) {
    return reply(status, {{"error", {{"message", message},
            {"type", status >= 500 ? "server_error" : "invalid_request_error"}, {"code", status}}}});
}

#ifndef _WIN32
struct les_worker {
    pid_t pid = -1;
    int port = 0;
    std::string model;

    bool running() {
        if (pid < 0) {
            return false;
        }
        int status;
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == 0 || (result < 0 && errno == EINTR)) {
            return true;
        }
        pid = -1;
        model.clear();
        return false;
    }

    void stop() {
        if (!running()) {
            return;
        }
        kill(pid, SIGTERM);
        const auto deadline = std::chrono::steady_clock::now() + 10s;
        while (running() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(50ms);
        }
        if (pid > 0) {
            kill(pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
            pid = -1;
        }
        model.clear();
    }

    ~les_worker() { stop(); }

    void load(const std::string & binary, const std::string & name, const json & profile,
              const std::filesystem::path & base, const std::function<bool()> & cancelled) {
        if (running() && model == name) {
            return;
        }
        stop();
        port = server_get_free_port();
        require(port > 0, "Cannot allocate a worker port");

        std::vector<std::string> args{binary, "--worker"};
        for (const auto & item : profile.at("args").items()) {
            if (item.key() == "no-mmap") {
                if (item.value().get<bool>()) {
                    args.push_back("--no-mmap");
                }
            } else {
                args.push_back("--" + item.key());
                args.push_back(item.key() == "model" || item.key() == "moe-experts"
                        ? resolve_path(base, item.value()) : scalar(item.value()));
            }
        }
        args.insert(args.end(), {"--host", "127.0.0.1", "--port", std::to_string(port)});

        // Environment belongs to the profile; shell-level tuning must not leak into it.
        std::map<std::string, std::string> env;
        for (char ** entry = environ; entry && *entry; ++entry) {
            const std::string value(*entry);
            const auto separator = value.find('=');
            const auto key = value.substr(0, separator);
            if (key.rfind("LLAMA_", 0) && key.rfind("GGML_", 0) && key.rfind("LES_", 0) && key.rfind("MB_", 0)) {
                env[key] = value.substr(separator + 1);
            }
        }
        for (const auto & item : profile.at("env").items()) {
            env[item.key()] = scalar(item.value());
        }
        env["LLAMA_MOE_F_NOCACHE"] = "0";
        std::vector<std::string> environment;
        for (const auto & item : env) {
            environment.push_back(item.first + "=" + item.second);
        }
        std::vector<char *> argv, envp;
        for (auto & arg : args) { argv.push_back(arg.data()); }
        for (auto & value : environment) { envp.push_back(value.data()); }
        argv.push_back(nullptr);
        envp.push_back(nullptr);
        const int rc = posix_spawn(&pid, binary.c_str(), nullptr, nullptr, argv.data(), envp.data());
        if (rc) {
            pid = -1;
            throw std::runtime_error("Cannot start worker: " + std::string(std::strerror(rc)));
        }
        std::cerr << "moeshift: loading " << name << " (pid " << pid << ")\n";
        const auto deadline = std::chrono::steady_clock::now() + 600s;
        httplib::Client health("127.0.0.1", port);
        health.set_connection_timeout(1);
        health.set_read_timeout(1);
        while (!cancelled() && std::chrono::steady_clock::now() < deadline && running()) {
            auto response = health.Get("/health");
            if (response && response->status == 200) {
                model = name;
                return;
            }
            std::this_thread::sleep_for(100ms);
        }
        stop();
        throw std::runtime_error("Worker failed to load model: " + name + " (see worker log)");
    }
};

void apply_overrides(json & config, int argc, char ** argv) {
    for (int i = 0; i < argc; i += 2) {
        require(i + 1 < argc, "Expected a value for " + std::string(argv[i]));
        const std::string key(argv[i]);
        if (key == "--port") {
            config["port"] = json::parse(argv[i + 1]);
            continue;
        }
        const std::map<std::string, std::string> env_options{
            {"--moe-protected-slots", "LLAMA_MOE_PROTECTED_SLOTS"},
            {"--moe-decode-window", "LLAMA_MOE_DECODE_WINDOW"}};
        require(key.size() > 2 && key.substr(0, 2) == "--", "Expected an option");
        for (auto & profile : config.at("models")) {
            const auto env = env_options.find(key);
            if (env != env_options.end()) profile["env"][env->second] = argv[i + 1];
            else if (key == "--no-mmap") profile["args"]["no-mmap"] = json::parse(argv[i + 1]);
            else profile["args"][key.substr(2)] = argv[i + 1];
        }
    }
}

void validate_config(const json & config, const std::filesystem::path & base, std::string & binary,
                     common_params & params, std::set<std::string> & aliases) {
    check_keys(config, {"binary", "host", "port", "models"});
    if (config.contains("binary")) binary = resolve_path(base, config.at("binary"));
    else {
#ifdef __APPLE__
        uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::vector<char> path(size);
        require(_NSGetExecutablePath(path.data(), &size) == 0, "Cannot resolve executable");
        binary = std::filesystem::canonical(path.data()).string();
#else
        throw std::runtime_error("MoEShift requires macOS");
#endif
    }
    require(access(binary.c_str(), X_OK) == 0, "Worker binary is not executable: " + binary);
    require(config.at("host").is_string(), "host must be a string");
    params.hostname = scalar(config.at("host"));
    require(params.hostname == "127.0.0.1" || params.hostname == "::1" || params.hostname == "localhost",
            "MoEShift only accepts localhost addresses");
    require(config.at("port").is_number_integer(), "port must be an integer");
    require(config.at("port") > 0 && config.at("port") <= 65535, "port must be in 1..65535");
    params.port = config.at("port").get<int>();
    params.timeout_read = params.timeout_write = 600;
    const auto & models = config.at("models");
    require(models.is_object() && !models.empty(), "models must be a nonempty object");
    const std::set<std::string> allowed_env{
        "LLAMA_GPU_KEEPALIVE", "LLAMA_MOE_DECODE_WINDOW", "LLAMA_MOE_LOOKAHEAD_PREFETCH",
        "LLAMA_MOE_PREFETCH_READ", "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH",
        "LLAMA_MOE_PREFILL_PIPELINE", "LLAMA_MOE_PROTECTED_SLOTS", "LLAMA_MOE_QUIET"};
    for (const auto & entry : models.items()) {
        scalar(entry.key());
        check_keys(entry.value(), {"args", "env"});
        const auto & args = entry.value().at("args");
        const auto & env = entry.value().at("env");
        check_keys(args, moeshift_model_options);
        check_keys(env, allowed_env);
        resolve_path(base, args.at("model"));
        resolve_path(base, args.at("moe-experts"));
        require(args.at("alias").is_string(), "alias must be a string");
        const auto alias = scalar(args.at("alias"));
        require(alias == entry.key(), "alias must match its models key: " + entry.key());
        require(alias.find(',') == std::string::npos, "One alias is allowed per profile");
        for (const auto & arg : args.items()) {
            if (arg.key() == "no-mmap") {
                require(arg.value().is_boolean(), "no-mmap must be a boolean");
            } else {
                scalar(arg.value());
            }
        }
        for (const auto & value : env.items()) { scalar(value.value()); }
        if (env.contains("LLAMA_MOE_PROTECTED_SLOTS") && scalar(env.at("LLAMA_MOE_PROTECTED_SLOTS")) != "0") {
            const auto depth = env.contains("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH")
                ? std::atoi(scalar(env.at("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH")).c_str()) : 0;
            require(env.contains("LLAMA_MOE_DECODE_WINDOW") && scalar(env.at("LLAMA_MOE_DECODE_WINDOW")) == "12" &&
                    depth >= 1 && depth <= 16 &&
                    env.contains("LLAMA_MOE_LOOKAHEAD_PREFETCH") && scalar(env.at("LLAMA_MOE_LOOKAHEAD_PREFETCH")) == "1",
                    "Protected LRFU cache requires decode window 12 and enabled prefetch depth 1-16; set protected slots to 0 to disable it");
        }
        aliases.insert(alias);
    }
}
#endif
} // namespace

int run_les_router(const char * config_path, const volatile std::sig_atomic_t & stop_requested, int argc, char ** argv) {
#ifdef _WIN32
    (void) config_path;
    (void) stop_requested;
    std::cerr << "moeshift configuration router currently requires macOS or Linux\n";
    return 1;
#else
    bool validating_file = false;
    try {
        const auto base = std::filesystem::absolute(config_path).parent_path();
        std::ifstream input(config_path);
        require(input.good(), std::string("Cannot open configuration file: ") + config_path);
        validating_file = true;
        auto config = json::parse(input);
        std::string binary;
        common_params params;
        std::set<std::string> aliases;
        if (argc) {
            // Validate the saved file separately only when the overridden profile fails.
            auto overridden = config;
            try {
                apply_overrides(overridden, argc, argv);
                validate_config(overridden, base, binary, params, aliases);
                config = std::move(overridden);
            } catch (const std::exception &) {
                validating_file = true;
                validate_config(config, base, binary, params, aliases);
                validating_file = false;
                throw;
            }
        } else {
            validate_config(config, base, binary, params, aliases);
        }
        const auto & models = config.at("models");
        validating_file = false;
        std::mutex mutex;
        std::condition_variable cv;
        bool busy = false;
        les_worker worker;
        server_http_context http;
        require(http.init(params), "HTTP initialization failed");
        http.get("/health", [](const server_http_req &) {
            return reply(200, {{"status", "ok"}});
        });
        http.get("/v1/models", [&aliases](const server_http_req &) {
            json data = json::array();
            for (const auto & alias : aliases) {
                data.push_back({{"id", alias}, {"object", "model"}, {"owned_by", "local"}, {"created", 0}});
            }
            return reply(200, {{"object", "list"}, {"data", data}});
        });
        const auto infer = [&](const server_http_req & req) -> server_http_res_ptr {
            const auto body = json::parse(req.body, nullptr, false);
            if (!body.is_object() || !body.contains("model") || !body["model"].is_string()) {
                return error(400, "A JSON object with a string model field is required");
            }
            const auto name = body["model"].get<std::string>();
            if (!models.contains(name)) {
                return error(404, "Unknown model: " + name);
            }
            const auto cancelled = [&req, &stop_requested] { return stop_requested || req.should_stop(); };
            std::unique_lock<std::mutex> lock(mutex);
            while (busy && !cancelled()) { cv.wait_for(lock, 100ms); }
            if (cancelled()) { return error(503, "Request cancelled or server stopping"); }
            busy = true;
            // The lease survives until the complete HTTP response, including SSE, is released.
            const auto lease = std::shared_ptr<void>(nullptr, [&](void *) {
                std::lock_guard<std::mutex> guard(mutex);
                busy = false;
                cv.notify_all();
            });
            lock.unlock();
            try {
                worker.load(binary, name, models.at(name), base, cancelled);
                auto proxy = std::make_unique<server_http_proxy>("POST", "http", "127.0.0.1", worker.port,
                        req.path, req.headers, req.body, req.files, cancelled, 600, 600);
                proxy->next = [next = std::move(proxy->next), lease](std::string & output) { return next(output); };
                return proxy;
            } catch (const std::exception & e) {
                return error(503, e.what());
            }
        };
        http.post("/v1/chat/completions", infer);
        http.post("/v1/completions", infer);
        require(http.start(), "HTTP startup failed");
        http.is_ready.store(true);
        if (isatty(STDERR_FILENO)) {
            const char * term = std::getenv("TERM");
            const bool color = !std::getenv("NO_COLOR") && (!term || std::strcmp(term, "dumb") != 0);
            std::cerr << "\n" << (color ? "\033[1;32m" : "") << "✓  MoEShift is ready!"
                      << (color ? "\033[0m" : "") << "\n\n"
                      << "   Local API   " << http.listening_address << "/v1\n";
            for (const auto & alias : aliases) std::cerr << "   Model       " << alias << '\n';
            std::cerr << "\n   Press Ctrl-C to stop.\n\n";
        } else {
            std::cerr << "moeshift: Ready. Listening on " << http.listening_address << "/v1 with";
            for (const auto & alias : aliases) std::cerr << " " << alias;
            std::cerr << "\n";
        }
        while (!stop_requested) { std::this_thread::sleep_for(100ms); }
        http.stop();
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !busy; });
        }
        worker.stop();
        if (http.thread.joinable()) { http.thread.join(); }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "moeshift: " << (validating_file ? std::string(config_path) + ": " : "") << e.what() << '\n';
        return validating_file ? invalid_router_config : 1;
    }
#endif
}
