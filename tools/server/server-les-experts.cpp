// `moeshift experts`: the C++ version of tools/moe/experts.py (extract, verify) and
// tools/moe/strip_experts.py (strip), so that preparing a model needs no Python.
//
// extract  copies every routed expert tensor into one file per layer, expert-major
//          (layer_NNN.bin), and writes manifest.json in the same format as experts.py.
// verify   checks the manifest's metadata and every expert's bytes against the GGUF.
// strip    writes one GGUF without the expert tensors (shards merged), then reads it back
//          and compares every kept tensor. The engine builds expert tensors from the manifest,
//          so the original GGUF is not needed afterwards.
//
// All file reads and writes bypass the page cache (F_NOCACHE), so preparing a model does not
// evict the experts another model keeps cached. Each output file is written to a temporary
// name, synced and renamed.

#include "server-les-experts.h"

#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

constexpr size_t chunk_bytes = 8u << 20;

void require(bool valid, const std::string & message) {
    if (!valid) {
        throw std::runtime_error(message);
    }
}

const std::map<std::string, int> family_order{
    {"ffn_gate_exps", 0}, {"ffn_up_exps", 1}, {"ffn_down_exps", 2}, {"ffn_gate_up_exps", 3}};

int64_t int_value(const gguf_context * ctx, const std::string & key, int64_t fallback) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return fallback;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default: throw std::runtime_error(key + " is not an integer");
    }
}

std::string str_value(const gguf_context * ctx, const std::string & key, const std::string & fallback) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    return id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING ? fallback : gguf_get_val_str(ctx, id);
}

// A GGUF's metadata; tensor data stays in the file.
struct gguf_file {
    fs::path path;
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;

    explicit gguf_file(const fs::path & file) : path(file) {
        gguf_init_params params{true, &meta};
        gguf = gguf_init_from_file(file.c_str(), params);
        require(gguf != nullptr, "Cannot read GGUF " + file.string());
    }
    ~gguf_file() {
        if (meta) ggml_free(meta);
        if (gguf) gguf_free(gguf);
    }
    gguf_file(const gguf_file &) = delete;
    gguf_file & operator=(const gguf_file &) = delete;

    size_t offset(int64_t tensor) const {
        return gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor);
    }
};

struct fd_file {
    int fd = -1;
    fd_file(const fs::path & path, int flags) {
        fd = open(path.c_str(), flags | O_CLOEXEC, 0644);
        require(fd >= 0, "Cannot open " + path.string() + ": " + std::strerror(errno));
#ifdef F_NOCACHE
        fcntl(fd, F_NOCACHE, 1);
#endif
    }
    ~fd_file() {
        if (fd >= 0) close(fd);
    }
    fd_file(const fd_file &) = delete;
    fd_file & operator=(const fd_file &) = delete;
};

void read_at(int fd, void * data, size_t size, size_t offset, const std::string & label) {
    auto * out = static_cast<char *>(data);
    while (size) {
        const ssize_t n = pread(fd, out, size, (off_t) offset);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "Unexpected end of file while reading " + label);
        out += n;
        offset += (size_t) n;
        size -= (size_t) n;
    }
}

void write_all(int fd, const void * data, size_t size, const fs::path & path) {
    const auto * in = static_cast<const char *>(data);
    while (size) {
        const ssize_t n = write(fd, in, size);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "Cannot write " + path.string() + ": " + std::strerror(errno));
        in += n;
        size -= (size_t) n;
    }
}

// Written under a temporary name and renamed by commit(); removed if not committed.
struct atomic_output {
    fs::path path;
    fs::path tmp;
    std::unique_ptr<fd_file> file;
    bool committed = false;

    explicit atomic_output(const fs::path & target) : path(target) {
        fs::create_directories(target.parent_path());
        tmp = target.parent_path() / ("." + target.filename().string() + ".tmp." + std::to_string(getpid()));
        file = std::make_unique<fd_file>(tmp, O_WRONLY | O_CREAT | O_TRUNC);
    }
    ~atomic_output() {
        if (!committed) {
            file.reset();
            std::error_code ignored;
            fs::remove(tmp, ignored);
        }
    }
    void write(const void * data, size_t size) { write_all(file->fd, data, size, tmp); }
    void commit() {
        require(fsync(file->fd) == 0, "Cannot sync " + tmp.string() + ": " + std::strerror(errno));
        file.reset();
        fs::rename(tmp, path);
        committed = true;
    }
};

void check_space(const fs::path & folder, uintmax_t needed) {
    const uintmax_t free = fs::space(folder).available;
    if (needed > free) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s needs %.2f GB of free space; %.2f GB available",
                      folder.c_str(), needed / 1e9, free / 1e9);
        throw std::runtime_error(message);
    }
}

std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char) std::toupper(c); });
    return text;
}

std::vector<fs::path> resolve_model_paths(fs::path model) {
    model = fs::absolute(model);  // keeps shard names in symlink-based Hugging Face snapshots
    const int64_t count = int_value(gguf_file(model).gguf, "split.count", 1);
    if (count == 1) {
        return {model};
    }
    static const std::regex split_re(R"(^(.+-)(\d+)-of-(\d+)(\.gguf)$)");
    const std::string name = model.filename().string();
    std::smatch match;
    require(std::regex_match(name, match, split_re), "Cannot derive split names from '" + name + "'");
    const int width = (int) match[2].length();
    const int count_width = (int) match[3].length();
    std::vector<fs::path> paths;
    std::string missing;
    for (int64_t i = 1; i <= count; ++i) {
        char numbers[64];
        std::snprintf(numbers, sizeof(numbers), "%0*lld-of-%0*lld", width, (long long) i, count_width, (long long) count);
        paths.push_back(model.parent_path() / (match[1].str() + numbers + match[4].str()));
        if (!fs::exists(paths.back())) {
            missing += (missing.empty() ? "" : ", ") + paths.back().string();
        }
    }
    require(missing.empty(), "Missing GGUF shards: " + missing);
    return paths;
}

std::optional<std::set<int>> parse_layers(const std::string & spec) {
    if (spec.empty()) {
        return std::nullopt;
    }
    std::set<int> result;
    size_t start = 0;
    while (start <= spec.size()) {
        const size_t end = std::min(spec.find(',', start), spec.size());
        const std::string item = spec.substr(start, end - start);
        const size_t dash = item.find('-');
        if (dash != std::string::npos) {
            const int first = std::stoi(item.substr(0, dash));
            const int last = std::stoi(item.substr(dash + 1));
            require(last >= first, "Invalid layer range '" + item + "'");
            for (int layer = first; layer <= last; ++layer) result.insert(layer);
        } else if (item.find_first_not_of(" \t") != std::string::npos) {
            result.insert(std::stoi(item));
        }
        start = end + 1;
    }
    return result;
}

struct expert_entry {
    int layer = 0;
    std::string family;
    std::string name;
    std::string quant_type;
    int64_t block_size = 0;
    std::vector<int64_t> shape;
    size_t bytes = 0;
    size_t bytes_per_expert = 0;
    size_t shard = 0;
    size_t source_offset = 0;
};

struct model_info {
    std::string arch;
    int64_t expert_count = 0;
    int64_t expert_used_count = 0;
};

struct gguf_model {
    std::vector<std::unique_ptr<gguf_file>> shards;
    model_info info;

    explicit gguf_model(const fs::path & model) {
        for (const auto & path : resolve_model_paths(model)) {
            shards.push_back(std::make_unique<gguf_file>(path));
        }
        const gguf_context * first = shards.front()->gguf;
        info.arch = str_value(first, "general.architecture", "unknown");
        info.expert_count = int_value(first, info.arch + ".expert_count", 0);
        info.expert_used_count = int_value(first, info.arch + ".expert_used_count", 0);
    }

    // The routed expert tensors, ordered by layer and family as experts.py orders them.
    std::vector<expert_entry> experts(const std::optional<std::set<int>> & layers) const {
        static const std::regex tensor_re(R"(^blk\.(\d+)\.(ffn_[^.]+)\.weight$)");
        std::vector<expert_entry> entries;
        for (size_t s = 0; s < shards.size(); ++s) {
            const gguf_file & shard = *shards[s];
            for (int64_t i = 0; i < gguf_get_n_tensors(shard.gguf); ++i) {
                const std::string name = gguf_get_tensor_name(shard.gguf, i);
                std::smatch match;
                if (!std::regex_match(name, match, tensor_re) || !family_order.count(match[2].str())) continue;
                const int layer = std::stoi(match[1].str());
                if (layers && !layers->count(layer)) continue;

                const ggml_tensor * tensor = ggml_get_tensor(shard.meta, name.c_str());
                require(tensor != nullptr, "Missing tensor metadata for '" + name + "'");
                expert_entry entry;
                entry.layer = layer;
                entry.family = match[2].str();
                entry.name = name;
                entry.quant_type = upper(ggml_type_name(tensor->type));
                entry.block_size = ggml_blck_size(tensor->type);
                for (int d = 0; d < ggml_n_dims(tensor); ++d) entry.shape.push_back(tensor->ne[d]);
                entry.bytes = gguf_get_tensor_size(shard.gguf, i);
                if (entry.shape.size() < 3 || entry.shape[2] != info.expert_count) {
                    throw std::runtime_error("'" + name + "' does not have " + std::to_string(info.expert_count) +
                                             " experts on axis 2");
                }
                require(entry.bytes % (size_t) info.expert_count == 0,
                        "'" + name + "' byte size is not divisible by the expert count");
                entry.bytes_per_expert = entry.bytes / (size_t) info.expert_count;
                entry.shard = s;
                entry.source_offset = shard.offset(i);
                entries.push_back(std::move(entry));
            }
        }
        std::sort(entries.begin(), entries.end(), [](const expert_entry & a, const expert_entry & b) {
            return a.layer != b.layer ? a.layer < b.layer : family_order.at(a.family) < family_order.at(b.family);
        });
        return entries;
    }
};

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
    std::tm utc{};
    gmtime_r(&seconds, &utc);
    char text[64];
    const size_t n = std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%S", &utc);
    std::snprintf(text + n, sizeof(text) - n, ".%06lld+00:00", (long long) micros);
    return text;
}

// "--key value" and "--flag" options; flags are named in `flags`.
std::map<std::string, std::string> parse_options(int argc, char ** argv, const std::set<std::string> & values,
                                                 const std::set<std::string> & flags) {
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (flags.count(key)) {
            options[key] = "1";
        } else if (values.count(key)) {
            require(i + 1 < argc, "Expected a value for " + key);
            options[key] = argv[++i];
        } else {
            throw std::runtime_error("Unknown option " + key);
        }
    }
    return options;
}

std::string required(const std::map<std::string, std::string> & options, const std::string & key) {
    const auto found = options.find(key);
    require(found != options.end() && !found->second.empty(), key + " is required");
    return found->second;
}

std::string value(const std::map<std::string, std::string> & options, const std::string & key) {
    const auto found = options.find(key);
    return found == options.end() ? std::string() : found->second;
}

int extract(int argc, char ** argv) {
    const auto options = parse_options(argc, argv, {"--model", "--out-dir", "--layers"}, {"--force"});
    const gguf_model model(required(options, "--model"));
    const fs::path out_dir = fs::weakly_canonical(fs::absolute(required(options, "--out-dir")));
    fs::create_directories(out_dir);
    require(options.count("--force") || fs::is_empty(out_dir),
            "Output directory '" + out_dir.string() + "' is not empty; pass --force to overwrite");

    const auto entries = model.experts(parse_layers(value(options, "--layers")));
    require(!entries.empty(), "No routed expert tensors found");
    uintmax_t total = 0;
    for (const auto & entry : entries) total += entry.bytes;
    check_space(out_dir, total);

    std::map<int, std::vector<const expert_entry *>> by_layer;
    for (const auto & entry : entries) by_layer[entry.layer].push_back(&entry);

    std::vector<std::unique_ptr<fd_file>> sources;
    for (const auto & shard : model.shards) sources.push_back(std::make_unique<fd_file>(shard->path, O_RDONLY));

    json manifest_entries = json::array();
    std::vector<char> buffer;
    size_t done = 0;
    for (const auto & [layer, layer_entries] : by_layer) {
        char file_name[32];
        std::snprintf(file_name, sizeof(file_name), "layer_%03d.bin", layer);
        size_t stride = 0;
        for (const auto * entry : layer_entries) stride += entry->bytes_per_expert;

        size_t family_offset = 0;
        for (const auto * entry : layer_entries) {
            manifest_entries.push_back({
                {"layer", entry->layer},
                {"tensor_family", entry->family},
                {"tensor_name", entry->name},
                {"original_gguf_tensor_name", entry->name},
                {"quant_type", entry->quant_type},
                {"block_size", entry->block_size},
                {"shape", entry->shape},
                {"exact_byte_length", entry->bytes},
                {"bytes_per_expert", entry->bytes_per_expert},
                {"source_shard", model.shards[entry->shard]->path.string()},
                {"source_shard_index", entry->shard},
                {"source_offset", entry->source_offset},
                {"repacked_file", file_name},
                {"repacked_offset", family_offset},
                {"expert_stride", stride},
            });
            family_offset += entry->bytes_per_expert;
        }

        // Several experts per write: about 64 MiB at a time.
        const size_t per_write = std::max<size_t>(1, (64u << 20) / stride);
        buffer.resize(per_write * stride);
        atomic_output output(out_dir / file_name);
        for (int64_t first = 0; first < model.info.expert_count; first += (int64_t) per_write) {
            const int64_t last = std::min<int64_t>(model.info.expert_count, first + (int64_t) per_write);
            char * out = buffer.data();
            for (int64_t expert = first; expert < last; ++expert) {
                for (const auto * entry : layer_entries) {
                    read_at(sources[entry->shard]->fd, out, entry->bytes_per_expert,
                            entry->source_offset + (size_t) expert * entry->bytes_per_expert,
                            entry->name + " expert " + std::to_string(expert));
                    out += entry->bytes_per_expert;
                }
            }
            output.write(buffer.data(), (size_t) (out - buffer.data()));
        }
        output.commit();
        std::cout << "moeshift: extracted layer " << ++done << "/" << by_layer.size() << std::endl;
    }

    std::vector<std::string> model_files;
    for (const auto & shard : model.shards) model_files.push_back(shard->path.string());
    const json manifest = {
        {"schema_version", 1},
        {"experts_kind", "omni_moe_gguf"},
        {"layout", "layer_major_expert"},
        {"generated_at", utc_now()},
        {"source", {{"model_files", model_files}}},
        {"model", {{"arch", model.info.arch}, {"expert_count", model.info.expert_count},
                   {"expert_used_count", model.info.expert_used_count}}},
        {"entries", manifest_entries},
    };
    const std::string text = manifest.dump(2) + "\n";
    atomic_output output(out_dir / "manifest.json");
    output.write(text.data(), text.size());
    output.commit();

    std::cout << "moeshift: wrote " << entries.size() << " tensors in " << by_layer.size() << " layers ("
              << total << " bytes)\n" << (out_dir / "manifest.json").string() << std::endl;
    return 0;
}

json read_manifest(const fs::path & experts, fs::path & manifest_path) {
    manifest_path = fs::is_directory(experts) ? experts / "manifest.json" : experts;
    std::ifstream stream(manifest_path);
    require(stream.is_open(), "Cannot open " + manifest_path.string());
    json manifest;
    stream >> manifest;
    return manifest;
}

int verify(int argc, char ** argv) {
    const auto options = parse_options(argc, argv, {"--model", "--experts", "--layers"}, {"--metadata-only"});
    const gguf_model model(required(options, "--model"));
    fs::path manifest_path;
    const json manifest = read_manifest(fs::absolute(required(options, "--experts")), manifest_path);
    const auto expected = model.experts(parse_layers(value(options, "--layers")));
    std::map<std::string, const expert_entry *> by_name;
    for (const auto & entry : expected) by_name[entry.name] = &entry;
    const bool metadata_only = options.count("--metadata-only") > 0;

    std::vector<std::unique_ptr<fd_file>> sources;
    if (!metadata_only) {
        for (const auto & shard : model.shards) sources.push_back(std::make_unique<fd_file>(shard->path, O_RDONLY));
    }
    std::map<std::string, std::unique_ptr<fd_file>> expert_files;
    std::vector<char> a(chunk_bytes), b(chunk_bytes);
    std::set<std::string> seen;
    const auto & entries = manifest.at("entries");
    size_t done = 0;
    for (const auto & item : entries) {
        const std::string name = item.at("tensor_name").get<std::string>();
        const auto found = by_name.find(name);
        require(found != by_name.end(), "Unexpected manifest tensor '" + name + "'");
        const expert_entry & source = *found->second;
        seen.insert(name);
        require(item.at("shape").get<std::vector<int64_t>>() == source.shape, "Metadata mismatch for '" + name + "': shape");
        require(upper(item.at("quant_type").get<std::string>()) == source.quant_type,
                "Metadata mismatch for '" + name + "': quant_type");
        require(item.at("exact_byte_length").get<size_t>() == source.bytes,
                "Metadata mismatch for '" + name + "': exact_byte_length");
        require(item.at("bytes_per_expert").get<size_t>() == source.bytes_per_expert,
                "Metadata mismatch for '" + name + "': bytes_per_expert");
        if (metadata_only) continue;

        const std::string file = item.at("repacked_file").get<std::string>();
        auto & experts = expert_files[file];
        if (!experts) experts = std::make_unique<fd_file>(manifest_path.parent_path() / file, O_RDONLY);
        const size_t offset = item.at("repacked_offset").get<size_t>();
        const size_t stride = item.at("expert_stride").get<size_t>();
        for (int64_t expert = 0; expert < model.info.expert_count; ++expert) {
            for (size_t at = 0; at < source.bytes_per_expert; at += chunk_bytes) {
                const size_t n = std::min(chunk_bytes, source.bytes_per_expert - at);
                const std::string label = name + " expert " + std::to_string(expert);
                read_at(sources[source.shard]->fd, a.data(), n, source.source_offset + (size_t) expert * source.bytes_per_expert + at, label);
                read_at(experts->fd, b.data(), n, offset + (size_t) expert * stride + at, label);
                require(std::memcmp(a.data(), b.data(), n) == 0, "Byte mismatch for '" + name + "' expert " + std::to_string(expert));
            }
        }
        std::cout << "moeshift: verified tensor " << ++done << "/" << entries.size() << std::endl;
    }
    require(seen.size() == by_name.size(), "The manifest's tensors do not match the requested GGUF tensors");
    std::cout << "moeshift: verified " << entries.size() << " tensors ("
              << (metadata_only ? "metadata" : "metadata and bytes") << ")" << std::endl;
    return 0;
}

int strip(int argc, char ** argv) {
    const auto options = parse_options(argc, argv, {"--model", "--experts", "--out"}, {"--force"});
    const gguf_model model(required(options, "--model"));
    fs::path manifest_path;
    const json manifest = read_manifest(fs::absolute(required(options, "--experts")), manifest_path);
    const fs::path out = fs::absolute(required(options, "--out"));
    for (const auto & shard : model.shards) {
        require(!fs::exists(out) || !fs::equivalent(out, shard->path), "--out must not be the source GGUF");
    }
    require(options.count("--force") || !fs::exists(out), out.string() + " exists; pass --force to overwrite");

    std::set<std::string> drop;
    for (const auto & item : manifest.at("entries")) {
        drop.insert(item.value("original_gguf_tensor_name", item.at("tensor_name").get<std::string>()));
    }
    require(!drop.empty(), "The manifest has no entries");

    // The engine reads the kept tensors at gguf_add_tensor's offsets, which assume the default alignment.
    require(int_value(model.shards.front()->gguf, "general.alignment", GGUF_DEFAULT_ALIGNMENT) == GGUF_DEFAULT_ALIGNMENT,
            "GGUFs with a non-default alignment are not supported");

    std::unique_ptr<gguf_context, decltype(&gguf_free)> result(gguf_init_empty(), gguf_free);
    gguf_set_kv(result.get(), model.shards.front()->gguf);
    for (const char * key : {"split.count", "split.no", "split.tensors.count"}) gguf_remove_key(result.get(), key);

    struct kept { size_t shard; int64_t tensor; };
    std::vector<kept> keep;
    size_t dropped = 0;
    uintmax_t dropped_bytes = 0;
    for (size_t s = 0; s < model.shards.size(); ++s) {
        const gguf_file & shard = *model.shards[s];
        for (int64_t i = 0; i < gguf_get_n_tensors(shard.gguf); ++i) {
            const char * name = gguf_get_tensor_name(shard.gguf, i);
            if (drop.count(name)) {
                ++dropped;
                dropped_bytes += gguf_get_tensor_size(shard.gguf, i);
                continue;
            }
            gguf_add_tensor(result.get(), ggml_get_tensor(shard.meta, name));
            keep.push_back({s, i});
        }
    }
    require(dropped == drop.size(), "The GGUF lacks some of the manifest's expert tensors");

    std::vector<uint8_t> meta(gguf_get_meta_size(result.get()));
    gguf_get_meta_data(result.get(), meta.data());
    size_t end = meta.size();
    if (!keep.empty()) {
        const int64_t last = (int64_t) keep.size() - 1;
        end += GGML_PAD(gguf_get_tensor_offset(result.get(), last) + gguf_get_tensor_size(result.get(), last),
                        GGUF_DEFAULT_ALIGNMENT);
    }
    check_space(out.parent_path(), end);

    std::vector<std::unique_ptr<fd_file>> sources;
    for (const auto & shard : model.shards) sources.push_back(std::make_unique<fd_file>(shard->path, O_RDONLY));
    std::vector<char> buffer(chunk_bytes);
    const std::vector<char> zeros(GGUF_DEFAULT_ALIGNMENT, 0);
    {
        atomic_output output(out);
        output.write(meta.data(), meta.size());
        size_t at = meta.size();
        for (size_t j = 0; j < keep.size(); ++j) {
            const gguf_file & shard = *model.shards[keep[j].shard];
            const size_t target = meta.size() + gguf_get_tensor_offset(result.get(), (int64_t) j);
            require(target >= at && target - at < GGUF_DEFAULT_ALIGNMENT, "Unexpected tensor offset");
            output.write(zeros.data(), target - at);
            const size_t size = gguf_get_tensor_size(shard.gguf, keep[j].tensor);
            for (size_t done = 0; done < size; done += chunk_bytes) {
                const size_t n = std::min(chunk_bytes, size - done);
                read_at(sources[keep[j].shard]->fd, buffer.data(), n, shard.offset(keep[j].tensor) + done,
                        gguf_get_tensor_name(shard.gguf, keep[j].tensor));
                output.write(buffer.data(), n);
            }
            at = target + size;
        }
        output.write(zeros.data(), end - at);
        output.commit();
    }

    // Read the result back: same tensors, types, sizes and bytes.
    const gguf_file check(out);
    require(gguf_get_n_tensors(check.gguf) == (int64_t) keep.size(), "The stripped GGUF has the wrong tensor count");
    fd_file written(out, O_RDONLY);
    std::vector<char> other(chunk_bytes);
    for (size_t j = 0; j < keep.size(); ++j) {
        const gguf_file & shard = *model.shards[keep[j].shard];
        const std::string name = gguf_get_tensor_name(shard.gguf, keep[j].tensor);
        const int64_t id = gguf_find_tensor(check.gguf, name.c_str());
        require(id >= 0 && gguf_get_tensor_type(check.gguf, id) == gguf_get_tensor_type(shard.gguf, keep[j].tensor) &&
                gguf_get_tensor_size(check.gguf, id) == gguf_get_tensor_size(shard.gguf, keep[j].tensor),
                "The stripped GGUF does not match for '" + name + "'");
        const size_t size = gguf_get_tensor_size(shard.gguf, keep[j].tensor);
        for (size_t done = 0; done < size; done += chunk_bytes) {
            const size_t n = std::min(chunk_bytes, size - done);
            read_at(sources[keep[j].shard]->fd, buffer.data(), n, shard.offset(keep[j].tensor) + done, name);
            read_at(written.fd, other.data(), n, check.offset(id) + done, name);
            require(std::memcmp(buffer.data(), other.data(), n) == 0, "Byte mismatch in '" + name + "' after stripping");
        }
    }

    char summary[256];
    std::snprintf(summary, sizeof(summary), "moeshift: kept %zu tensors, dropped %zu (%.2f GB); wrote %.2f GB and verified it",
                  keep.size(), dropped, dropped_bytes / 1e9, end / 1e9);
    std::cout << summary << "\n" << out.string() << std::endl;
    return 0;
}

}  // namespace

int run_les_experts(int argc, char ** argv) {
    const std::string usage =
        "Usage: moeshift experts extract --model GGUF --out-dir DIR [--layers SPEC] [--force]\n"
        "       moeshift experts verify  --model GGUF --experts DIR [--layers SPEC] [--metadata-only]\n"
        "       moeshift experts strip   --model GGUF --experts DIR --out GGUF [--force]\n";
    if (argc < 1 || std::string(argv[0]) == "--help" || std::string(argv[0]) == "-h") {
        std::cout << usage;
        return argc < 1 ? 1 : 0;
    }
    try {
        const std::string command = argv[0];
        if (command == "extract") return extract(argc, argv);
        if (command == "verify") return verify(argc, argv);
        if (command == "strip") return strip(argc, argv);
        std::cerr << usage;
        return 1;
    } catch (const std::exception & e) {
        std::cerr << "moeshift: " << e.what() << std::endl;
        return 1;
    }
}
