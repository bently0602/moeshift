#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

inline int les_parse_protected_slots(const char * value, int maximum) {
    if (!value) return 24;
    if (!*value || *value < '0' || *value > '9')
        throw std::invalid_argument("LLAMA_MOE_PROTECTED_SLOTS must be a nonnegative integer");
    char * end = nullptr;
    const long slots = std::strtol(value, &end, 10);
    if (*end || slots > maximum)
        throw std::invalid_argument("LLAMA_MOE_PROTECTED_SLOTS exceeds the supported per-layer capacity " + std::to_string(maximum));
    return int(slots);
}

inline int les_protected_slots(int maximum) {
    const int slots = les_parse_protected_slots(std::getenv("LLAMA_MOE_PROTECTED_SLOTS"), maximum);
    if (slots > maximum)
        throw std::invalid_argument("default protected cache exceeds the per-layer capacity " + std::to_string(maximum));
    return slots;
}

// Logical ownership changes; expert bytes and tensor addresses never move.
struct les_protected_cache {
    int recent_limit = 0;
    std::vector<uint8_t> group; // 0 free, 1 recent, 2 protected
    std::vector<float> frequency;
    uint64_t steps = 0, protected_hits = 0, admissions = 0;

    bool enabled() const { return recent_limit != 0; }
    void configure(int recent, int width, int experts) {
        assert(recent > 0 && width > recent);
        recent_limit = recent;
        group.assign(width, 0);
        frequency.assign(experts, 0);
    }
    void reset(const std::vector<int32_t> & ids) {
        for (size_t i = 0; i < group.size(); ++i) group[i] = ids[i] == -2 ? 1 : 0;
        std::fill(frequency.begin(), frequency.end(), 0);
        steps = protected_hits = admissions = 0;
    }
    void begin_step(const std::vector<uint8_t> & requested) {
        assert(requested.size() == frequency.size());
        // shortcut: fixed 32-token half-life, calibrate with routing traces if workload performance requires it.
        constexpr float decay = 0.9785720621f; // 2^(-1/32)
        for (size_t e = 0; e < frequency.size(); ++e)
            frequency[e] = decay * frequency[e] + (requested[e] ? 1.0f : 0.0f);
        ++steps;
    }
    void reconcile(const std::vector<int32_t> & ids) {
        for (size_t i = 0; i < group.size(); ++i) if (ids[i] == -1) group[i] = 0;
    }
    int count(uint8_t which) const { return std::count(group.begin(), group.end(), which); }
    int victim(uint8_t which, const std::vector<int32_t> & ids,
               const std::vector<uint64_t> & ages, const std::vector<uint8_t> & requested,
               const std::vector<uint8_t> & predicted) const {
        for (int pass = 0; pass < 2; ++pass) {
            int best = -1;
            for (size_t i = 0; i < group.size(); ++i) {
                const int e = ids[i];
                if (group[i] != which || e < 0 ||
                    (!requested.empty() && requested[e]) ||
                    (!pass && !predicted.empty() && predicted[e])) continue;
                if (best < 0 || (which == 2 && frequency[e] < frequency[ids[best]]) ||
                    ((which != 2 || frequency[e] == frequency[ids[best]]) && ages[i] < ages[best])) best = int(i);
            }
            if (best >= 0) return best;
        }
        return -1;
    }
    void hit(int slot, const std::vector<int32_t> & ids,
             const std::vector<uint64_t> & ages, const std::vector<uint8_t> & requested,
             const std::vector<uint8_t> & predicted) {
        reconcile(ids);
        if (group[slot] != 2) return;
        ++protected_hits;
        if (count(1) >= recent_limit) {
            const int old = victim(1, ids, ages, requested, predicted);
            if (old < 0) return; // All recent slots are pinned by current work.
            group[old] = 2;
        }
        group[slot] = 1;
    }
    int miss(const std::vector<int32_t> & ids,
             const std::vector<uint64_t> & ages, const std::vector<uint8_t> & requested,
             const std::vector<uint8_t> & predicted) {
        reconcile(ids);
        int free = -1;
        for (size_t i = 0; i < group.size(); ++i) if (ids[i] == -1) { free = int(i); break; }
        if (count(1) < recent_limit && free >= 0) { group[free] = 1; return free; }
        const int old = victim(1, ids, ages, requested, predicted);
        // Only a multi-row step can pin every recent slot; it takes a free slot, or else the
        // coldest protected one, into the protected group.
        if (old < 0) {
            if (free >= 0) { group[free] = 2; return free; }
            return victim(2, ids, ages, requested, predicted);
        }
        if (free >= 0) {
            group[old] = 2; group[free] = 1; ++admissions;
            return free;
        }
        const int cold = victim(2, ids, ages, requested, predicted);
        if (cold >= 0 && frequency[ids[old]] >= frequency[ids[cold]]) {
            group[old] = 2; group[cold] = 1; ++admissions;
            return cold;
        }
        return old;
    }
};
