#include "../src/llama-moe-protected-cache.h"
#include <cmath>
#include <iostream>
#include <random>
#include <set>

static void test_capacity(int protected_slots) {
    les_protected_cache cache;
    cache.configure(15, 15 + protected_slots, 512);
    std::vector<int32_t> ids(15 + protected_slots, -1), slots(512, -1);
    std::vector<uint64_t> ages(15 + protected_slots, 0);
    std::mt19937 random(42);
    uint64_t age = 0;
    auto install = [&](int expert, const std::vector<uint8_t> & requested) {
        int slot = slots[expert];
        if (slot >= 0) {
            cache.hit(slot, ids, ages, requested, {});
        } else {
            slot = cache.miss(ids, ages, requested, {});
            assert(slot >= 0);
            assert(ids[slot] != -2);
            assert(ids[slot] < 0 || requested.empty() || !requested[ids[slot]]);
            if (ids[slot] >= 0) slots[ids[slot]] = -1;
            ids[slot] = expert;
            slots[expert] = slot;
        }
        ages[slot] = ++age;
        return slot;
    };
    for (int step = 0; step < 3000; ++step) {
        std::set<int> used;
        while (used.size() < 10) used.insert(random() % 100);
        std::vector<uint8_t> requested(512, 0);
        for (int expert : used) requested[expert] = 1;
        cache.begin_step(requested);
        for (int expert : used) install(expert, requested);
        for (int expert : used) assert(slots[expert] >= 0 && ids[slots[expert]] == expert);

        // A speculative writer owns this slot until publication or cancellation.
        const int pending = install(100 + random() % 400, {});
        slots[ids[pending]] = -1;
        ids[pending] = -2;
        install(511, {});
        assert(ids[pending] == -2);
        ids[pending] = -1;
        cache.reconcile(ids);
        assert(cache.count(1) <= 15 && cache.count(2) <= protected_slots);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= 0) assert(slots[ids[i]] == int(i));
            else assert(!cache.group[i]);
        }
        if (step % 257 == 0) {
            std::fill(ids.begin(), ids.end(), -1);
            std::fill(slots.begin(), slots.end(), -1);
            cache.reset(ids);
            assert(!cache.count(1) && !cache.count(2) && cache.steps == 0);
        }
    }
    cache.frequency[0] = 8;
    cache.steps = 128;
    for (int i = 0; i < 32; ++i) cache.begin_step(std::vector<uint8_t>(512, 0));
    assert(std::abs(cache.frequency[0] - 4) < 0.0001f && cache.steps == 160);
}

// A multi-row decode step pins up to rows * 10 experts; every one must get a slot.
static void test_multi_row(int rows) {
    les_protected_cache cache;
    cache.configure(20, 44, 512);
    std::vector<int32_t> ids(44, -1), slots(512, -1);
    std::vector<uint64_t> ages(44, 0);
    std::mt19937 random(7);
    uint64_t age = 0;
    for (int step = 0; step < 2000; ++step) {
        std::set<int> used;
        const size_t want = 10 + random() % (10 * rows - 9);
        while (used.size() < want) used.insert(random() % 160);
        std::vector<uint8_t> requested(512, 0);
        for (int expert : used) requested[expert] = 1;
        cache.begin_step(requested);
        for (int expert : used) {
            int slot = slots[expert];
            if (slot >= 0) {
                cache.hit(slot, ids, ages, requested, {});
            } else {
                slot = cache.miss(ids, ages, requested, {});
                assert(slot >= 0);
                assert(ids[slot] < 0 || !requested[ids[slot]]);
                if (ids[slot] >= 0) slots[ids[slot]] = -1;
                ids[slot] = expert;
                slots[expert] = slot;
            }
            ages[slot] = ++age;
        }
        for (int expert : used) assert(slots[expert] >= 0 && ids[slots[expert]] == expert);
        assert(cache.count(1) <= 20 && cache.count(2) <= 24);
    }
}

static void test_demand_lrfu() {
    les_protected_cache cache;
    cache.configure(2, 4, 6);
    std::vector<int32_t> ids = {0, 1, 2, 3};
    std::vector<uint64_t> ages = {1, 2, 3, 4};
    cache.group = {1, 1, 2, 2};
    std::vector<uint8_t> requested = {1, 0, 1, 0, 1, 0};
    cache.begin_step(requested);
    const auto scores = cache.frequency;
    assert((scores == std::vector<float>{1, 0, 1, 0, 1, 0}));
    cache.hit(0, ids, ages, requested, {}); // Recent hit.
    cache.hit(2, ids, ages, requested, {}); // Protected hit.
    // Both recent slots are pinned, as only a multi-row step can do: the coldest protected
    // slot (expert 1, older than expert 3 at equal frequency) is used and stays protected.
    assert(cache.miss(ids, ages, requested, {}) == 1 && cache.group[1] == 2);
    assert(cache.frequency == scores); // Count the entire router set once, before allocation.
    for (int i = 0; i < 200; ++i) {
        const int prefetched = cache.miss(ids, ages, {}, {});
        assert(prefetched >= 0);
        ids[prefetched] = -2;
        assert(cache.frequency == scores); // Unconsumed and cancelled prefetches earn no credit.
        ids[prefetched] = -1;
    }

    ids = {0, 1, 2, 3};
    cache.reset(ids);
    cache.group = {1, 1, 2, 2};
    requested = {0, 0, 1, 0, 0, 0};
    for (int i = 0; i < 256; ++i) cache.begin_step(requested);
    assert(cache.frequency[2] > 40);
    requested = {1, 0, 0, 0, 0, 0};
    for (int i = 0; i < 33; ++i) cache.begin_step(requested);
    assert(cache.frequency[0] > cache.frequency[2]); // New routing phase overtakes the old hot expert.
    cache.frequency[3] = cache.frequency[2] + 1;
    assert(cache.miss(ids, ages, {}, {}) == 2); // Admit the recent hot expert over stale history.

    cache.group = {1, 1, 2, 2};
    std::vector<uint8_t> predicted = {0, 0, 1, 0, 0, 0};
    assert(cache.victim(2, ids, ages, {}, predicted) == 3);
    requested = {0, 0, 1, 1, 0, 0};
    assert(cache.victim(2, ids, ages, requested, predicted) == -1);
    ids = {-2, -1, -1, -1};
    cache.reset(ids);
    assert(cache.count(1) == 1 && cache.count(2) == 0 && cache.steps == 0);
    for (float f : cache.frequency) assert(f == 0);
}

int main() {
    assert(les_parse_protected_slots(nullptr, 497) == 24);
    assert(les_parse_protected_slots("0", 497) == 0);
    assert(les_parse_protected_slots("497", 497) == 497);
    for (const char * invalid : {"", "-1", "+1", "1x", "498", "999999999999999999999999"}) {
        bool rejected = false;
        try { (void) les_parse_protected_slots(invalid, 497); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    for (int slots : {1, 23, 32, 64}) test_capacity(slots);
    test_demand_lrfu();
    for (int rows : {1, 2, 4}) test_multi_row(rows);
    std::cout << "Protected cache invariants passed at 1, 23, 32 and 64 slots; demand LRFU accounting, phase change and 1-4 row steps passed\n";
}
