#include "bridge.h"
#import <Metal/Metal.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <cassert>
struct ticket {
    id<MTLSharedEvent> event = nil;
    uint64_t value = 0;
    std::mutex mutex;
};
static std::mutex registry_mutex;
static std::unordered_map<void *, ticket *> registry;

void * les_io_create(void) {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    auto * t = new ticket;
    t->event = [device newSharedEvent];
    if (!t->event) abort();
    return t;
}
void les_io_destroy(void * handle) {
    auto * t = static_cast<ticket *>(handle);
    if (!t) return;
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto i = registry.begin(); i != registry.end();) {
        if (i->second == t) i = registry.erase(i); else ++i;
    }
    [t->event release];
    delete t;
}
void les_io_register(void * handle, void * data) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto result = registry.emplace(data, static_cast<ticket *>(handle));
    assert(result.second || result.first->second == handle);
}
uint64_t les_io_begin(void * handle) {
    auto * t = static_cast<ticket *>(handle);
    std::lock_guard<std::mutex> lock(t->mutex);
    return ++t->value;
}
void les_io_signal(void * handle, uint64_t value) {
    auto * t = static_cast<ticket *>(handle);
    t->event.signaledValue = value;
}
void * les_io_lookup(void * data, uint64_t * value) {
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    const auto it = registry.find(data);
    if (it == registry.end()) return nullptr;
    ticket * t = it->second;
    std::lock_guard<std::mutex> lock(t->mutex);
    *value = t->value;
    return t;
}
void les_io_encode_wait(void * handle, uint64_t value, void * command_buffer) {
    ticket * t = static_cast<ticket *>(handle);
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>) command_buffer;
    [cb encodeWaitForEvent:t->event value:value];
}

// Late prefetch. Value v of the event belongs to the guess recorded when v was encoded. A GPU
// signal reaches a spinning thread in about 16 us; a flag written by a kernel would not be visible
// before the command buffer ends. The thread sleeps while nothing is outstanding.
namespace {
constexpr uint64_t kLateRing = 1024;
struct late_guess { int layer; const int32_t * ids; };
id<MTLSharedEvent> late_event = nil;
std::atomic<uint64_t> late_encoded{0};
late_guess late_ring[kLateRing];
std::mutex late_wake_mutex;
std::condition_variable late_wake;
std::mutex late_cb_mutex;
les_late_fn late_fn = nullptr;
void * late_user = nullptr;

void late_spin() {
    uint64_t done = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(late_wake_mutex);
            late_wake.wait(lock, [&] { return late_encoded.load(std::memory_order_acquire) > done; });
        }
        const uint64_t seen = late_event.signaledValue;
        if (seen <= done) {
#if defined(__aarch64__)
            __asm__ volatile("yield");
#endif
            continue;
        }
        std::lock_guard<std::mutex> lock(late_cb_mutex);
        for (uint64_t v = done + 1; v <= seen; ++v) {
            const late_guess g = late_ring[v % kLateRing];
            if (late_fn) late_fn(late_user, g.layer, g.ids);
        }
        done = seen;
    }
}
}

void les_late_set_callback(les_late_fn fn, void * user) {
    std::lock_guard<std::mutex> lock(late_cb_mutex);
    late_fn = fn;
    late_user = user;
}

void les_late_signal(void * command_buffer, int layer, const void * ids) {
    static std::once_flag once;
    std::call_once(once, [] {
        late_event = [MTLCreateSystemDefaultDevice() newSharedEvent];
        std::thread(late_spin).detach();
    });
    // one guess per graph split, and splits are encoded in order, so values follow GPU order
    const uint64_t v = late_encoded.load(std::memory_order_relaxed) + 1;
    late_ring[v % kLateRing] = { layer, static_cast<const int32_t *>(ids) };
    [(id<MTLCommandBuffer>) command_buffer encodeSignalEvent:late_event value:v];
    {
        std::lock_guard<std::mutex> lock(late_wake_mutex);
        late_encoded.store(v, std::memory_order_release);
    }
    late_wake.notify_one();
}
