#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) void * les_io_create(void);
__attribute__((visibility("default"))) void les_io_destroy(void * ticket);
__attribute__((visibility("default"))) void les_io_register(void * ticket, void * data);
__attribute__((visibility("default"))) uint64_t les_io_begin(void * ticket);
__attribute__((visibility("default"))) void les_io_signal(void * ticket, uint64_t value);
void * les_io_lookup(void * data, uint64_t * value);
void les_io_encode_wait(void * ticket, uint64_t value, void * command_buffer);

// Late prefetch: the GPU signals the CPU in the middle of a command buffer, as soon as the next
// layer's expert guess (an argsort row in host-visible memory) is computed. A thread spins on the
// event while signals are outstanding and passes each guess to the callback, in order.
typedef void (*les_late_fn)(void * user, int layer, const int32_t * ids);
__attribute__((visibility("default"))) void les_late_set_callback(les_late_fn fn, void * user);
void les_late_signal(void * command_buffer, int layer, const void * ids);
#ifdef __cplusplus
}
#endif
