# Late prefetch

During generation, the lookahead guesses layer L+1's experts by routing layer L's FFN input through
layer L+1's router. Late prefetch makes a better guess later: at the end of layer L, it routes the
new residual through layer L+1's own norm and router. That names 82–86% of the experts layer L+1
uses with 16 guesses, against 55–58% for the lookahead's 8. The guess reaches the CPU about 350 µs
before layer L+1's step, while layer L+1's attention runs, and its reads overlap that attention.
Output does not change: prefetch only moves reads earlier.

Enable it in the model profile (Flash Next only):

```json
{
  "env": { "LLAMA_MOE_LATE_PREFETCH": 8, "LLAMA_MOE_LATE_READ": 8 }
}
```

- `LLAMA_MOE_LATE_PREFETCH` is the number of guesses per layer. They are kept from eviction, as
  the lookahead's prediction is.
- `LLAMA_MOE_LATE_READ` caps how many of them are read per layer (default: all). Only guesses not
  yet in the layer's window are read, best first.
- `LLAMA_MOE_LATE_MIX=1` makes the guess with the next layer's full hyper-connection mix: more
  accurate (91–94% with 16 guesses), but about 600 MB more bf16 weights read per token.
- For one-token steps, the late guess replaces the lookahead guess, which is no longer computed.
  Steps of several tokens (speculative checks) keep the lookahead.

## How the guess reaches the CPU

- **Graph** (`src/models/qwen4exp.cpp`): after layer L's last combine, the next layer's grouped
  norm is applied to the residual, each of the 4 streams is routed and the logits are summed (the
  same ranking as routing the stream mean). The argsort is named `les_late-<L+1>` and is a graph
  output, so its memory is not reused while the CPU reads it.
- **Metal** (`ggml-metal-ops.cpp`): after that node, the encoder ends at its next barrier, the
  command buffer signals a shared event, and a new encoder continues. Deferring the signal to the
  next barrier lets the guess run alongside the next layer's first kernels.
- **Bridge** (`bridge.h`, `bridge-cpu.mm`): a thread spins on the event while signals are
  outstanding and sleeps otherwise. It passes each guess, in order, to the expert runtime. A flag
  written by a kernel is not visible to the CPU before the command buffer ends; an event signalled
  between encoders is, about 16 µs after the guess.

## Reads

- A guess is used only while layer L's step has run and layer L+1's has not. Both steps count
  themselves under their layer's lock, and the guess checks under layer L+1's lock, so a late guess
  can never claim slots that layer L+1's step has already given to the GPU.
- Claiming a slot never evicts the other guesses, nor any expert of the layer's latest step (the
  GPU may read those until the token ends). The second rule applies to every prefetch.
- The reads are split by matrix and queued on the demand readers, best guess first.
- When layer L+1's step needs an expert that is still being read, it maps the expert to that
  slot and attaches the read to its demand batch. The expert gets a negative slot id, like a miss,
  so the GPU's missing-expert phase waits for the batch's event, while the resident experts start
  at once. Waiting on the CPU instead made generation slower: the GPU then waited for slot ids.
- Late reads are drained before prefill, multi-row steps, phase changes and parking, since they
  write decode windows and hold pending slots.

With `LLAMA_MOE_QUIET=0` the worker reports, at exit, `late prefetch: signals stale reads used
evicted-unused attached`.

Measurements are in [MACM5MAX36GB.md](../../MACM5MAX36GB.md#late-prefetch-n6a).
