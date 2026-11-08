# Speculative decoding

The server's speculative decoding (`--spec-type`) drafts a few tokens and checks them with one
pass of the target model. Under LES, that check is one decode step of several rows. Enable it in
the model profile:

```json
{
  "args": { "spec-type": "ngram-simple" },
  "env": { "LLAMA_MOE_DECODE_ROWS": 4 }
}
```

- `env.LLAMA_MOE_DECODE_ROWS` sets how many tokens one decode step may carry. Every expert of
  every row must be resident at once, so the narrowest per-layer window bounds it: with a 44-slot
  window and 10 experts per token, the limit is 4. Speculation needs at least 2; with 1 it is
  disabled with a message. Flash Next's profile has no protected slots (15–20-slot windows, so 1–2
  rows); speculation needs `LLAMA_MOE_PROTECTED_SLOTS` back above 0.
- Drafts are capped at `rows - 1` tokens (`LLAMA_SPEC_DRAFT_CAP`, set by the worker), so a check
  never leaves the decode phase. n-gram drafts, which ignore `spec-draft-n-max`, are truncated.
- A batch of several tokens counts as decode only while the decode phase is active. The last
  tokens of a prompt still use the prefill phase. The context, the graph and the expert runtime
  share this decision (`llama_moe_slot_runtime::decode_step`).
- The profile also accepts `spec-draft-model` (resolved like `model`), `spec-draft-n-max`,
  `spec-draft-n-min` and `spec-draft-ngl`.

## Rollback

A rejected draft is undone from recurrent-state snapshots (`n_rs_seq`, one per draft token)
instead of restoring a checkpoint and re-evaluating the accepted tokens. Every drafting type
requests them (`common_params_speculative::need_n_rs_seq`).

- A ubatch of n > 1 tokens writes snapshots for its last `min(n, n_rs_seq)` tokens and can be
  rolled back by that many (`llama_memory_recurrent::rs_valid`).
- A single-token ubatch carries no draft. It writes no snapshots and cannot be rolled back, so
  ordinary decode steps cost the same as without rollback. Writing all snapshots on every step
  made generation about 15% slower.
- A rollback is refused when the last ubatch's snapshots cannot serve it, or after a previous
  rollback, instead of reading a stale snapshot.

A recurrent draft model (for example Qwen3.5) gets snapshots as well, because the server removes
rejected tokens from the draft context after each check. Its own drafted tokens are single-token
steps, so they are undone with the drafting checkpoint instead.

## Expert windows

A multi-row step can pin every recent slot of the protected cache; its remaining misses then take
a free or protected slot (see [protected decode cache](protected-decode-cache.md)). Before the
step, the runtime waits for the layer's pending prefetch reads. Prefetch guesses from all rows are
merged rank by rank. In the Metal graph, the resident/missing expert split applies to every decode
step, so each row waits only for its own missing experts.

On this machine, n-gram drafting pays off only on text that repeats its context. A separate draft
model was slower: each drafted row reads its own experts, and rejected rows waste those reads.

Measurements are in [MACM5MAX36GB.md](../../MACM5MAX36GB.md#speculative-decoding-phases-a-and-b).
