# Community benchmarks

Run the same LLM requests on your Mac and share the results through the
[Benchmark result issue form](https://github.com/davide-troiani/moeshift/issues/new?template=benchmark_result.yml).
Keep your normal working conditions: other apps, battery operation, Low Power
Mode and different Mac models are all useful contexts. Describe them manually
in the issue; the runner collects no system or personal information.

## Run the benchmark

Requires Python 3 (standard library only), an installed or built MoEShift, and a
working model configuration. Download or clone this repository and open Terminal
in its root folder. Set the `model` field in both `short.json` and `long.json` to your configured
model ID. Compare results for the same model and quantization; the historical
Flash Next/Strata comparison in the main README uses a separate 256-output-token run. The configured context must accommodate the long prompt plus
512 output tokens; `args.ctx-size` of at least `8192` is recommended.

Stop any running MoEShift instance and disconnect other inference clients, then:

```sh
python3 benchmarks/run.py
```

For a source build or a specific configuration:

```sh
python3 benchmarks/run.py --binary ./build/bin/moeshift --config /path/to/config.json
```

The runner uses normal MoEShift configuration discovery unless `--config` is
provided. It uses localhost port 8080, overriding the configured port; choose
another with `--port 8081`. Runtime tuning remains in your JSON configuration.
The runner refuses an occupied port and only stops processes it started.
Server output stays in Terminal and is not included in the report.

For each prompt, it starts a fresh server, waits for `/health`, runs one warmup
and three measured requests sequentially, then stops the server and worker.
The router loads the model lazily: the first warmup includes loading in its
wall time. Warmups are saved separately and excluded from all medians.
The runner allows 60 seconds for router startup and up to 600 seconds per HTTP
request. Failures or Ctrl-C stop the owned processes without creating a report.

Results are saved to `benchmarks/benchmark-results.json`. Existing reports are
never overwritten; use `--output benchmarks/benchmark-results-2.json` for another
session. No upload occurs automatically. Review the JSON, then attach it to an
issue. Generated reports in this folder are ignored by Git.

## Fixed requests and measurements

Both curl examples and the runner read the exact same saved payloads:

| Payload | Prompt content | Output limit |
|---|---|---|
| [short.json](short.json) | About 50–60 tokens; everyday local inference | 512 tokens |
| [long.json](long.json) | About 6,000 tokens; synthetic workload observations | 512 tokens |

A local check with the repository's `models/ggml-vocab-qwen35.gguf` vocabulary
counts 60 and 5,994 content tokens respectively, excluding chat-template tokens.
This is a vocabulary reference, not a verification with the full Qwen 3.8 model.
The server's `usage.prompt_tokens` records the actual templated input count;
counts can differ with another tokenizer or chat template.

Requests fix `temperature: 0`, `seed: 42`, `stream: false`,
`cache_prompt: false`, `max_tokens: 512` and
`chat_template_kwargs.enable_thinking: false`. Do not edit the prompts or
sampling settings for comparable reports. Each report includes the protocol
identifier and SHA-256 of each payload. Include the repository revision manually
in the issue so the runner version is also traceable.

512 is a maximum: a reply can end earlier. Every run preserves actual input and
output token counts and its finish reason. Compare runs with similar output
lengths. Fixed sampling does not guarantee identical output across different
runtime versions or hardware.

The report contains only allowlisted benchmark fields:

- `timings.prompt_ms` and `prompt_per_second`: server prompt-processing metrics.
- `timings.predicted_ms` and `predicted_per_second`: server generation metrics.
- `inference_ms`: sum of server prompt and generation times, excluding startup.
- `wall_ms`: client elapsed time for the complete HTTP request.
- `usage`, `finish_reason` and `timings.cache_n`: actual token counts, termination
  and prompt-cache use. Cached prompts, missing metrics and failed requests are
  rejected rather than summarized as valid results.
- Individual warmup and measured results, plus medians of the three measured
  runs. Speed medians are calculated from individual reported speeds.

No hardware, power state, swap, process list, configuration, file paths, raw
responses or generated text are collected in the report. Describe hardware,
software versions, model source, configuration changes and external conditions
manually in the issue. Swap is system-wide; if you choose to report it, distinguish
before/after readings from memory attributable to MoEShift.

Prompt caching is disabled, but expert caches and macOS file caches remain active.
These are warm-run measurements under your stated conditions, not cold-storage
benchmarks. Other applications can stay open; avoid other requests to MoEShift
while measuring so queue time does not distort the results. This protocol differs
from the historical 256-output-token comparison in the main README.

## Manual curl alternative

With MoEShift already running on port 8080, each script makes one request:

```sh
sh benchmarks/short.sh > /tmp/moeshift-short.json
sh benchmarks/long.sh > /tmp/moeshift-long.json
```

For the same procedure, restart MoEShift before each prompt, run that prompt once
as warmup, then three more times, saving each response separately. Exclude the
warmup when calculating medians. Adapt the URL in the scripts if your port differs.
Curl responses include generated text and server metadata; review them before
sharing. The Python report is the simpler attachment for the issue form.

## Runner checks

No model is required for the fake-server integration tests:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/test-benchmarks.py
```
