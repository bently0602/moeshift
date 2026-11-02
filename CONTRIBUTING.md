# Contributing to MoEShift

MoEShift is maintained by Davide Troiani. The alpha focuses on local inference
on Macs with Apple Silicon; see the [supported LLM architectures](docs/architecture.md#supported-llm-architectures).

## Issues and pull requests

- Search [existing issues](https://github.com/davide-troiani/moeshift/issues).
  Use the issue forms for bugs, suggestions and benchmark results.
- Open an issue before substantial changes so we can agree on scope. Small fixes
  can go straight to a pull request.
- Keep each PR focused. Describe the problem, the change and how you tested it.
  Include Mac hardware and configuration for performance or memory claims.
- Follow nearby code style, reuse existing components and discuss new
  dependencies first. Include a regression check when behavior changes.
- Review and understand everything you submit. Disclose substantial AI
  assistance in the PR description and take responsibility for the result.

There is no required reviewer count or release schedule. The maintainer reviews
and merges changes. Build outputs, model weights, generated expert files,
credentials and personal configurations must stay out of commits.

## Share benchmarks

Follow the [community benchmark instructions](benchmarks/README.md) and attach
the JSON report to a **Benchmark result** issue. Different Macs, battery use and
other applications running alongside inference are welcome: describe your
hardware, configuration and conditions manually. The runner does not collect
system or personal information. Screenshots are optional.

## Build and test

See [building from source](docs/build.md). Use the checks relevant to your change;
the native tests run through CTest. For router changes:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/test-moeshift-router.py build/bin/moeshift
```

For runtime changes, also test with a real model when possible. State which
checks you ran and which require hardware or files you do not have.

## Commits and license

Use concise English commit subjects and separate unrelated changes. Sign off
each commit with `git commit -s` to certify that you have the right to contribute
it under the project's MIT license. The sign-off must match the author identity.
Preserve existing copyright and third-party license notices.

For security reports, see [SECURITY.md](SECURITY.md).
