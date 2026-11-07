# Working on MoEShift

This repository started from `davide-troiani/moeshift`. Its goal is now the best engine for one
model on one machine:

- **Model:** Qwen 3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_XXS (GGUF plus prepared experts).
- **Machine:** Mac Studio, Apple M5 Max, 36 GB unified memory, 512 GB internal SSD.

Changes may specialize for this model and machine. Other models, chips and platforms are not
targets. Code for them may be simplified or removed when that helps the target, in its own commit.
Measurements and the current configuration are in [MACM5MAX36GB.md](MACM5MAX36GB.md).

- **Speed and quality.** Prefer changes that keep greedy output identical, and verify that against
  the previous build on real prompts. A change that alters output must say so and include a
  measured quality comparison. Do not claim benchmark improvements without data; state what was
  not tested.
- **Memory and SSD.** Keep the worker well inside 36 GB so that macOS does not swap, and avoid
  writing to the internal SSD at run time. Large traces and scratch output go to `/Volumes/Data`.
- **Licenses.** Code borrowed from other projects (for example the MIT-licensed Strata engine)
  keeps its copyright and license notice. Preserve existing licenses and notices.
- **Commits.** Commit or push only when explicitly authorized. Use separate focused commits with an
  author-matching Signed-off-by (`git commit -s`). Do not rewrite history or change repository
  visibility without explicit permission. Do not commit models, expert files, binaries or
  credentials.
- Include documentation with the corresponding changes, and match the surrounding code style.

See [configuration](configs/README.md) and the runtime documents under
[docs/les](docs/les/). Remaining upstream documentation is reference material.
