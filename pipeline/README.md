# NAVI Research Pipeline

A local ground replica for studying classification and scene-description failures
in Earth-observation VLMs. Python + `uv`, with one model loaded at a time and an
explicit batch workflow. It is **not** flight software or an exact reproduction of
the paper's benchmark.

**Status:** implemented but not executed or dependency-resolved on this machine.
No models or datasets have been downloaded. Tests and GPU compatibility are
unverified. The commands below are for a suitable machine when you are ready.

## Hardware

These are planning estimates, not measured requirements. Image patching, context
length, output length, precision, and library versions all affect memory use.

| Work | Suggested hardware |
| --- | --- |
| Unit tests / fake backend / dataset preparation | Ordinary CPU, 8 GB RAM; no GPU or model weights |
| Gemma 3 4B, 4-bit, batch size 1 | NVIDIA CUDA GPU with 12-16 GB VRAM, 32 GB system RAM; begin with one image and a short context |
| Recommended shared inference machine | NVIDIA GPU with 24 GB VRAM (e.g. RTX 3090/4090 or A5000), 32-64 GB RAM, 100 GB free SSD |
| Later QLoRA experiments | Start planning for 24-48 GB VRAM and 64 GB RAM; validate batch size, image tokens, and checkpointing first |

An 8 GB GPU might support reduced settings, but do not assume it fits the supplied
20,000-token input budget. CPU float32 inference needs substantial RAM and will
be slow; it is not recommended for the experiment campaign. MPS has an explicit
unquantized profile but is untested. Use a compatible host NVIDIA driver; a Nix
shell cannot supply or repair the host GPU driver. No latency estimates are yet
available.

## Setup

Run all commands **from `pipeline/`**. Keep datasets, weights, caches, and results
here; generated directories are ignored by Git. Python 3.11 is the default.

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv lock
uv sync --locked --extra inference --extra quantization
```

There is intentionally no invented `uv.lock`: no resolver was run during this
implementation. On the first target machine, resolve dependencies, validate the
environment, and commit the resulting `uv.lock`. Thereafter use `--locked`.
Choose a PyTorch wheel/index compatible with that machine's CUDA environment
before freezing the environment. Do not silently switch devices or precision.
For tests, manifest work, or the fake backend, `uv sync --locked` alone suffices.

If system tools are missing, enter a Nix shell, for example
`nix shell nixpkgs#uv nixpkgs#git`. Add platform-specific system libraries through
a Nix shell as needed, not global package installation. Native GPU libraries on
NixOS may require additional host integration; no universal GPU shell is claimed.

Accept the Gemma model terms on Hugging Face and provide `HF_TOKEN` securely for
the initial download. Do not put credentials in configs. Download explicitly:

```sh
uv run --locked --extra inference --extra quantization python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-3-4b-it', revision='main', cache_dir='.cache/huggingface'))"
```

Replace `model.revision` in the experiment config with the commit hash printed in
the snapshot path. Downloading may require substantially more disk than the
eventual quantized model's VRAM. Runtime model loading is offline by default.

## Prepare Data

Obtain AID or **EuroSAT RGB** separately, respecting their terms. Put the original
class folders directly under `data/AID/` or `data/EuroSAT/`. No automatic downloads
or multispectral-to-RGB conversion are provided.

```sh
uv run --locked navi prepare --dataset data/AID --taxonomy aid18 --output manifests/aid.jsonl
uv run --locked navi validate --config configs/gemma-aid.json
```

Alternatively:

```sh
uv run --locked navi prepare --dataset data/EuroSAT --taxonomy eurosat --output manifests/eurosat.jsonl
uv run --locked navi validate --config configs/gemma-eurosat.json
```

Preparation creates a SHA-256 manifest and adjacent `.audit.json`. It preserves
original labels, deterministically stratifies by mapped class, and reserves 20%
per class for held-out evaluation (rounded down). Optional `--train-fraction` and
`--validation-fraction` reserve disjoint pools for later training. Freeze the
manifest; do not regenerate splits while tuning. Tiny classes can have no held-out
examples: inspect the audit. Byte-identical duplicates are rejected; geographic
overlap and near-duplicate imagery still need human review.

**AID caveat:** the paper's stated merges and exclusions yield 17 classes from
canonical AID, not its reported 18. No canonical folder supplies `Highway
Interchange`. The `aid18` profile retains the paper's 18-label prompt vocabulary
and reports missing classes; it does not fabricate examples. The inferred mapping
and exclusions are documented in `data.py` and the audit. The paper's 88.16%
accuracy is not an acceptance threshold for this reconstruction.

`--taxonomy folders` supports other class-folder datasets. Set the configuration's
labels to the audit's observed labels. If using the fake backend, also set
`model.fake_label` to one permitted label; it is a fixed synthetic response, not a
classifier. All paths in configs are relative to `pipeline/`, not the config file.

## Run Later

Start with `limit: 1`, optionally reduce `max_new_tokens`, and measure fit before
running a larger batch. Freeze those settings before comparisons. The supplied
AID config selects up to 36 images and enables one scripted dialogue question;
EuroSAT selects up to 20 and disables dialogue. A limit uses seeded, balanced
round-robin selection, then restores manifest order.

```sh
uv run --locked --extra inference --extra quantization navi run --config configs/gemma-aid.json
uv run --locked --extra inference --extra quantization navi run --config configs/gemma-aid.json
uv run --locked navi compare --clean runs/FIRST_RUN --other runs/SECOND_RUN
```

Use the two actual run paths printed by the commands. Each run gets a new
directory. A saved `replay-config.json` pins the resolved model commit and can be
passed to `--config` for replay in the same project. Model identity alone is not
enough: also preserve `uv.lock`, code, prompts, data, adapter files, and hardware
details. Seeds do not guarantee bitwise equality across devices or libraries.

For a no-model plumbing check after preparing EuroSAT:

```sh
uv run --locked navi run --config configs/fake-eurosat.json
```

Synthetic runs are marked in model metadata and metrics; never report their
accuracy as VLM performance. To evaluate the held-out set later, make a separate
config with `split: "heldout"` and pass `--allow-heldout`. The runner does not
evaluate training/validation splits. It never silently resumes or overwrites a run.

## Architecture

```text
config + manifest
  -> validation and fixed image selection
  -> model loading
  -> image decoding -> input transform -> input defence
  -> detector inference -> parse gate -> bounded retries
  -> persisted detection records
  -> optional grouped, text-only dialogue
  -> metrics and run summary
```

| Module | Responsibility |
| --- | --- |
| `config.py`, `records.py` | Validated configuration and stage contracts |
| `data.py` | Dataset mapping, splits, hashes, image decoding |
| `models.py` | Lazy Transformers backend, local PEFT adapter loading, fake backend |
| `parsing.py`, `detector.py` | Label/description parser and inference attempts |
| `dialogue.py` | Scripted questions over detector records, not original images |
| `transforms.py` | Separate identity input-transform and defence stages |
| `runner.py` | Stage ordering, bounded batch execution, failure handling |
| `artifacts.py`, `evaluation.py` | Persistence/provenance and reference-aware metrics |
| `cli.py` | Local command interface only |

Ground-truth labels and source paths stay out of model requests. Dialogue uses
opaque image IDs and final detection records. The same VLM serves both stages,
without shared conversational memory. Its native processor/chat template is
recorded. Detector output is ordinary text, not constrained JSON generation.

The parser accepts plain or individually quoted fields, exact case-sensitive
labels, and multiline descriptions. It rejects missing/duplicate fields and
leading Markdown wrappers. It does not validate factuality or sanitize arbitrary
description text. Three **total attempts** are allowed by default, without sleep
or prompt repair; every raw response is saved. Inference errors abort the run,
whereas exhausted format retries produce an invalid detection and continue.

Default decoding is greedy for a repeatable initial baseline, unlike the paper's
temperature-0.2 sampling. Repeating an identical greedy request is normally
ineffective at recovering a format failure; set `max_attempts: 1` for cheaper
smoke tests. For a stochastic profile, set `do_sample: true`; temperature 0.2 then
applies. Retry seeds are deterministic and distinct. Keep the chosen policy fixed
across paired conditions. Token budgets bound generation length, not wall-clock
execution; there is no worker watchdog.

Deliberately omitted: spacecraft queues, FastAPI, hot reload, persistent service
loops, LangGraph, interactive chat, distributed inference, and flight telemetry.
Keep the state transitions and data boundaries, not the deployment infrastructure.

## Results and Research Rules

Each run contains `config.json`, `replay-config.json`, `provenance.json`,
`model.json`, frozen manifest/references, `inputs.json`, `attempts.jsonl`,
`detections.jsonl`, optional `dialogue.jsonl`, `metrics.json`, `per_class.csv`, and
`summary.json`. Interrupted/failed runs may contain only a subset; traces are
flushed after every attempt. A hard kill may leave no summary.

Metrics include all-example accuracy, valid-only accuracy, macro/weighted F1,
per-class scores, an invalid-output confusion column, first/final parse failures,
attempt counts, and mean/median detector latency. Macro F1 includes every configured
class, including zero-support classes. CUDA peaks measure PyTorch allocated/reserved
memory during generation; RSS is the process-lifetime high-water mark. These are
not total GPU occupancy, per-image CPU peaks, energy, or spacecraft performance.

`compare` requires identical references and reports label agreement, exact
description agreement, valid misclassification rate on clean-correct examples,
and invalid outcomes separately. `--target-label LABEL` adds targeted counts,
excluding examples already belonging to that target. A comparison is not
automatically an attack-success measurement. Text differences are not hallucination
or semantic-correctness measures.

Before implementing an attack, agree its threat model, accessible channels,
strength, query/compute budget, and reference-preservation checks with supervisors.
Add explicit artifact/budget validation at the transform boundary. Current
transforms are identity only; attacks, defences, gradient access, LoRA training,
and human-reviewed description scoring are **not implemented**. Local PEFT adapter
loading is available via `model.adapter_path` and `--extra lora`, but training
needs a separate workflow. Adapter directories must include `adapter_config.json`
and `adapter_model.safetensors`. A second Transformers VLM family can use the backend
interface, but has not been selected or verified.

Later experiments should compare clean/attacked/defended-clean/defended-attacked
conditions, record training and attack cost, distinguish replay transfer from
target-model re-optimization, and reserve held-out images until final testing.
AID/EuroSAT labels are references for classification, not authoritative captions:
create human-reviewed feature/relationship annotations for descriptions. Do not
claim a clean model output is ground truth or extrapolate this ground replica to
actual spacecraft safety.

## Running Implemented Attacks and Defences

Visual prompt injection:
For the patch attack the following command can be run.
NAVI_ATTACK_PATCH='{"shape":"square","size":0.2,"x":0.5,"y":0.5,"color":"random"}' \
  uv run python -m navi_pipeline.cli run --config configs/gemma-aid.json
The shape can be a square, rectangle, circle or ellipse. The size is written as a fraction of the image width and the x and y positions refer to the center of the patch written as a fraction of image width and height respectively. The color can either be randomly selected or written in the form [r,g,b] with values 0-255.

The extended text attack can be run in two modes using the following commands:
Faint/low-amplitude text (recommended variant of the two)
NAVI_ATTACK_STEGO_MODE=faint NAVI_ATTACK_STEGO_TEXT=Forest NAVI_ATTACK_STEGO_ALPHA=0.08 \
  uv run python -m navi_pipeline.cli run --config configs/gemma-aid.json

True LSB (negative control — no effect expected due to image resizing when running the model)
NAVI_ATTACK_STEGO_MODE=lsb NAVI_ATTACK_STEGO_TEXT=Forest \
  uv run python -m navi_pipeline.cli run --config configs/gemma-aid.json

## Development Checks

These commands are documented for later; none were run during implementation.
Tests use synthetic images/scripted backends and do not download or load a VLM.

```sh
uv run --locked pytest
uv run --locked ruff check .
```

Exact development contracts and remaining verification work are in `AGENTS.md`.
