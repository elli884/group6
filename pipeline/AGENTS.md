# Project Instructions

## Scope and Execution

- Work only inside `pipeline/`. Read reference documents in `../sources/`, but do not modify them or unrelated worktree changes.
- This is a university ground research replica of NAVI-Orbital. Do not describe it as flight software or claim equivalence to published benchmark numbers.
- Use Python 3.11 and `uv`. Manage optional inference/quantization/PEFT dependencies in `pyproject.toml`; commit a real generated `uv.lock` once dependency resolution is authorized.
- Initial implementation was intentionally not executed. Do not run inference, tests, downloads, dependency resolution, or installation on the current slow machine without user authorization. Static reads/reviews are permitted. Do not claim tests pass.
- If system dependencies are needed, use a Nix shell. Keep persistent Nix configuration here. Never install system packages globally or assume a shell provides a host GPU driver.
- No model/network activity at import time. All model loads must occur through `create_backend` during an explicit run.
- Use apply_patch for manual edits. Do not commit or revert other work without explicit permission.

## Commands and Paths

All commands below assume the working directory is `pipeline/`. They describe future authorized execution, not steps to run automatically now.

- First environment resolution: `uv lock`; subsequent setup: `uv sync --locked`.
- GPU setup: `uv sync --locked --extra inference --extra quantization`; add `--extra lora` only for adapter loading.
- Tests: `uv run --locked pytest`.
- Lint: `uv run --locked ruff check .`.
- AID manifest: `uv run --locked navi prepare --dataset data/AID --taxonomy aid18 --output manifests/aid.jsonl`.
- Validate without loading a model: `uv run --locked navi validate --config configs/gemma-aid.json`.
- Inference: `uv run --locked --extra inference --extra quantization navi run --config configs/gemma-aid.json`.
- Mock mission log on a normal run: `uv run --locked navi run --config configs/fake-eurosat.json --mock-mission-log Forest`.
- One image: `uv run --locked navi mock-mission-log --config configs/gemma-aid.json --image path\to\photo.jpg --label Forest`.
- One catalog log plus one image, using the config model (Gemma 3 for the AID config): `uv run --locked --extra inference --extra quantization navi run --config configs/gemma-aid.json --mission-log wheat_not_airport --image path\to\photo.jpg`.
- Comparison: `uv run --locked navi compare --clean runs/FIRST_RUN --other runs/SECOND_RUN`.
- Project-local caches: `UV_CACHE_DIR=$PWD/.cache/uv` and `HF_HOME=$PWD/.cache/huggingface`. The backend explicitly uses `.cache/huggingface` as its model cache. Never commit credentials or cache files.
- CLI `--project-dir` is a global option before the subcommand. All configured paths are relative to this directory, not a config's parent. `project_path` rejects absolute paths and escaping symlinks/traversal. Dataset functions reject all symlinks and `..` path components.

## Module Boundaries

- `config.py`: frozen Pydantic models with extra fields forbidden, schema version 1, path resolution, and configuration validation.
- `records.py`: dataclass stage contracts. `ModelInput` has only image ID, decoded image, and prompt. `ModelRequest` has only prompt, optional image, and seed. Never add ground truth or source paths to either.
- `data.py`: taxonomy mapping, deterministic content-disjoint manifest splits, checksum verification, EXIF orientation and RGB conversion. No model imports.
- `models.py`: `Backend.generate(ModelRequest) -> ModelResponse`, `metadata() -> dict`, `close() -> None`; one Transformers implementation and one synthetic fake. Own processor templates, precision, device placement, token budgets, model/adapter provenance, and accelerator measurements here.
- `parsing.py`: pure text validation; no inference and no reference labels except the allowed vocabulary.
- `detector.py`: infer, parse, retry, emit attempt callback, return final detection. Never reference-aware repair or model-driven control flow.
- `log_guard.py`: external mission-log audit only. Provenance, gold-standard class profiles, and an XML report that separates historical records from the audit verdict. Not called by dialogue or the runner, and not substituted into the dialogue prompt.
- `mission_log.py`: optional test injection. `overwrite_mission_log` replaces every identified class with one hardcoded label and a class-free description. It runs only when `navi run --mock-mission-log` or `navi mock-mission-log` asks for it.
- `mission_logs.py`: the 20 confirmed six-row logs. `navi run --mission-log CASE --image FILE` classifies that file with the config model, then sends the chosen log through the paper dialogue prompt and attaches the same image. A run without that flag never reads this catalog.
- `dialogue.py`: fixed groups in manifest order, independent scripted questions, text-only context records. No access to reference data or original images. The full classification log is inserted as written. It does not know whether that log was overwritten.
- `transforms.py`: separately named input-transform and input-defence stages, currently identity. They receive model-visible input only. This boundary is not a sandbox for arbitrary Python code.
- `runner.py`: conductor, explicit ordering, image selection, stage invocation, persistence coordination, fatal-error handling. Do not embed prompting/model internals or scoring here.
- `artifacts.py`: JSON/JSONL storage, Git/source/environment provenance, process RSS. No inference.
- `evaluation.py`: the only reasoning stage allowed to inspect reference labels. Compute metrics from saved structured results, not parsed free-form explanations.
- `cli.py`: argument parsing and command dispatch. Keep commands local and synchronous.

## Configuration Contract

Top-level fields: `schema_version`, `name`, `manifest_path`, `labels`, `split`, `seed`, `limit`, `output_dir`, `model`, `detection`, `dialogue`. Config labels are an ordered, unique vocabulary, with `__INVALID__` reserved. Seeds are uint32. Only development and explicitly authorized heldout splits are evaluated.

Model fields: backend (`transformers`/`fake`), fake label, model ID/revision, device (`cpu`/`cuda`/`mps`), dtype, quantization, local adapter path, offline loading, input/output token budgets, sampling toggle, temperature. Default fake label is Forest and must belong to the vocabulary. Default real model is `google/gemma-3-4b-it`. A mutable bootstrap revision is resolved and pinned within each load; `replay-config.json` stores the immutable commit for later runs.

Detection configuration names the template and total attempt count (1-10, default 3). Dialogue configuration names the template, enabled flag, questions, and group size. Unknown fields fail validation rather than being ignored. Add a schema version change when altering persisted semantics; do not invent migration layers without a concrete consumer.

## Data and Split Contract

- Manifest format: one JSON object per line, fields `image_id`, `path`, `sha256`, `original_label`, `label`, `split`.
- IDs are `img-` plus the first 16 lowercase SHA-256 hex digits. Full file hash is retained. Reject duplicate IDs, paths, or byte-identical content across the entire manifest before filtering by split. Detecting near-duplicates/geographic leakage is still manual.
- Supported splits are development, heldout, train, validation. Preparation shuffles sorted mapped-class groups with a seeded RNG; allocates floor(class_size * fraction) to heldout, train, validation in that order; leaves the remainder for development. Final manifest order is source path order.
- Run limits select deterministically by balanced round-robin across labels, then restore manifest order. Save the entire split manifest and the selected reference subset in every run.
- Images are loaded with EXIF transpose, converted to RGB, and stripped of metadata; multi-frame/animated inputs are rejected. Native VLM preprocessing follows. EuroSAT input must be the RGB release; no spectral compositing is implemented.
- AID reconstruction is explicitly approximate: the stated merges produce 17 observed categories, with Highway Interchange missing. The prompt profile retains 18 labels; audit missing classes and excluded source folders. Never invent samples to match published counts.
- Manifests/audits are not overwritten. Reserve training/validation/heldout pools before tuning. Never use heldout runs for prompt, attack, defence, or model selection.

## Prompt, Parser, and Retry Contract

- `prompts/detection.txt` follows paper section 4.D.2. Only `{user_label_list}` is substituted. No forced JSON/grammar generation in the baseline.
- A valid output begins with `Label:` followed by `Description:`. Allow surrounding whitespace/blank lines and paired single/double quotes around individual lines. Labels match the vocabulary exactly and case-sensitively. Descriptions can span lines and must be nonempty. Reject duplicate plain or common Markdown-prefixed field lines and leading Markdown wrappers. Unstructured description content is not sanitized or fact-checked.
- Repeat the same prompt/image on format failures only; no correction message, no sleep, no ground-truth feedback. Seed each attempt from SHA-256 of run seed, image ID, and attempt number. Record the raw response and error before retrying.
- Greedy decoding is the default reconstruction choice, not the paper's temperature sampling. Seeds do not make identical greedy retries useful; document this and use one attempt for low-cost smoke checks if desired. A stochastic experiment must explicitly enable sampling and hold that policy fixed across conditions.
- Exhausted parse retries yield an invalid result and processing continues. Infrastructure/inference errors are recorded once and abort the run. There is no wall-time watchdog and no automatic device/precision fallback.
- Dialogue uses the paper prompt from section 4.D.3, with opaque image IDs and manifest order in place of timestamps. `{classification_results}` is the full JSON log it is given. No profile filter, no XML wrapping, and no instruction that scene text is untrusted. No truncation, retrieval, history, or image reinspection. The backend rejects over-budget prompts. An external audit in `log_guard.py` may report on the same records without changing this prompt.
- Default runs pass detector records straight through. `--mock-mission-log LABEL` (bare flag uses Forest) overwrites those labels before `ask_questions`. `navi mock-mission-log --config CONFIG --image FILE --label LABEL` does the same for one file. The identified class stays in `detections.jsonl`. The prompt sees the hardcoded label. `--mission-log CASE --image FILE` classifies that file with the config model and sends one catalog log. Dialogue.py still passes no image; this switch attaches the loaded image before the backend call.

## Model Contract

- Lazy imports for torch/Transformers/PEFT. Default baseline uses Gemma3ForConditionalGeneration; other families route through AutoModelForImageTextToText, subject to model-specific verification. Encoder-decoder and remote-code models are unsupported.
- Explicit CPU float32, CUDA supported dtype, or MPS float16/float32. bitsandbytes quantization is CUDA-only. Four-bit is NF4 without double quantization, not the flight Q4_0 GGUF format. No silent CPU offload or fallback.
- Decode generated continuation only; check input token limit and native context limit without truncation. Record processor config, template hash, generation defaults/overrides, tokenizer/model commit information, package versions, device, dtype, and quantization.
- Local PEFT adapter files are hashed and loaded for inference only. Model and adapter loading request safetensors, not pickle weights. The shared adapter affects both detector and dialogue. No training, adapter interchange across architectures, or claim of training-time transfer is implemented.
- Fake output is constant and marked synthetic. Never treat it as a classifier or benchmark result.

## Results Contract

- New UUID directory per run. Persist requested config, replay config with resolved commit, provenance, model metadata, full manifest, selected references, template/preprocessing details, attempts, detections, optional dialogue, metrics, per-class CSV, and final summary.
- Attempts contain number, derived seed, exact prompt, raw response, status, parsed fields, error, elapsed seconds, and backend metadata. Final detections contain ID, status, parsed fields, and all attempts. Every attempt is flushed/fsynced to JSONL before a retry.
- Status distinguishes valid, parse_error, inference_error. Run summary distinguishes completed, failed, interrupted. Partial results are retained, not evaluated as a complete baseline. Detection metrics may exist if later dialogue failed; compare rejects any noncompleted run.
- Model-load time and whole-run time are separate. Detector timing includes preprocessing/generation/decoding/parsing and all attempts. CUDA memory is the generation-stage PyTorch allocated/reserved peak; RSS is process-lifetime high-water, excluding children. Do not label these as energy, system GPU utilization, or per-image CPU peaks.
- All-image accuracy includes invalid outputs as errors. Also report valid-only accuracy, first/final parse failures, macro/weighted F1, per-class support and confusion including an invalid column. Macro F1 includes all configured labels, even absent ones.
- Paired comparison requires identical reference records and vocabularies. Denominator for classification changes is clean-correct examples. Invalid outcomes are reported separately, not silently dropped or counted as valid misclassification. Targeted success additionally excludes examples whose true label is already the target.
- Exact description agreement is descriptive only; semantic factuality needs reviewed feature/relationship references. A clean generated caption is not ground truth. Parse validity is not hallucination absence.

## Future Research Work

Before wiring nonidentity transformations, add a validated threat model, accessible prompt/image channels, budgets, perturbation measurements, reference-preservation checks, and saved transformed artifacts. Preserve clean originals. Do not give attacks reference labels, storage handles, or arbitrary orchestration control unless explicitly part of the approved threat model. For gradient methods, introduce a backend-specific differentiable path rather than using decoded text as a gradient interface.

Keep LoRA training a separate artifact-producing workflow with recorded data splits, trigger definition, base-model revision, adapter hashes, training settings and compute cost. Compare base model, clean-finetuned adapter, and backdoored adapter with and without triggers. Evaluate defences on clean and attacked inputs. Distinguish source-generated transfer replay from target-specific optimization. Add uncertainty estimates for main experiments; current comparisons are descriptive, not statistical significance tests.

## Verification Still Required

Resolve and lock dependencies on approved hardware; run Ruff and fake-only pytest; fix any discovered API/style issues. Then run one real Gemma image, text-only dialogue, and two fixed-set baselines. Check CUDA/processor/PEFT behavior and actual peak memory before scaling. Smoke-test a second backbone before the attack campaign. Have a teammate reproduce one saved result from the README. Until then report the implementation as statically reviewed, not runtime validated.
