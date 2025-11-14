# Complexity Reduction Notes

## Hotspots
- Duplicated prompt-contract builders live in both `train/sft_cloud.py` and `eval/runner.py`, risking drift in formatting rules and increasing maintenance load.
- `llmfuse/llmfuse.py` couples FUSE operations, prompt construction, LLM invocation, and state parsing inside one 400+ line class, creating tightly bound logic and repeated patterns for each operation.
- `train/generate_data.py` mixes sanitization helpers, deterministic content synthesis, targeted scenario orchestration, and subprocess/FUSE control in a single module, making it hard to reason about or extend individual behaviors.
- `common/model.py` conflates backend selection, Gemini configuration, Hugging Face loading, and generation-time policy knobs, leading to broad try/except blocks and branching logic that obscure failure handling paths.
- Sanitization and normalization helpers (tree cleanup, readdir canonicalization, content stripping) are sprinkled through training, evaluation, and the FUSE runtime without a shared contract or tests.

## Refactor Themes
- Extract a shared `common/contracts.py` (or similar) that houses `_detect_operation_line`, prompt header text, and operation-specific expectations so training, evaluation, and runtime inference all import the same definitions.
- Introduce an `LLMClient` abstraction that encapsulates prompt assembly, request/response handling, retry/backoff, and telemetry; let `LLMFuse` depend on it instead of building raw prompts per method.
- Split `llmfuse/llmfuse.py` into focused collaborators (e.g., `StateParser`, `OperationDispatcher`, `DeviceHandlers`) to reduce method size, isolate parsing logic, and make state transitions testable without FUSE.
- Break `train/generate_data.py` into packages: keep orchestration thin, move deterministic content generation and sanitization into reusable modules, and carve out scenario generators so targeted edge cases remain composable.
- Wrap backend-specific logic in `common/model/` submodules (e.g., `gemini.py`, `hf.py`) behind a small interface, enabling caching, dependency injection during tests, and clearer error propagation.
- Centralize sanitization/normalization utilities and add unit coverage so every pipeline stage (data gen → fine-tune → eval → runtime) can rely on identical post-processing.

## Roadmap
1. Establish shared prompt/contract utilities and migrate training + eval + runtime code to consume them; add regression tests covering the three operation classes.
2. Carve out an `LLMClient` module and refactor `llmfuse/llmfuse.py` to delegate prompt calls and parsing; add isolated tests for state parsing and `/dev/llm` behaviors.
3. Modularize the data generator by extracting content + sanitization helpers and introducing scenario registries; ensure CLI remains stable via integration smoke tests.
4. Restructure `common/model.py` into backend-specific adapters behind a small facade; document configuration via README and add env-driven tests/stubs.
5. Set up lightweight complexity guards (e.g., `ruff` cyclomatic checks, `pytest` coverage on utilities) to prevent entropy regression once the refactors land.
