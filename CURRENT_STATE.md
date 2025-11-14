# LLM-FUSE – Current Checkpoint (Updated November 12, 2025)

## TL;DR
- **Model:** `qwen3-4b-sft-8epochs-distributed` (trained Nov 11 on 15 k XML + `<END_FS>` corpus)
- **Datasets:**
  - Train: `data/train/fuse_15000_1762901698.jsonl` (state completions end with `<END_FS>`)
  - Eval: `data/eval/fuse_100_1762901721.jsonl`, `data/eval/fuse_200_1762901737.jsonl`
- **Eval (vLLM, 8 k context, stop=`<END_FS>`):**
  - 100-case: **98.0 % accuracy / 90.0 % exact / 0.982 avg sim** (`custom_eval_…_20251112_025213.json`)
  - 200-case: **99.5 % accuracy / 88.5 % exact / 0.995 avg sim** (`custom_eval_…_20251112_025836.json`)
- **Residual misses:** 8 (100-case) / 22 (200-case), all due to real model mistakes (ordering, body truncation) – no more XML truncation artifacts.

## What Changed Since the Oct 19 snapshot
1. **XML Contract Hardening**
   - Added deterministic `<filesystem>…</filesystem>` serialization everywhere plus the `<END_FS>` sentinel.
   - Evaluator, training prompts, Modal runtime, and local inference now all clamp generation to that sentinel.
2. **Data Regeneration**
   - Rebuilt the 15 k train corpus and both eval splits with canonical mtimes + end markers.
3. **Longer Context + Stop Tokens**
   - vLLM + transformers evals run with 8 192 token context and stop on `<END_FS>`; Modal inference service mirrors this and strips the sentinel before returning trees.
4. **Training Refresh**
   - Reran 8-epoch SFT on the regenerated corpus (Modal app `ap-K2WOxKvrsakYq9IpOaCdzE`, W&B run `vgsnspz6`).
5. **Sanitization Fixes**
   - `_strip_chatter` no longer splits on ``` fences, eliminating the self-inflicted XML truncation.

## Current Artifacts
| Artifact | Path / ID | Notes |
| --- | --- | --- |
| Train dataset | `data/train/fuse_15000_1762901698.jsonl` | 15 k mixed ops, `<END_FS>` appended to all state completions |
| Eval datasets | `data/eval/fuse_100_1762901721.jsonl`, `data/eval/fuse_200_1762901737.jsonl` | Canonical XML + sentinel |
| Model checkpoint | `/root/output_models/qwen3-4b-sft-8epochs-distributed` | Latest weights on Modal volume |
| Eval reports | `eval_results/custom_eval_qwen3-4b-sft-8epochs-distributed_20251112_025213.json` (100), `_20251112_025836.json` (200) | Pulled locally |
| Modal eval apps | `ap-dYie0i1LbswWQMct8CjuoL` (100), `ap-8e6czZVsb8tUkmCg51WGdu` (200) | Both detached runs succeeded |

## Failure Breakdown (post-fix)
- **Ordering_diff:** majority of remaining misses; model emits correct files but reorders siblings (esp. after create/rename).
- **Body_diff:** truncate/write ops where file bodies are not resized correctly (8 cases on 200-set, 2 cases on 100-set).
- **Attr_diff:** a couple of create/symlink ops where metadata (size/path) isn’t updated.
- **Path_diff:** one rename sample forgot to drop the source directory.
> No more invalid XML / truncated outputs – all failures now reflect real modeling gaps.

## Open TODOs
1. **LLMFUSE Runtime**
   - Wire the trained model into `llmfuse.py` via the Modal inference endpoint (or local `common/model`) so we can mount the filesystem end-to-end.
   - Requires Docker/Linux environment for FUSE (mac host can only drive via container).
2. **Ordering + Truncate Accuracy**
   - Consider data augmentations / decoding constraints to keep insertion order deterministic and enforce byte-accurate truncations.
3. **Automated XML Regression Tests**
   - Snapshot a few XML trees + ensure sanitizer + inference never drop terminal tags again.

## Next Focus (per latest instruction)
- **Implement mountable FUSE backed by the new model:**
  1. Reuse the inference plumbing from `eval/runner.py` / `infra/modal_llmfuse.py` inside `llmfuse.py` so each FUSE op hits the Modal service and enforces `<END_FS>`.
  2. Build/run the `Dockerfile.llmfuse` helper to mount inside Linux, since macOS FUSE is unavailable.
  3. Validate basic ops (`mkdir`, `write`, `read`, `rename`) on the mounted path and compare against reference state.
- All code + datasets already live in repo; only missing piece is the runtime wiring + mount test.

