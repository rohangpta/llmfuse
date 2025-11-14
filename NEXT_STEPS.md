## Next Steps after 10K dataset training

- ✅ Regenerated clean 10K dataset (`data/train/fuse_10000_1760927775.jsonl`) and removed the prior noisy file.
- ✅ Fine-tuned `qwen3-4b` for 8 epochs on the 10K corpus; modal run `ap-a4bl04NoCHxwEg8Crdwc7U`, W&B `orcjsjn4`.
- ✅ Evaluated on new 200-sample set (`data/eval/fuse_200_1760930270.jsonl`); metrics: 82% accuracy, 68% exact match, 0.891 avg similarity (`custom_eval_qwen3-4b-sft-8epochs-distributed_20251020_032052.json`).

### Outstanding failure patterns (64 misses)

1. **`read` drift (14/20 wrong)** – wrong deterministic template, truncated docs, or tree output instead of content. Needs stronger emphasis on literal reading (maybe include file contents in prompt or add verifier).
2. **`readdir` hallucinations (10/36)** – extra filenames or truncated arrays even though the target tree is known. Consider targeted data (populated dirs) or runtime validation.
3. **State-change over-write** – `mkdir`, `create`, `write`, `rmdir`, etc. often dump entire subtrees or duplicate branches instead of minimal diffs.
4. **Format slips** – missing root header for `chmod`/`chown`, truncated code blocks after writes.

### Proposed actions

- **Generator tweaks**: include file bodies in the prompt state for `<R>` operations so `read` answers are literal copy-outs.
- **Contract enforcement**: add post-generation check (reject `readdir` outputs that reference unknown names; re-ask with guidance).
- **Targeted data**: oversample minimal-tree cases (root-only, single file) and “read-after-write” combos to reinforce literal copying; add more directory-populated readdir examples.
- **Evaluation**: expand eval to 500 samples once generator changes land; track per-op accuracy to verify regressions.
- **Optional model improvements**: try 10-epoch or qwen3-8b runs once data fixes are in to see if plateau shifts.
