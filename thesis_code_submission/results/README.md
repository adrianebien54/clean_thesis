# `results/` — recorded results behind every table and figure

These are the actual outputs of the experiments, copied unchanged from the working
tree. They make every thesis table and figure regenerable **without a GPU, without the
dataset, and without retraining** — see Path A in the main README.

The `exp/` tree keeps the directory names used during the experiments (timestamps and
all), because the eval and plot scripts look for their inputs at those exact paths. To
use it, copy it into the C³ root:

```bash
cp -r results/exp/. /path/to/C-3-Framework/exp/
```

What is **not** here: model checkpoints, TensorBoard event files, and per-run code
copies. The pruned checkpoints of the main sweep alone are 2.3 GB and the full
experiment tree is ~1.6 TB. Two evaluations therefore cannot be re-run from this
folder (`eval_bestcombo_repeats_testset.py`, `viz_density_examples.py`); their outputs
are included pre-computed.

---

| Directory | Experiment | Backs |
|---|---|---|
| `exp/360p_lr_optim_grid/csrnet_logpara100_anchor772x519_06-20_14-41/` | CSRNet optimiser × LR grid, LOG_PARA anchored at 772×519 (→ 25.05) | `tab:lr_grid`, CSRNet half |
| `exp/360p_lr_optim_grid/mobilecount_logpara2550_anchor772x519_06-17_14-26/` | MobileCount optimiser × LR grid, LOG_PARA 638.73 (`_lr1e-2` = the high-LR edge extension) | `tab:lr_grid`, MobileCount half |
| `exp/batchsize_sweep_logpara_06-21_22-41/` | Batch-size sweep, AdamW, sqrt-scaled LR. `*_merged.json` folds in the bs = 6 anchors from the LR grids | `tab:batchsize` |
| `exp/aug_paper_schemes_386x260_07-04/` | The three augmentation arms × 2 nets, under the original untuned hyperparameters. `testset_`/`trainsplit_`/`lastepoch_` are evaluations of the same runs on different splits and checkpoints | `tab:aug_paper_schemes` |
| `exp/bestcombo_resolutions_repeats_800ep/` | **The main sweep.** 2 nets × 6 resolutions × 5 same-seed repeats, 800 epochs | `tab:resolution_test`, `tab:last-checkpoint-mae` right block, `fig:acc_vs_mem` |
| `exp/bestcombo_resolutions_06-24_11-49/`<br>`exp/combined_resolution/` | The earlier single-run 1500-epoch resolution sweep and its evaluations. Retained final-epoch weights, so it is the only sweep scoreable on the train split. Also holds the inference-memory and FLOPs measurements | `tab:last-checkpoint-mae` left block, memory axis of `fig:acc_vs_mem` |

## File conventions

* `*_results.json` — one record per run, written incrementally by the sweep driver
  (which is what makes the sweeps resumable). Contains the full configuration of the
  run plus its metrics, so each file documents its own experiment.
* `testset_*` — the held-out **test** split (112 images), scored at the best-validation
  checkpoint. Never used for any selection decision.
* `trainsplit_*`, `lastepoch_*`, `best_vs_last_*` — the overfitting/drift diagnostics.
* `*_table.md` — the rendered table, as it appears in the thesis.

The largest file is `bestcombo_repeats_results.json` (24 MB): it holds the full
per-epoch validation curve for all 60 runs, which is what
`eval_bestcombo_repeats_best_vs_last.py` and the learning-curve figures read.

## A note on `exp/aa_final/`

Some result files carry a `"source"` field pointing at `exp/aa_final/…`. That directory
was renamed to `exp/360p_lr_optim_grid/` after those records were written. The scripts
in this submission have been corrected to the new name; the stale strings inside the
JSON payloads were left as-is, since they are recorded provenance rather than
resolvable paths.
