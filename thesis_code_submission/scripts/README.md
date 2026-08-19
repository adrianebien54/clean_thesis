# `scripts/` — index

64 scripts, grouped by pipeline stage. Every one carries a docstring stating what it
does, what it reads and what it writes; the one-liners below are those docstrings'
first lines.

Run them from the C³ root with `python scripts/<group>/<name>.py`. Most accept the
environment overrides listed in Section 4 of the main README (`SWEEP_DIR`,
`MAX_EPOCH`, `ONLY_NET`, `RES`, `REPS`).

Naming convention: `run_*`/`sweep_*` train, `eval_*`/`measure_*` score trained
checkpoints, `merge_*`/`combine_*` consolidate result JSONs into tables, `plot_*`/`viz_*`
draw figures.

---

## `data_prep/` — building the dataset

Section 3 of the main README gives the order and the arguments.

| Script | |
|---|---|
| `split_tenebrio.py` | Split TenebrioVision_Images into train/val/test folders (count-stratified, 840/168/112, seed 42). |
| `resize_splits_to_resolutions.py` | Resize base train/val/test images into each resolution sub-folder. |
| `precompute_tenebrio_densities_hdf5.py` | Precompute Tenebrio ground-truth density maps as HDF5 files, per resolution variant. |
| `make_csrnet9_patches_386x260.py` | Build the offline CSRNet-paper patch dataset from the 386x260 Tenebrio train split. |
| `verify_train.py` | Smoke-test the training loop end-to-end against a random in-memory batch. |

## `tuning/` — the three tuning sweeps → `tab:lr_grid`, `tab:batchsize`, `tab:aug_paper_schemes`

All at 386×260, seed 3035, 1500 epochs. Hyperparameters were tuned at this one
resolution and then reused across all six.

| Script | |
|---|---|
| `sweep_csrnet_logpara_anchor772x519.py` | Re-run the CSRNet optimizer × LR grid with a smaller density scale. → `tab:lr_grid`, CSRNet half |
| `sweep_mobilecount_logpara2550.py` | Re-run all MobileCount cells with a larger density scale. → `tab:lr_grid`, MobileCount half |
| `sweep_batchsize_logpara.py` | Batch-size sweep at each net's per-model LOG_PARA (AdamW only). |
| `merge_batchsize_logpara_results.py` | Consolidate the batch-size sweep into one final table. → `tab:batchsize` |
| `run_aug_paper_schemes_386x260.py` | Paper-augmentation comparison under the ORIGINAL (untuned) hyperparameters. |
| `eval_aug_paper_schemes_testset.py` | Score the aug runs' best-val-MAE checkpoints on the TEST split. → `tab:aug_paper_schemes` |
| `eval_aug_paper_schemes_lastepoch.py` | Score the 6 last-epoch checkpoints on the TRAIN split. |
| `eval_aug_paper_schemes_trainsplit.py` | Score the 6 selected checkpoints on the TRAIN split (840 full images). |
| `eval_aug_paper_schemes_trainsplit_772x519.py` | The same, at 772×519. |

The last three support the argument that the augmented arms degrade on train and
validation *together* — i.e. the drift is not memorisation.

## `resolution/` — the cross-resolution study → `tab:resolution_test`, `tab:e4_budget`, `fig:allres_trend`, `fig:acc_vs_mem`

Two sweeps live here. The **repeats** sweep (5 same-seed repeats × 6 resolutions ×
2 nets, 800 epochs) produces the headline numbers. The earlier **single** 1500-epoch
sweep is still cited: it is the only one whose final-epoch weights were kept, so it is
the only one that can be scored on the training split.

| Script | |
|---|---|
| `run_bestcombo_resolutions_repeats.py` | 5× same-seed repeats of the best-combo cross-resolution comparison (800 epochs). **The main sweep.** |
| `eval_bestcombo_repeats_testset.py` | Evaluate every repeat on the TEST split; report mean ± std. → `tab:resolution_test` |
| `eval_bestcombo_repeats_best_vs_last.py` | Best-val vs end-of-budget validation MAE per repeat (validation-only companion; not in a thesis table). |
| `run_best_combo_resolutions.py` | The earlier single-run sweep, 1500 epochs. |
| `finish_bestcombo_resolutions_1544.py` | Finish that sweep's two interrupted 1544×1038 cells. |
| `combine_resolution_results.py` | Fold the 386×260 tuning-era point into the single-run sweep. |
| `eval_bestcombo_resolutions_testset.py` | Score the single-run sweep on the TEST split. |
| `eval_bestcombo_best_vs_last_trainsplit.py` | Best vs last checkpoint on the TRAIN split (single 1500-ep run). → `tab:e4_budget` |
| `measure_inference_memory.py` | Peak GPU memory of batch-1 inference per net × resolution. → `fig:acc_vs_mem` |
| `measure_flops_time.py` | FLOPs and batch-1 inference time per net × resolution. Measured, but the thesis quotes published FLOPs instead. |

## `figures/`

The first three rows produce the three figures in the current draft. The rest are
analysis figures from earlier drafts, kept because they are what the corresponding
prose was written against.

| Script | |
|---|---|
| `plot_test_error_vs_resolution_repeats.py` | Test MAE/MSE vs resolution, mean ± std over the repeats. → **`fig:allres_trend`** |
| `plot_mae_vs_memory_repeats.py` | Accuracy-vs-memory trade-off, from the repeat sweep. → **`fig:acc_vs_mem`** |
| `viz_density_examples.py` | Density-map grids for example plates; `--all 80_18` builds the appendix figure. → **`fig:density_maps`** |
| `rank_net_disagreement.py` | Rank TEST images by CSRNet/MobileCount disagreement — picks which image the above should render. |
| `plot_mae_vs_resolution_repeats.py` | MAE-vs-resolution with error bars from the repeat sweep. |
| `plot_resolution_mae_curves_thesis_2x3.py` | Per-resolution val-MAE learning curves, 2×3 grid. |
| `plot_aug_paper_schemes.py` | Val curves + train→val→test slope chart for the augmentation arms. |
| `plot_aug_best_vs_last.py` | Best-checkpoint vs last-epoch MAE (train and val), all three arms. |
| `plot_aug_train_vs_val.py` | Train-vs-validation MAE of the selected checkpoints, aug arms only. |
| `plot_overfit_movement_map.py` | Overfitting-vs-instability movement map for the resolution study. |
| `plot_overfit_3panel_merged.py` | The two 1×3 overfit figures, stacked. |
| `plot_mae_vs_memory.py` | Accuracy-vs-memory from the single 1500-epoch run (superseded by the repeats version). |
| `plot_mae_vs_resolution.py` | MAE-vs-resolution from the single run (likewise). |
| `plot_resolution_mae_curves.py` | Per-epoch val-MAE curves, faceted by resolution. |
| `plot_resolution_loss.py` | Per-epoch validation-*loss* curves across all 6 resolutions. |
| `plot_lr_grid_curves.py` | Val-MAE curves for the full optimizer × LR grid. |
| `plot_csrnet_logpara_anchor772x519.py` | Curves for the CSRNet LOG_PARA=25.05 LR-grid sweep. |
| `plot_mobilecount_logpara2550.py` | Curves for the MobileCount LOG_PARA=638.73 LR-grid sweep. |
| `plot_mobilecount_logpara2550_overfit.py` | Overfit-style val-MAE panels for the same sweep. |

## `exploratory/` — **not used in the thesis**

Kept for completeness and because several are cited in the thesis's methodology
narrative as the steps that led to the final protocol. None of their outputs appear in
a table or figure, and their result files are **not** shipped in `results/`.

| Script | | Superseded by |
|---|---|---|
| `sweep_lr_schedule.py`, `sweep_lr_schedule_finish.py` | OFAT tuning of the LR-schedule shape. | the full grid |
| `compare_optimizers.py`, `compare_optimizers_archs.py`, `compare_optimizers_archs_finish.py` | Early Adam-vs-AdamW comparisons. | `tuning/sweep_*_logpara*` |
| `sweep_lr_grid.py`, `sweep_lr_grid_highlr.py` | The optimizer × LR grid before the LOG_PARA re-anchoring. | `tuning/sweep_*_logpara*` |
| `sweep_batchsize.py`, `merge_batchsize_results.py` | Batch-size sweep at the old LOG_PARA. | `tuning/sweep_batchsize_logpara.py` |
| `sweep_logpara_resolutions.py`, `sweep_logpara_variable.py` | Diagnosed the high-resolution target collapse and motivated area-scaled `LOG_PARA`. | folded into `loading_data.py` |
| `sweep_optimizer_seeds.py`, `plot_seed_sweep_curves.py`, `continue_seed3035_to1500.py` | Early seed-variance work at 800 epochs / old LOG_PARA. | the 5 same-seed repeats |
| `run_aug_schemes_360p.py`, `simple_aug.py`, `viz_aug.py` | The `basic`/`extended` augmentation sets (rotation, radiometric, sparse noise). Dropped: they target farm imagery this dataset does not contain. | `tuning/run_aug_paper_schemes_386x260.py` |
| `run_aug_seed_noise_386x260.py` | Seed-noise estimation for the augmentation comparison. Never completed; seed noise is discussed as a limitation instead. | — |
| `pool_bestcombo_resolutions_6runs.py` | Pools the single 1500-epoch run into the repeat statistics as a sixth sample (n=6). Not adopted — the thesis reports the five 800-epoch repeats alone. | `resolution/eval_bestcombo_repeats_testset.py` |
| `precompute_tenebrio_densities.py` | Writes density maps as `.csv`. | the `_hdf5.py` version — **use that one** |
| `plot_training_curves.py` | Generic 4-panel tfevents plot. | — |
