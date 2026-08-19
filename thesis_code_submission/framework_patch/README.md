# `framework_patch/` — what to overlay onto C-3-Framework

Copy this directory's contents over a clean C-3-Framework checkout (commit `1325440`):

```bash
rsync -a --exclude README.md framework_patch/ /path/to/C-3-Framework/
```

(The `--exclude` matters: a plain `cp -r` would replace C-3-Framework's own
`README.md` with this file.)

It adds Tenebrio dataset support and the MobileCount architecture (8 new files) and
modifies 6 upstream files. Nothing else in C³ is touched — the other datasets and
architectures keep working.

---

## New files

| File | What it is |
|---|---|
| `datasets/Tenebrio/setting.py` | All dataset-level configuration: `DATA_PATH` (which resolution variant to train on), batch sizes, the `LOG_PARA` density scaling and its anchoring, and the `AUG` augmentation switch. Every option is documented inline. |
| `datasets/Tenebrio/Tenebrio.py` | `torch.utils.data.Dataset`. Reads a PNG plus its `.h5` density map, pads the image to a multiple of 8 so the encoder strides divide evenly, and pads the density map to match when it is at full image scale. |
| `datasets/Tenebrio/loading_data.py` | Builds the train/val loaders. Two things happen here that matter for the thesis: `LOG_PARA` is **rescaled by image area** at load time (`_compute_log_para`), and the augmentation pipeline is assembled from `cfg_data.AUG`. Geometric augmentations are applied jointly to image and density map so the count is preserved; photometric ones touch the image only. |
| `datasets/Tenebrio/__init__.py` | empty, package marker |
| `gen_density_map.py` | Standalone Gaussian density-map helper. The maps actually used were produced by `scripts/data_prep/precompute_tenebrio_densities_hdf5.py`. |
| `models/SCC_Model/MobileCount.py`<br>`MobileCountx1_25.py`, `MobileCountx2.py` | **Vendored, not mine.** The MobileCount reference implementation (<https://github.com/SelinaFelton/MobileCount>), placed into C³'s `SCC_Model` package so `models/CC.py` can construct it. Only the ×1.0 variant is used in the thesis. |

## Modified upstream files

| File | Change |
|---|---|
| `trainer.py` | **Optimiser and LR-schedule selection.** Upstream hardcodes `optim.Adam` + `StepLR`. Now reads `cfg.OPTIMIZER` (`Adam`/`AdamW`/`SGD`), `cfg.WEIGHT_DECAY` and `cfg.LR_SCHEDULE` (`step`/`constant`/`cosine`) — this is what makes `tab:lr_grid` possible. Also: the scheduler stepping was moved so the LR decay is applied once per epoch as intended, `torch.load(..., weights_only=False)` for PyTorch ≥ 2.6, and `cfg_data` is forwarded to the logger. |
| `train.py` | Registers the `Tenebrio` dataset and the three `MobileCount*` nets in the dispatch chains. Also replaces the upstream `is` string comparisons with `==` (they are `SyntaxWarning`s on modern Python and silently false). |
| `models/CC.py` | Three `elif` branches so `CrowdCounter` can build the MobileCount variants. |
| `misc/transforms.py` | Four new joint transforms for the augmentation study: `RandomVerticallyFlip`, `RandomRotationJoint` (rotates image and density map together), `RandomRadiometric` (per-channel offset/gain), `AddSparseGaussianNoise`. |
| `misc/utils.py` | `logger()` now records the **effective runtime config** — it dumps the live `cfg` and `cfg_data` dictionaries into each run's log file. Upstream copied the `config.py` source text instead, which is misleading here because the sweep drivers set every value in memory at runtime (see the main README, Section 4). |
| `config.py` | Defaults only. **Not the thesis configuration** — the sweep drivers set every value at runtime. See the caveat below before running `train.py` by hand. |

C³'s `trainer_for_CMTL.py` and `trainer_for_M2TCC.py` also need the same one-line
`torch.load(weights_only=False)` fix on PyTorch ≥ 2.6, but they are not shipped here:
they serve the CMTL and SANet architectures, neither of which this thesis uses.

---

## Caveat: `config.py` alone is not runnable

`config.py` no longer defines `RESUME` / `RESUME_PATH` (upstream did), but `trainer.py`
still reads `cfg.RESUME`. Every experiment driver sets it explicitly
(`config.cfg.RESUME = False`), so all sweeps run fine — but invoking C³'s own entry
point directly:

```bash
python train.py     # AttributeError: 'EasyDict' object has no attribute 'RESUME'
```

fails. To use `train.py` by hand, add to `config.py`:

```python
__C.RESUME = False
__C.RESUME_PATH = ''
```

and set `__C.DATASET = 'Tenebrio'` plus the net and hyperparameters you want.
`config.py` is shipped byte-identical to the version the results were produced with,
so this line was not added silently. **The drivers in `scripts/`, not `train.py`, are
the intended entry points.**

## Known documentation inaccuracy

The header comment in `datasets/Tenebrio/setting.py` states that density maps are
stored as `.csv` and are built by `split_tenebrio.py` + `resize_splits_to_resolutions.py`.
That describes an early version of the pipeline. The code actually reads **`.h5`**
(see `Tenebrio.py`, which opens `f['density']`), and the maps are produced by
`scripts/data_prep/precompute_tenebrio_densities_hdf5.py`.

The files here are byte-identical to the ones the results were produced with, so the
stale comment has been left in place rather than silently edited. Section 3 of the main
README gives the correct pipeline.

---

## Licence

C-3-Framework is MIT-licensed and its `LICENSE` file applies to the modified files
above. MobileCount retains its own upstream licence.
