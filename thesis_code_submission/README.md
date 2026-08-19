# Crowd counting for *Tenebrio molitor* larvae — thesis code submission

Code accompanying the BSc thesis *"[thesis title]"*.

The study compares **CSRNet** and **MobileCount** on the **TenebrioVision** dataset
across six input resolutions, using the **C³ Framework** as the training codebase.
This submission contains the parts that are mine: the Tenebrio dataset adapter and
augmentation pipeline, the patches that make C³ do what the thesis needed, every
experiment driver, and the small result files that back each table and figure.

**None of the external code is re-uploaded here.** Section 2 explains how to fetch
C-3-Framework and apply the patch in `framework_patch/` on top of it.

---

## 1. What you can reproduce, and how expensive it is

There are two entry points. Pick based on what you want to check.

| | Path A — regenerate tables & figures | Path B — retrain from scratch |
|---|---|---|
| Needs a GPU | no | yes (16 GB) |
| Needs the dataset | no | yes |
| Needs external code | no | yes (C-3-Framework) |
| Time | seconds | ~8 days for the main sweep alone |
| What it proves | the numbers in the thesis follow from the recorded runs | the recorded runs follow from the data |

**Path A works immediately after unpacking this folder** — the per-run metrics of every
sweep are included under `results/`, so the tables and figures can be rebuilt without
training anything. (Two evaluations are the exception: they read model checkpoints,
which are too large to ship. Section 5 says which, and includes their output.)
Start here:

```bash
# from anywhere; no install beyond numpy + matplotlib
mkdir -p /tmp/repro && cp -r results/exp scripts /tmp/repro/
cd /tmp/repro
python scripts/figures/plot_mae_vs_memory_repeats.py     # -> thesis figure (fig:acc_vs_mem)
python scripts/tuning/merge_batchsize_logpara_results.py # -> thesis table  (tab:batchsize)
```

Section 5 maps every thesis table and figure to the exact command that produces it.

---

## 2. Setting up the external code

### 2.1 C-3-Framework

All training runs through the C³ Framework (Gao et al., 2019).

```bash
git clone https://github.com/gjy3035/C-3-Framework.git
cd C-3-Framework
git checkout 1325440          # the commit this work was built on
```

Then overlay this submission on top of it:

```bash
C3=/path/to/C-3-Framework
# adds Tenebrio support + the modifications (--exclude keeps this folder's own
# README.md from replacing C-3-Framework's)
rsync -a --exclude README.md framework_patch/ "$C3"/
cp -r scripts       "$C3"/          # the experiment drivers
mkdir -p "$C3"/exp
cp -r results/exp/. "$C3"/exp/      # optional: the recorded results, for Path A
```

`framework_patch/` modifies 6 upstream files and adds 8 new ones.
`framework_patch/README.md` lists exactly what each one changes and why.

### 2.2 MobileCount

The MobileCount architecture is **already included** in `framework_patch/models/SCC_Model/`
(C³ does not ship it). It is a vendored copy of the reference implementation from
<https://github.com/SelinaFelton/MobileCount>, unmodified except for being placed in
C³'s `SCC_Model` package. Nothing extra to clone.

CSRNet is part of C³ and needs no changes. Its VGG-16 frontend is initialised from the
torchvision ImageNet weights, downloaded automatically on first run.

### 2.3 Python environment

```bash
cd "$C3"
python -m venv .venv && source .venv/bin/activate
pip install -r /path/to/this/submission/requirements.txt
```

Python 3.12.3. See `requirements.txt` for the exact versions the reported results were
produced with; install PyTorch matched to your CUDA version first.

### 2.4 The dataset

TenebrioVision (Papadopoulos et al., ICPRAM 2024) is not redistributed here. Obtain the
images and the COCO-style annotation JSON from the dataset authors, then follow Section 3.

---

## 3. Preparing the data

Run everything from the C³ root, with the venv active. The end state is what
`datasets/Tenebrio/setting.py` expects: one folder per resolution variant, each
holding `{train,val,test}/{img,den}`.

```bash
# 1. count-stratified split into 840 / 168 / 112 images (seed 42)
python scripts/data_prep/split_tenebrio.py \
    --source /path/to/TenebrioVision_Images \
    --dest   exp/data/TenebrioVision_Original

# put the annotation JSON where step 3 looks for it
cp /path/to/TenebrioVision_Annotations.json exp/data/TenebrioVision_Original/

# 2. resize the 3088x2076 originals into the six resolution variants (LANCZOS)
python scripts/data_prep/resize_splits_to_resolutions.py \
    --data-dir exp/data/TenebrioVision_Original
mkdir -p exp/data/Tenebrio
mv exp/data/TenebrioVision_Original/[0-9]*x[0-9]* exp/data/Tenebrio/

# 3. render ground-truth density maps (.h5) for every variant
python scripts/data_prep/precompute_tenebrio_densities_hdf5.py \
    --data-dir    exp/data/Tenebrio \
    --annotations exp/data/TenebrioVision_Original/TenebrioVision_Annotations.json
```

Resulting layout:

```
exp/data/
├── TenebrioVision_Original/            3088x2076 source splits + annotations
│   └── {train,val,test}/img/
└── Tenebrio/
    ├── 49x33/  97x65/  193x130/  386x260/  772x519/  1544x1038/
    │   └── {train,val,test}/
    │       ├── img/                    resized PNGs
    │       └── den/                    density maps, one .h5 per image
    └── 386x260_csrnet9/                only for the augmentation experiment (step 4)
```

Density maps are Gaussians on the bbox centres, renormalised so the map integrates to
the object count. Sigma scales with resolution (24 / 12 / 6 / 3 / 2 / 1 px, largest to
smallest) so a larva covers the same fraction of the plate at every size.

```bash
# 4. only needed for the augmentation comparison: the offline 18-patch train set
python scripts/data_prep/make_csrnet9_patches_386x260.py
```

Sanity check that the framework, the patch and the data agree:

```bash
python scripts/data_prep/verify_train.py
```

---

## 4. How an experiment is configured

The sweep drivers do **not** read settings from `config.py`. Each one imports `config`
and overwrites `config.cfg.*` and `cfg_data.*` in memory before calling the trainer, so
one script can run a whole grid in a single process. Reading the top ~60 lines of any
driver tells you the full configuration of that experiment — for example
`scripts/resolution/run_bestcombo_resolutions_repeats.py`:

```python
config.cfg.DATASET   = "Tenebrio"
config.cfg.MAX_EPOCH = 800
BEST = {
    "CSRNet":      {"optimizer": "AdamW", "lr": 4e-5, "bs": 1, "lp_base": 100.0},
    "MobileCount": {"optimizer": "AdamW", "lr": 1e-3, "bs": 6, "lp_base": 2550.0},
}
```

The checked-in `config.py` is therefore a set of defaults, not the thesis
configuration. What each run actually used is recorded per run: the patched
`misc/utils.py` writes the **live** `cfg` and `cfg_data` into the run's log file at
startup, so every experiment folder documents itself.

> **The drivers in `scripts/`, not C³'s `train.py`, are the entry points.** Running
> `python train.py` directly raises `AttributeError: … 'RESUME'`, because `config.py`
> does not define a key that `trainer.py` reads and the drivers supply. See
> `framework_patch/README.md` for the two lines that fix it if you want to train a
> single configuration by hand.

Common environment overrides, honoured by most drivers:

| Variable | Meaning |
|---|---|
| `C3_ROOT` | project root, if the scripts are not two levels below it |
| `SWEEP_DIR` | where the sweep writes; re-running skips completed cells (relaunch-safe) |
| `MAX_EPOCH` | epoch budget — set `MAX_EPOCH=2` for a smoke test |
| `ONLY_NET` | restrict to `CSRNet` or `MobileCount` |
| `RES`, `REPS` | restrict the resolution / repeat list |

Sweeps are resumable. Completed cells are skipped from the results JSON, and a
half-finished cell resumes from `latest_state.pth`, so an interrupted sweep can be
relaunched with the same command.

---

## 5. Reproducing each thesis table and figure

Tables are referenced by their LaTeX `\label`, which is stable across draft revisions.

Every row's **regenerate** command is Path A: it rebuilds the table or figure from the
result files shipped in `results/`, no GPU and no dataset needed. The **retrain**
command is Path B, and reruns the training that produced those files.

| Thesis item | Regenerate (Path A) | Retrain (Path B) | Cost |
|---|---|---|---|
| `tab:aug_paper_schemes`<br>Augmentation schemes | `figures/plot_aug_paper_schemes.py` | `data_prep/make_csrnet9_patches_386x260.py`<br>`tuning/run_aug_paper_schemes_386x260.py`<br>`tuning/eval_aug_paper_schemes_testset.py` | 6 runs × 1500 ep |
| `tab:lr_grid`<br>Optimiser × learning rate | numbers are read directly from<br>`results/exp/360p_lr_optim_grid/*/…json` | `tuning/sweep_csrnet_logpara_anchor772x519.py`<br>`tuning/sweep_mobilecount_logpara2550.py` | 14 runs × 1500 ep |
| `tab:batchsize`<br>Batch-size sweep | `tuning/merge_batchsize_logpara_results.py` | `tuning/sweep_batchsize_logpara.py`, then the merge script | 8 runs × 1500 ep |
| `tab:resolution_test`<br>Cross-resolution test MAE | `results/exp/bestcombo_resolutions_repeats_800ep/`<br>`testset_repeats_table.md` (epoch ranges: the<br>per-rep `best_epoch` in `testset_repeats_results.json`) | `resolution/run_bestcombo_resolutions_repeats.py`<br>`resolution/eval_bestcombo_repeats_testset.py` | **60 runs × 800 ep ≈ 184 GPU-h** |
| `fig:allres_trend`<br>Test MAE/MSE vs resolution | `figures/plot_test_error_vs_resolution_repeats.py` | (reuses the 60-run sweep above) | — |
| `tab:e4_budget`<br>Checkpoint vs epoch-1500 MAE<br>(single 1500-ep run) | `results/exp/combined_resolution/`<br>`trainsplit_best_vs_last.json` | `resolution/run_best_combo_resolutions.py`<br>`resolution/finish_bestcombo_resolutions_1544.py`<br>`resolution/combine_resolution_results.py`<br>`resolution/eval_bestcombo_best_vs_last_trainsplit.py` | 12 runs × 1500 ep |
| `fig:acc_vs_mem`<br>Accuracy–memory frontier | `figures/plot_mae_vs_memory_repeats.py` | + `resolution/measure_inference_memory.py` | seconds (GPU) |
| `fig:density_maps`<br>Appendix density-map grid | `figures/viz_density_examples.py --all 80_18`<br>(picker: `figures/rank_net_disagreement.py`) | needs checkpoints + dataset | seconds (GPU) |
| `tab:baseline_hyperparameters`<br>`tab:final_configs` | not computed — the C³ defaults and the values selected by the three sweeps above. Both are stated in the sweep drivers' headers. | — | — |

Two rows have no Path A command because the evaluation reads model **checkpoints**,
not logged metrics: `eval_bestcombo_repeats_testset.py` and `viz_density_examples.py`.
Checkpoints are far too large to include (the repeats sweep alone is 2.3 GB after
pruning to one checkpoint per run), so their outputs are shipped pre-computed —
`testset_repeats_table.md` is the table as it appears in the thesis.

**Verified:** rebuilding `fig:acc_vs_mem` and `fig:allres_trend` from the shipped
results reproduces the figures in the thesis byte-for-byte, and the regenerated
`tab:batchsize` and `tab:e4_budget` numbers are identical to the shipped ones.

---

## 6. Layout of this submission

```
.
├── README.md                  this file
├── requirements.txt
│
├── framework_patch/           overlay onto a C-3-Framework clone (see its README)
│   ├── datasets/Tenebrio/     dataset adapter, loader, augmentation config   [new]
│   ├── models/SCC_Model/      MobileCount, vendored                          [new]
│   ├── misc/transforms.py     joint image+density augmentations           [modified]
│   ├── trainer.py             optimiser / LR-schedule selection           [modified]
│   └── …
│
├── scripts/                   experiment drivers, by pipeline stage
│   ├── data_prep/     (5)     dataset construction — Section 3
│   ├── tuning/        (9)     the three tuning sweeps + their evaluations
│   ├── resolution/   (10)     the final cross-resolution study + measurements
│   ├── figures/      (19)     every plot and visualisation
│   └── exploratory/  (21)     runs that did NOT feed the thesis — see its README
│
└── results/exp/               per-run metrics backing every table (~25 MB)
```

`scripts/README.md` is a one-line index of all 64 scripts. Every script also carries a
docstring stating what it does, what it reads, and what it writes.

The `results/exp/` tree deliberately mirrors the directory names used during the
experiments (`bestcombo_resolutions_repeats_800ep/`, `aug_paper_schemes_386x260_07-04/`,
…) so that the scripts find their inputs unmodified. `results/README.md` maps those
names to what they contain.

---

## 7. Changes made when packaging this submission

The scripts here are the ones that produced the results, with four mechanical edits.
They are listed so nothing is hidden:

1. **Absolute paths removed.** Scripts hardcoded
   `ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")`. This is now
   `ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])` —
   the project root is inferred from the script's location, or set via `C3_ROOT`.
   This is why `scripts/` must sit directly inside the C³ root.
2. **A stale directory name fixed.** Four scripts pointed at `exp/aa_final/`, which had
   been renamed to `exp/360p_lr_optim_grid/` after they were written; they failed with
   `FileNotFoundError` until now.
3. **Scripts grouped into subfolders.** They were previously flat in `scripts/`.
   Cross-references inside docstrings use bare filenames and still resolve by name.
4. **`verify_train.py` given the same `sys.path` setup its siblings already had.**
   It alone imported `config` without putting the project root on the path, so it
   previously ran only with `PYTHONPATH` set.
5. **One deliberate version pin.** `figures/plot_mae_vs_memory_repeats.py` is kept at
   the revision that produced the thesis figure (five-repeat means). The working tree
   later gained an optional mode that pools the 1500-epoch run in as a sixth sample;
   that aggregation ships as `exploratory/pool_bestcombo_resolutions_6runs.py` but is
   not used by any thesis number.

No behaviour, hyperparameter, or numerical result is affected. `framework_patch/` and
`results/` are byte-identical copies of the working tree.

**Verified end-to-end:** a clean `C-3-Framework` checkout at commit `1325440`, patched
and populated exactly as Sections 2 and 5 describe, builds both networks and completes
a training step, and regenerates the thesis figure byte-for-byte.

---

## 8. Hardware and expected runtime

All results were produced on one NVIDIA GeForce RTX 5070 Ti (16 GB), PyTorch 2.11,
CUDA. Timings from the recorded runs, per training run at 800 epochs:

| Cell | Time / run |
|---|---|
| MobileCount @ 49×33 | 23 min |
| CSRNet @ 386×260 | 2.0 h |
| CSRNet @ 772×519 | 5.4 h |
| CSRNet @ 1544×1038 | 20.1 h |

The full 60-run repeat sweep took **184 GPU-hours**. Use `MAX_EPOCH=2` plus `RES` /
`ONLY_NET` / `REPS` to smoke-test any driver in a few minutes before committing to a
full sweep.

Runs are **not** bit-reproducible even at a fixed seed: cuDNN kernel selection and the
unseeded DataLoader workers are nondeterministic. That is deliberate — the five
same-seed repeats behind `tab:resolution_test` exist to quantify exactly that spread,
so re-running is expected to land near, not on, the reported numbers.

---

## 9. Credits

* **C³ Framework** — Gao, Lin, Zhao, Wang, Gao & Wen, *C³ Framework: An Open-source
  PyTorch Code for Crowd Counting*, arXiv:1907.02724 (2019).
  <https://github.com/gjy3035/C-3-Framework>
* **MobileCount** — Wang, Gao, Lin & Yuan, *MobileCount: An efficient encoder-decoder
  framework for real-time crowd counting*, Neurocomputing (2020).
* **CSRNet** — Li, Zhang & Chen, CVPR 2018.
* **TenebrioVision** — Papadopoulos et al., ICPRAM 2024, pp. 187–196.

C-3-Framework is MIT-licensed; the upstream `LICENSE` applies to the files in
`framework_patch/` that derive from it.
