# Topological Cellular Fiber Segmentation

Code used for  `A topological object-proposal pipeline for cellular fiber segmentation`

The repository contains two segmentation workflows for H&E `.tif` images:

- `cellpose/`: [Cellpose/CPSAM segmentation](https://github.com/mouseland/cellpose).
- `tda/`: persistent-homology object proposal and post-processing.

## Setup

Install the two Micromamba/Conda environments:

```bash
micromamba create -f envs/cellpose.yml
micromamba create -f envs/tda.yml
```

The environment names used by the scripts are:

- `env-cellpose`
- `env-tda`

## Configure Paths

Before running, edit the variables at the top of:

- `cellpose/cellpose.sh`
- `tda/tda.sh`

Set:

- `INPUT_DIR`: directory containing input `.tif` images.
- `OUTPUT_DIR`: directory where segmentations and benchmark files will be written.

## Run on SLURM

Submit the jobs with:

```bash
sbatch cellpose/cellpose.sh
sbatch tda/tda.sh
```

Each script writes:

- segmentation outputs
- preview files, for TDA
- per-image timing logs
- benchmark summary tables

## Run TDA on One Image

```bash
micromamba run -n segm python tda/run_segmentation.py \
  --image input.tif \
  --labels-out segm_labels.tif \
  --preview-out segm_preview.tif
```
