# ENSF 519 Assignment 2 — Object Detection on a Table

## Setup

**Option 1**: Local environment using `uv`

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). For example, on macOS with `curl`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. From the assignment directory, run one of the following commands to create
   a virtual environment.

   If you have a CPU-only machine, run:

   ```bash
   uv sync --extra cpu
   ```

   On a machine with a compatible NVIDIA GPU and driver, run:

   ```bash
   uv sync --extra cuda
   ```

   The CUDA environment uses PyTorch wheels built for CUDA 12.6. This command
   installs the dependencies in `pyproject.toml` and `uv.lock` and creates a
   virtual environment in `.venv/`.

The first two notebooks are for testing purposes and can be completed on a CPU. A CUDA GPU is recommended for the full experiments in `02_training_and_testing.ipynb`.

Start Jupyter from the assignment directory and run the notebooks in numerical
order. The relative file paths in the notebooks assume that this directory is
the current working directory.

**Option 2**: Using TALC

1. Connect to TALC using your UCalgary IT account, and clone the assignment into your home directory:

   ```bash
   ssh <your_IT_account_name>@talc.ucalgary.ca
   git clone --no-local /work/TALC/ensf519.53_2026f/assignment_templates/assignment2.git ENSF519_assignment2
   ```
2. For interactive jobs, it is suggested to use a CPU node while working through the first two notebooks:

   ```bash
   salloc --partition=cpu16 --time=01:00:00 --cpus-per-task=4 --mem=8G
   ```

   Before working through the third notebook, request a GPU node instead:

   ```bash
   salloc --partition=gpu --time=01:00:00 --gpus-per-node=1 --cpus-per-task=4 --mem=8G
   ```
3. On the allocated node, load Python and activate the course environment. On
   a GPU node, also load the CUDA 12.6.2 toolkit before activating the
   environment.

   ```bash
   module load python/3.14.6
   module load cuda/12.6.2 # GPU jobs only
   source /work/TALC/ensf519.53_2026f/envs/a2/bin/activate
   cd ENSF519_assignment2
   jupyter lab --no-browser --port=8888
   ```

   Omit the `module load cuda/12.6.2` line for a CPU job.

For more details about remote editing, requesting computing resources, and connecting to JupyterLab through port forwarding, refer to the TALC guide published earlier in the course.

## Assignment overview

In this assignment, you will build a small object detector for tabletop objects using PyTorch. The prepared VMRD dataset contains ten object classes and uses COCO-style annotation files. The detector resizes each image to `256 × 256`, divides it into a `16 × 16` grid, and predicts objectness, a class, and a bounding box for every grid cell.

The structure of this assignment is as follows:

```
ENSF519_assignment2/
├── 00_dataset_and_tensors.ipynb
├── 01_cnn_model.ipynb
├── 02_training_and_testing.ipynb
├── assets/
│   ├── grid_targets.png
│   ├── model_architecture.png
│   └── object_detection_grid_nms.png
├── assignment2/
│   ├── __init__.py
│   ├── box_utils.py       # provided bounding-box utilities
│   ├── data.py            # *image statistics and DataLoaders
│   ├── evaluation.py      # provided decoding and evaluation utilities
│   ├── model.py           # *ResNet9 detector and output splitting
│   └── training.py        # *detection loss, training, and evaluation functions
├── data/
│   ├── VMRD_COCO.tar.gz.part-00
│   ├── VMRD_COCO.tar.gz.part-01
│   └── VMRD_COCO.tar.gz.part-02
├── .python-version
├── make_submission.py
├── make_submission.sh
├── pyproject.toml
├── report.md
├── README.md
└── uv.lock
```

In this assignment, you are expected to complete the implementation of the files with stars (*) above, conduct the experiments in `02_training_and_testing.ipynb`, and complete your report in `report.md`.

You are suggested to work through the notebooks from 00 to 02 in order:

1. `00_dataset_and_tensors.ipynb` first extracts the included dataset archive
   parts into `data/VMRD_COCO/`. It then introduces the dataset, COCO
   annotations, and grid targets. You will calculate the image statistics and
   complete the three DataLoader functions.
2. `01_cnn_model.ipynb` introduces the model and detection loss. You will implement the ResNet9 detector, loss calculations, training loop, and evaluation loop.
3. `02_training_and_testing.ipynb` trains a baseline model for 25 epochs. You will conduct controlled training-hyperparameter and architecture experiments, select a model using validation loss, precision, and recall, evaluate it on the test set, and visualize predictions.

Detailed instructions and code for checking your implementation are included in the notebooks. Use the validation set when comparing models and tuning settings. Use the test set only once, after selecting your final model.

Before starting the full experiments, make sure the sanity checks in Notebooks
00 and 01 run without errors. If you edit a Python file while a notebook is
open, rerun the relevant cells before continuing. Training results can vary
slightly between runs even when a random seed is set.

## Submission

After completing the code, notebooks, and `report.md`, create the submission
ZIP from the assignment directory:

- macOS/Linux: `sh make_submission.sh`
- Windows (PowerShell): `py make_submission.py` (or `python make_submission.py`)

Both commands create `ENSF519_assignment2_submission.zip`. The dataset,
virtual environment, and training-output folders are intentionally excluded.
Upload this ZIP file to D2L.

## Rubric

| Component                               |        Marks |
| --------------------------------------- | -----------: |
| Dataset transformations and DataLoaders |            2 |
| CNN model and detection loss            |            4 |
| Training and evaluation functions       |            4 |
| Experiments and report questions        |            8 |
| AI use declaration                      |            1 |
| Reflection                              |            1 |
| **Total**                         | **20** |

Detailed marks for individual methods and questions are provided in the
notebooks and [`report.md`](report.md).
