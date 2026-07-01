# ML Environment Configuration

Last updated: 25 June 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack optimised for modern machine learning, statistics, and detailed chemical kinetics analysis on CSC supercomputers (**Puhti / Mahti / Roihu**). The setup focuses heavily on a high-throughput **JAX + Equinox + ONNX** development ecosystem.

Instead of deploying traditional conda or pip environments, we leverage **Tykky** to wrap the Python stack inside a single-file container image. This design addresses the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

### Why Tykky?

* **Import Performance** — Library initialization times drop from several minutes down to seconds.
* **Reproducibility** — The entire execution stack remains frozen within a single, immutable file image.
* **Startup Latency** — Immediate execution startup proves critical when running high-volume, short MPI jobs.
* **Isolation** — The complete Python dependency stack remains strictly separated from cluster host system modules.

> [!NOTE]
> This configuration forms an integral module of the [CSC Environment Helpers Framework](https://www.google.com/search?q=https://github.com/boss507104/CSCEnvironmentHelpers). This dedicated machine learning track removes the rigid dependency boundaries imposed by co-processing engines, switching the foundation to Python 3.12 and enabling unconstrained updates to core numerical backend targets.

---

## Global Configuration

Execute the following block to load your paths and build tools.

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export CSC_USER="USERNAME"                  # Your CSC username
export ENV_NICKNAME="NICKNAME"              # Desired environment name
# --- USER CONFIGURATION END ---

# Derived Paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$CSC_USER/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime"

# Initialise directories
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
echo "Configuration loaded for $CSC_PROJECT."

```

**Directory Structure**

```plaintext
/scratch/
└── $CSC_PROJECT/
    └── $CSC_USER/
        └── Utilities/                      # $BASE_SCRATCH
            ├── .tykky_runtime/             # $TMP_BUILD_DIR
            └── Python/                     # $PYTHON_ROOT
                └── envs/
                    └── $ENV_NICKNAME-3.12/  # $ENV_PREFIX

```

> [!TIP]
> Move your active development folders into **your own Utilities directory** located on the parallel scratch filesystem. Verify that you have explicitly initialised this path via `mkdir -p` before starting the compilation toolchains.

---

## Dependency Overview

| Package | Version | Purpose / Constraint |
| --- | --- | --- |
| **Python** | 3.12 | Tykky container engine foundation base layer (Optimised interpreter). |
| **numpy** | $\ge$ 2.0.0 | Enabled next-generation vectorisation and performance features. |
| **jax[cuda12]** | $\ge$ 0.4.30 | Uncapped tracking configuration for high-throughput execution backends. |

All secondary python packages resolve automatically to their latest compatible releases using `pip-compile`.

---

## Installation Steps

### 1. Create Configuration Files

#### Base Conda Specification

Navigate to your working directory and generate the initial recipe layout:

```bash
cd $PYTHON_ROOT
nano -m base4ML.yml

```

**Insert the following block into `base4ML.yml**`

```yaml
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.12
  - pip
  - git
  - compilers
  - cmake
  - make
  - ninja

```

#### Heavy-Lifting Post-Installation Script

```bash
nano -m extra4ML.sh

```

**Insert the following block into `extra4ML.sh**`

```bash
#!/bin/bash
set -e

pip install --no-cache-dir pip-tools setuptools

cat <<'IN' > requirements.in
# --- Core Math & Data ---
numpy>=2.0.0
bottleneck
dask
h5py
pandas
polars
scipy
xarray
zarr

# --- Data Formats ---
netCDF4
pyarrow
pyfoam

# --- Data Acquisition ---
kagglehub

# --- JAX Ecosystem ---
jax[cuda12]>=0.4.30
diffrax
distrax
einops
equinox
jax2onnx
jaxopt
jaxtyping
lineax
onnx
optax
optimistix
sympy2jax

# --- Machine Learning ---
catboost
feature-engine
gymnasium
lightgbm
linear-tree
mlflow
mlxtend
scikit-learn
tensorboard
obliquetree
wandb
xgboost

# --- Hyperparameter Optimisation ---
optuna

# --- Statistics ---
statsmodels

# --- Clustering & Dimensionality Reduction ---
hdbscan
igraph
leidenalg
umap-learn

# --- Physics & CFD ---
cantera
foamlib
meshio

# --- Mathematical Tools ---
numba
pint
ruptures
sympy
tensorly

# --- Custom Utilities ---
DataGraph @ git+https://github.com/boss507104/DataGraph.git#subdirectory=DataGraph

# --- Visualisation & UI ---
cmocean
colorcet
ipykernel
ipywidgets
IPython
ipyvtklink
k3d
matplotlib
plotly
pyvista
rich
scikit-image
seaborn
tqdm
trame
vtk

# --- Config & CLI ---
hydra-core
PyYAML

# --- HPC / SLURM ---
submitit

# --- System & Dev ---
kneed
natsort
pytest
tabulate
typing-extensions
IN

python -m piptools compile --allow-unsafe requirements.in
python -m pip install --no-cache-dir -r requirements.txt

```

```bash
chmod +x extra4ML.sh

```

### 2. Build Tykky Container

Request an interactive session on a test node to execute the container packaging tools:

```bash
srun --account=$CSC_PROJECT --partition=small --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --time=01:00:00 --pty bash

```

If network limits delay package downloads, switch to `--partition=medium` to lengthen runtime.

```bash
module load tykky
export TMPDIR=$TMP_BUILD_DIR

conda-containerize new \
    --prefix $ENV_PREFIX \
    --post-install $PYTHON_ROOT/extra4ML.sh \
    $PYTHON_ROOT/base4ML.yml

```

---

## Environment Activation / Loader

Save the following initialization template to your scratch workspace utility path at `$BASE_SCRATCH/Python4ML.sh`.

```bash
cat <<EOF > $BASE_SCRATCH/Python4ML.sh
#!/bin/bash

# Paths
export ENV_PREFIX="$ENV_PREFIX"

# Tykky container bin path
export PATH="\$ENV_PREFIX/bin:\$PATH"

# JAX tuning
export JAX_PLATFORMS="gpu"
EOF

```

```bash
chmod +x $BASE_SCRATCH/Python4ML.sh

```

Load your runtime stack during production job preparation steps: `source $BASE_SCRATCH/Python4ML.sh`

---

## VS Code Kernel Registration

Run this block to expose your single-file container image to Jupyter interfaces or remote VS Code connections:

```bash
mkdir -p ~/.local/share/jupyter/kernels/$ENV_NICKNAME-ml

cat <<EOF > ~/.local/share/jupyter/kernels/$ENV_NICKNAME-ml/kernel.json
{
 "argv": [
  "$ENV_PREFIX/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python-3.12 ($ENV_NICKNAME Tykky-ML)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
EOF

```

```bash
echo "Jupyter kernel '$ENV_NICKNAME' has been registered."

```

**Verify Registered Runtimes**

```bash
# Query current workspace kernels
source $BASE_SCRATCH/Python4ML.sh
jupyter kernelspec list

# Erase deprecated entries if required
jupyter kernelspec uninstall -f <kernel_name>

```

---

## Validation

Verify engine compatibility directly from your compute terminal:

```bash
source $BASE_SCRATCH/Python4ML.sh

```

```bash
python -c "
import jax, equinox as eqx, jax2onnx
import numpy as np
from importlib.metadata import version

print(f'JAX:        {jax.__version__}')
print(f'Equinox:    {eqx.__version__}')
print(f'jax2onnx:   {version(\"jax2onnx\")}')
print(f'NumPy:      {np.__version__}')
"

```

---

## Adding / Updating Packages (Optional)

To drop fresh packages into an existing deployment without initiating full container builds, configure an incremental update profile:

```bash
cat <<EOF > $PYTHON_ROOT/update_tools.sh
#!/bin/bash
set -e

echo "psutil" >> $PYTHON_ROOT/requirements.in

pip-compile --allow-unsafe --reuse-hashes $PYTHON_ROOT/requirements.in
pip install --no-cache-dir -r $PYTHON_ROOT/requirements.txt
EOF

```

```bash
module load tykky
conda-containerize update \
    --post-install $PYTHON_ROOT/update_tools.sh \
    $ENV_PREFIX

```

Group multiple python dependencies inside a single update script to reduce runtime repackaging overhead costs.

---

## Troubleshooting

* **Total State Purge** — Erase the deployment prefix target directory using `rm -rf $ENV_PREFIX` and repeat from step 2.
* **Sluggish Updates** — Consolidate scattered pip invocations inside a singular update file script layer.
* **Compiler Linkage Faults** — Verify host compiler environments explicitly using `module load gcc cmake`.
