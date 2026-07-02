# ML Environment Configuration

Last updated: 2 July 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack optimised for modern machine learning, statistics, and detailed chemical kinetics analysis on CSC supercomputers (**Puhti / Mahti / Roihu**). The setup focuses heavily on a high-throughput **JAX + Equinox + ONNX** development ecosystem.

Instead of deploying traditional conda or pip environments directly on the parallel filesystem, we use **Tykky** to package the Python stack inside a single-file container image. This design reduces the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

### Why Tykky?

* **Import Performance** — Library initialisation times drop from several minutes to seconds.
* **Reproducibility** — The complete execution stack remains packaged inside a single container image.
* **Startup Latency** — Fast environment startup proves valuable for high-volume, short MPI jobs.
* **Isolation** — The Python dependency stack remains separated from the cluster host environment.

### Why uv?

This configuration uses **uv** to resolve and install Python packages during the Tykky build.

* **Fast Resolution** — uv resolves large scientific Python dependency trees substantially faster than conventional pip workflows.
* **Compatible Dependency Selection** — uv selects mutually compatible direct and transitive package versions.
* **Compiled Requirements** — The resolved package set is recorded in `requirements.txt`.
* **Consistent Workflow** — The same resolver handles both initial installation and later package updates.

The direct package specifications intentionally avoid strict version pins. Each new container build therefore resolves a currently compatible package set.

> [!NOTE]
> Because the package versions are not fixed in `requirements.in`, rebuilding the environment at a later date may produce newer package versions. The compiled `requirements.txt` generated during each build records the exact resolved package set for that build.

---

## Global Configuration

Execute the following block to configure the project paths and environment name.

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export CSC_USER="USERNAME"                  # Your CSC username
export ENV_NICKNAME="NICKNAME"              # Desired environment name
# --- USER CONFIGURATION END ---

# Derived paths
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
        └── Utilities/                       # $BASE_SCRATCH
            ├── .tykky_runtime/              # $TMP_BUILD_DIR
            └── Python/                      # $PYTHON_ROOT
                ├── base4ML.yml
                ├── extra4ML.sh
                └── envs/
                    └── $ENV_NICKNAME-3.12/  # $ENV_PREFIX
```

> [!TIP]
> Store the configuration files and temporary build data under your own `Utilities` directory on the parallel scratch filesystem. Create the directory before starting the build.

---

## Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| **Python** | 3.12 | Base interpreter supplied through the Tykky Conda specification |
| **uv** | Latest available during build | Python dependency resolution and installation |
| **NumPy** | Compatible version selected by uv | Core numerical array backend |
| **JAX** | Compatible CUDA 12 release selected by uv | Array programming and automatic differentiation |
| **Equinox** | Compatible version selected by uv | Neural-network and PyTree framework for JAX |
| **ONNX / jax2onnx** | Compatible versions selected by uv | Model export and interoperability |

All direct and transitive Python dependencies are resolved with `uv` and recorded in a compiled `requirements.txt` file during the container build.

---

## Installation Steps

### 1. Create the Configuration Files

Navigate to the Python configuration directory:

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
```

### 1.1 Create the Base Conda Specification

Create `base4ML.yml`:

```bash
nano -m base4ML.yml
```

Insert the following block:

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

### 1.2 Create the Post-Installation Script

Create `extra4ML.sh`:

```bash
nano -m extra4ML.sh
```

Insert the following block:

```bash
#!/bin/bash
set -e

# Redirect temporary files and package caches to the scratch build directory
export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Keep generated requirement files inside the scratch build directory
cd "$CW_BUILD_TMPDIR"

# Install uv for dependency resolution and installation
python -m pip install --no-cache-dir uv

cat <<'IN' > requirements.in
# --- Core Math & Data ---
numpy
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
jax[cuda12]
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
obliquetree
scikit-learn
tensorboard
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

# --- HPC / Slurm ---
submitit

# --- System & Development ---
kneed
natsort
pytest
tabulate
typing-extensions
IN

# Resolve a mutually compatible dependency set
uv pip compile requirements.in \
    --output-file requirements.txt

# Install the resolved dependency set into the Tykky environment
uv pip install \
    --requirements requirements.txt

# Preserve the resolved package specifications for later inspection
cp requirements.in "$CW_BUILD_TMPDIR/requirements.direct.in"
cp requirements.txt "$CW_BUILD_TMPDIR/requirements.resolved.txt"

# Remove package caches after the installation
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x extra4ML.sh
```

---

## 2. Build the Tykky Container

Request an interactive compute node before running the container build.

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

If the package resolution or downloads require more time, request a partition and time limit appropriate for the target CSC system.

Load Tykky:

```bash
module load tykky
```

Configure the temporary build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Build the container:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

After a successful build, verify that the environment directory exists:

```bash
ls -ld "$ENV_PREFIX"
```

---

## Environment Activation / Loader

Create the runtime initialisation script at `$BASE_SCRATCH/Python4ML.sh`.

```bash
cat <<EOF > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

# Paths
export ENV_PREFIX="$ENV_PREFIX"

# Tykky container executable path
export PATH="\$ENV_PREFIX/bin:\$PATH"

# Prefer the JAX GPU backend when GPU resources are available
export JAX_PLATFORMS="gpu"
EOF
```

Make the loader executable:

```bash
chmod +x "$BASE_SCRATCH/Python4ML.sh"
```

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

Verify the active Python executable:

```bash
which python
python --version
```

> [!NOTE]
> `JAX_PLATFORMS="gpu"` requires a GPU allocation and compatible CUDA driver environment. Remove or override this variable when running CPU-only workloads.

For a CPU-only session:

```bash
export JAX_PLATFORMS="cpu"
```

---

## VS Code Kernel Registration

Register the Tykky Python environment as a Jupyter kernel for remote VS Code sessions.

Create the kernel directory:

```bash
mkdir -p "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-ml"
```

Create `kernel.json`:

```bash
cat <<EOF > "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-ml/kernel.json"
{
  "argv": [
    "$ENV_PREFIX/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.12 ($ENV_NICKNAME Tykky ML)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF
```

Confirm the registration:

```bash
echo "Jupyter kernel '$ENV_NICKNAME-ml' has been registered."
```

List available kernels:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
jupyter kernelspec list
```

Remove an obsolete kernel when necessary:

```bash
jupyter kernelspec uninstall -f <kernel_name>
```

---

## Validation

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

For a CPU-only validation:

```bash
export JAX_PLATFORMS="cpu"
```

Verify the core package versions:

```bash
python -c "
import jax
import equinox as eqx
import numpy as np
from importlib.metadata import version

print(f'Python:     {version(\"pip\")}')
print(f'JAX:        {jax.__version__}')
print(f'Equinox:    {eqx.__version__}')
print(f'jax2onnx:   {version(\"jax2onnx\")}')
print(f'NumPy:      {np.__version__}')
print(f'Devices:    {jax.devices()}')
"
```

Verify that important scientific packages import correctly:

```bash
python -c "
import cantera
import h5py
import matplotlib
import onnx
import optax
import pandas
import scipy
import sklearn
import xarray

print('Core ML and scientific packages imported successfully.')
"
```

Inspect the installed package set:

```bash
uv pip freeze
```

Alternatively:

```bash
python -m pip freeze
```

---

## Resolved Dependency Records

The `uv pip compile` command creates:

```text
$TMP_BUILD_DIR/requirements.txt
```

This file contains the exact direct and transitive versions selected during the build.

The post-installation script also creates:

```text
$TMP_BUILD_DIR/requirements.direct.in
$TMP_BUILD_DIR/requirements.resolved.txt
```

The files serve different purposes:

* `requirements.direct.in` records the requested top-level packages without strict version pins.
* `requirements.resolved.txt` records the exact package versions selected by uv.
* Rebuilding from the unpinned direct requirements may select newer compatible packages.
* Reinstalling from the resolved requirements recreates the package set selected during that build, subject to wheel and platform availability.

Copy the resolved file to a permanent location after a successful build:

```bash
cp "$TMP_BUILD_DIR/requirements.resolved.txt" \
    "$PYTHON_ROOT/$ENV_NICKNAME-requirements.txt"
```

---

## Adding or Updating Packages

Tykky updates should install all requested changes through one post-installation script.

### 1. Create an Update Script

Create `update4ML.sh`:

```bash
nano -m "$PYTHON_ROOT/update4ML.sh"
```

Insert the following block:

```bash
#!/bin/bash
set -e

# Redirect temporary files and caches to scratch
export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
cd "$CW_BUILD_TMPDIR"

# Install uv inside the update context
python -m pip install --no-cache-dir uv

cat <<'IN' > update-requirements.in
psutil
IN

# Resolve the requested additions and their dependencies
uv pip compile update-requirements.in \
    --output-file update-requirements.txt

# Install the resolved updates
uv pip install \
    --requirements update-requirements.txt

# Clear temporary caches
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Add all packages required by the update to `update-requirements.in`.

Make the script executable:

```bash
chmod +x "$PYTHON_ROOT/update4ML.sh"
```

### 2. Apply the Update

Load Tykky:

```bash
module load tykky
```

Configure the build directories:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Update the existing environment:

```bash
conda-containerize update \
    --post-install "$PYTHON_ROOT/update4ML.sh" \
    "$ENV_PREFIX"
```

Group multiple package additions or upgrades inside one update script to minimise repeated container repackaging.

> [!NOTE]
> Incrementally installing unpinned packages can upgrade dependencies already present in the environment. Rebuild the complete environment when dependency consistency matters more than preserving the existing image.

---

## Rebuilding the Environment

Remove the current environment:

```bash
rm -rf "$ENV_PREFIX"
```

Clear the temporary build directory:

```bash
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Run the Tykky build again:

```bash
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

Because `requirements.in` contains no strict version pins, the rebuilt environment may contain newer compatible packages than the previous build.

---

## Troubleshooting

### Total Environment Reset

Remove the environment and temporary build directory:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Then repeat the container build.

### uv Cannot Find the Active Environment

Confirm that the post-installation script runs inside the Tykky build environment.

Inspect:

```bash
which python
python --version
python -m pip --version
```

Install uv through the active Python interpreter:

```bash
python -m pip install --no-cache-dir uv
```

### Package Resolution Fails

Run the compile command manually inside the build context:

```bash
cd "$CW_BUILD_TMPDIR"

uv pip compile requirements.in \
    --output-file requirements.txt
```

The resolver output should identify conflicting direct dependencies.

Because the guide does not enforce package versions, remove or replace packages that impose mutually incompatible constraints.

### Package Installation Exceeds the Home Quota

Verify that the temporary directories point to scratch:

```bash
echo "$TMPDIR"
echo "$PIP_CACHE_DIR"
echo "$UV_CACHE_DIR"
```

They should point under:

```text
$BASE_SCRATCH/.tykky_runtime
```

### JAX Reports That No GPU Is Available

Confirm that the shell runs inside a GPU allocation:

```bash
nvidia-smi
```

Check the JAX devices:

```bash
python -c "import jax; print(jax.devices())"
```

For CPU-only use:

```bash
export JAX_PLATFORMS="cpu"
```

### Import Errors After an Incremental Update

An unpinned incremental update may replace shared dependencies with newer releases.

Inspect the installed versions:

```bash
uv pip freeze
```

When the resulting environment becomes inconsistent, rebuild the complete Tykky image rather than stacking further updates.

### Compiler Linkage Errors

Inspect the available compiler modules:

```bash
module avail gcc
module avail cmake
```

Load the compiler modules required by the affected package before starting the Tykky build.

### The Build Takes Too Long

Request a longer interactive allocation appropriate for the CSC system and partition.

Avoid running the build directly on a login node.

---

## Notes

* The environment uses Python 3.12.
* Python packages are intentionally specified without strict version pins.
* uv selects a compatible direct and transitive dependency set during each build.
* The exact selected versions are recorded in the compiled `requirements.txt`.
* Rebuilding the environment later may select newer compatible releases.
* `jax[cuda12]` installs the CUDA 12-compatible JAX package set, but GPU execution still requires a GPU allocation and compatible host drivers.
* The Tykky image should be treated as the deployed runtime unit.
* Use batch or interactive compute nodes for environment builds and computational workloads.
* Avoid performing large package installations directly on CSC login nodes.
* Prefer a complete rebuild over repeated incremental updates when the dependency set changes substantially.
