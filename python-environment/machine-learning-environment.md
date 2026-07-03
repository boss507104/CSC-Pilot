# ML Environment Configuration

Last updated: 3 July 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack optimised for modern machine learning, statistics, and detailed chemical kinetics analysis on CSC supercomputers (**Puhti / Mahti / Roihu**). The setup focuses on a high-throughput **JAX + Equinox + ONNX** development ecosystem.

Instead of deploying traditional Conda or pip environments directly on the parallel filesystem, we use **Tykky** to package the Python stack inside a single-file container image. This design reduces the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

### Why Tykky?

* **Import Performance** — Library initialisation times drop from several minutes to seconds.
* **Reproducibility** — The complete execution stack remains packaged inside a single container image.
* **Startup Latency** — Fast environment startup is valuable for high-volume, short MPI jobs.
* **Isolation** — The Python dependency stack remains separated from the cluster host environment.

### Why uv?

This configuration uses **uv** inside the Tykky build environment to resolve and install Python packages.

* **Fast Resolution** — uv resolves large scientific Python dependency trees substantially faster than conventional pip workflows.
* **Compatible Dependency Selection** — uv selects mutually compatible direct and transitive package versions.
* **Compiled Requirements** — The resolved package set is recorded in `requirements.txt`.
* **Consistent Workflow** — The same resolver handles dependency compilation and installation.

The direct package specifications in `requirements.in` intentionally avoid strict version pins. Each new dependency compilation may therefore resolve a newer compatible package set.

`requirements.in` records the requested top-level packages, while `requirements.txt` records the exact direct and transitive versions selected by uv.

Dependency resolution takes place inside the Python 3.12 environment created by Tykky. No separate Conda, Miniforge, Mamba, Python module, or virtual environment is required.

---

## Global Configuration

Execute the following block to configure the project paths and environment name.

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export PROJECT_USER_DIR="Harry"             # Your directory under the CSC project
export ENV_NICKNAME="Dumbledore"            # Desired environment name
# --- USER CONFIGURATION END ---

# Derived paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime"

# Initialise directories
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "Configuration loaded for $CSC_PROJECT."
```

The configuration variables represent:

```text
CSC_PROJECT       CSC project ID
PROJECT_USER_DIR  Personal or shared directory under the CSC project
ENV_NICKNAME      Name assigned to the Python environment
```

For example:

```bash
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"
```

The resulting base path is:

```text
/scratch/project_xxxxxxx/Harry/Utilities
```

`Harry` and `Dumbledore` are fictional placeholder values used in this public documentation. Replace them with your actual project directory and preferred environment name.

`PROJECT_USER_DIR` is not necessarily the same as your CSC login username. It identifies the directory located directly under the CSC project scratch path.

**Directory Structure**

```plaintext
/scratch/
└── $CSC_PROJECT/
    └── $PROJECT_USER_DIR/
        └── Utilities/                       # $BASE_SCRATCH
            ├── .tykky_runtime/              # $TMP_BUILD_DIR
            ├── Python4ML.sh
            └── Python/                      # $PYTHON_ROOT
                ├── base4ML.yml
                ├── extra4ML.sh
                ├── update4ML.sh
                ├── requirements.in
                ├── requirements.txt
                └── envs/
                    └── $ENV_NICKNAME-3.12/  # $ENV_PREFIX
```

---

## Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| **Python** | 3.12 | Base interpreter created by Tykky |
| **uv** | Latest available during the build | Dependency resolution and installation |
| **NumPy** | Compatible version selected by uv | Core numerical array backend |
| **JAX** | Compatible CUDA 12 release selected by uv | Array programming and automatic differentiation |
| **Equinox** | Compatible version selected by uv | Neural-network and PyTree framework for JAX |
| **ONNX / jax2onnx** | Compatible versions selected by uv | Model export and interoperability |

The environment uses two dependency files:

```text
requirements.in   Direct, human-maintained package specifications
requirements.txt  Exact direct and transitive versions generated by uv
```

---

## Installation Steps

### 1. Create the Configuration Files

Create the Python configuration directory:

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
```

### 1.1 Create the Base Conda Specification

Create `base4ML.yml`:

```bash
nano -m base4ML.yml
```

Insert:

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

### 1.2 Create the Direct Dependency Specification

Create `requirements.in`:

```bash
nano -m requirements.in
```

Insert:

```text
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
```

### 1.3 Create the Post-Installation Script

Create `extra4ML.sh`:

```bash
nano -m extra4ML.sh
```

Insert:

```bash
#!/bin/bash
set -e

# Confirm that the build variables are available
: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"

# Redirect temporary files and package caches to scratch
export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Python 3.12 Tykky build environment
python -m pip install --no-cache-dir uv

# Resolve the complete dependency set
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt"

# Install the resolved dependency set
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.txt"

# Remove package caches
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x extra4ML.sh
```

---

## 2. Build the Tykky Container

Request an interactive compute node:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

Load Tykky:

```bash
module purge
module load tykky
```

Configure the temporary build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Verify the configuration files:

```bash
ls -l \
    "$PYTHON_ROOT/base4ML.yml" \
    "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/requirements.in"
```

Remove an existing or incomplete environment:

```bash
rm -rf "$ENV_PREFIX"
```

Build the container:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

During the build, Tykky performs the following operations:

```text
1. Creates the Python 3.12 base environment.
2. Runs extra4ML.sh inside that environment.
3. Installs uv.
4. Compiles requirements.in into requirements.txt.
5. Installs the resolved dependency set.
6. Packages the complete environment into the Tykky image.
```

After the build completes:

```bash
ls -ld "$ENV_PREFIX"
```

Confirm that the resolved dependency file was created:

```bash
ls -lh "$PYTHON_ROOT/requirements.txt"
```

---

## Environment Activation / Loader

Create `$BASE_SCRATCH/Python4ML.sh`:

```bash
cat <<EOF > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

# Environment path
export ENV_PREFIX="$ENV_PREFIX"

# Tykky executable path
export PATH="\$ENV_PREFIX/bin:\$PATH"

# Prefer the JAX GPU backend
export JAX_PLATFORMS="gpu"
EOF
```

Make it executable:

```bash
chmod +x "$BASE_SCRATCH/Python4ML.sh"
```

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

Confirm the Python version:

```bash
python --version
```

For CPU-only execution:

```bash
export JAX_PLATFORMS="cpu"
```

---

## VS Code Kernel Registration

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

List the available kernels:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
jupyter kernelspec list
```

Remove an obsolete kernel:

```bash
jupyter kernelspec uninstall -f <kernel_name>
```

---

## Validation

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

Use the CPU backend for validation on a CPU node:

```bash
export JAX_PLATFORMS="cpu"
```

Verify the core package versions:

```bash
python -c "
import sys
import jax
import equinox as eqx
import numpy as np
from importlib.metadata import version

print(f'Python:     {sys.version.split()[0]}')
print(f'JAX:        {jax.__version__}')
print(f'Equinox:    {eqx.__version__}')
print(f'jax2onnx:   {version(\"jax2onnx\")}')
print(f'NumPy:      {np.__version__}')
print(f'Devices:    {jax.devices()}')
"
```

Verify the main scientific packages:

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

Inspect the installed packages:

```bash
python -m pip freeze
```

Inspect the resolved requirements:

```bash
head -n 40 "$PYTHON_ROOT/requirements.txt"
```

---

## Dependency File Workflow

The dependency files serve different purposes:

```text
requirements.in
    Human-maintained list of direct dependencies.

requirements.txt
    uv-generated list containing exact direct and transitive versions.
```

The `requirements.txt` file is generated automatically during every Tykky build or update.

### Add a Package

Open `requirements.in`:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Add the new package on its own line.

For example:

```text
psutil
```

Save the file and apply an environment update or complete rebuild.

### Remove a Package

Open `requirements.in`:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Remove the package entry.

Save the file and apply an environment update or complete rebuild.

### Refresh Package Versions

The normal build command resolves the currently compatible package versions from `requirements.in`.

To preserve a particular dependency set, retain and commit the generated `requirements.txt`.

---

## Adding or Updating Packages

### 1. Edit the Direct Dependencies

Edit:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

### 2. Create the Update Script

Create `update4ML.sh`:

```bash
nano -m "$PYTHON_ROOT/update4ML.sh"
```

Insert:

```bash
#!/bin/bash
set -e

# Confirm that the build variables are available
: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"

# Redirect temporary files and package caches to scratch
export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Tykky update environment
python -m pip install --no-cache-dir uv

# Resolve the updated dependency set
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt"

# Install the updated dependency set
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.txt"

# Remove package caches
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x "$PYTHON_ROOT/update4ML.sh"
```

### 3. Apply the Update

Request a compute node:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

Load Tykky:

```bash
module purge
module load tykky
```

Configure the temporary build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Apply the update:

```bash
conda-containerize update \
    --post-install "$PYTHON_ROOT/update4ML.sh" \
    "$ENV_PREFIX"
```

Validate the updated environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
export JAX_PLATFORMS="cpu"

python --version
python -m pip freeze
```

---

## Rebuilding the Environment

Remove the existing environment:

```bash
rm -rf "$ENV_PREFIX"
```

Clear the temporary build directory:

```bash
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Request a compute node:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

Load Tykky:

```bash
module purge
module load tykky
```

Configure the build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
```

Run the build:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

The rebuild regenerates `requirements.txt` and installs the currently compatible dependency set.

---

## Complete Clean Installation

Run the global configuration block first.

Remove the existing environment and temporary build data:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
```

Create or confirm these files:

```text
$PYTHON_ROOT/base4ML.yml
$PYTHON_ROOT/requirements.in
$PYTHON_ROOT/extra4ML.sh
```

Make the installation script executable:

```bash
chmod +x "$PYTHON_ROOT/extra4ML.sh"
```

Request a compute node:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

Load Tykky:

```bash
module purge
module load tykky
```

Configure the build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
```

Build the environment:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

Create the loader:

```bash
cat <<EOF > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

export ENV_PREFIX="$ENV_PREFIX"
export PATH="\$ENV_PREFIX/bin:\$PATH"
export JAX_PLATFORMS="gpu"
EOF

chmod +x "$BASE_SCRATCH/Python4ML.sh"
```

Load and validate:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
export JAX_PLATFORMS="cpu"

python --version

python -c "
import jax
import equinox
import numpy
import onnx
import optax

print('ML environment is ready.')
print(jax.devices())
"
```

---

## Troubleshooting

### Total Environment Reset

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Run the Tykky build again.

### `requirements.txt` Is Missing

The file is created during the Tykky post-installation stage.

Run a complete build:

```bash
module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

### Package Resolution Fails

The uv output identifies the package constraints that cannot be resolved together.

Edit the direct dependency file:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Remove or replace the conflicting package and repeat the build.

### Package Installation Exceeds the Home Quota

The post-installation script redirects temporary files and caches to:

```text
$BASE_SCRATCH/.tykky_runtime
```

Run the global configuration block before starting the build.

### JAX Reports No GPU

Use a GPU allocation for GPU execution.

For CPU execution:

```bash
export JAX_PLATFORMS="cpu"
```

### Import Errors After an Update

Remove the environment and perform a complete rebuild:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Then run the normal Tykky build.

### Compiler Linkage Errors

Remove the current environment and rebuild it from the original configuration files:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Load Tykky and repeat the build.

---

## Notes

* The environment uses Python 3.12.
* `Harry` and `Dumbledore` are fictional placeholder values used in the public documentation.
* Replace `Harry` with the actual personal or shared directory under the CSC project.
* Replace `Dumbledore` with the preferred environment nickname.
* `PROJECT_USER_DIR` is not necessarily the same as the CSC login username.
* Dependency resolution runs inside the Tykky Python 3.12 build environment.
* No external Conda, Miniforge, Mamba, Python module, or virtual environment is required.
* `requirements.in` contains the direct dependency specifications.
* `requirements.txt` is generated automatically during the Tykky build.
* `requirements.txt` contains the exact direct and transitive versions selected by uv.
* Commit both dependency files when reproducible builds matter.
* Rebuild the environment after substantial Python, JAX, CUDA, compiler, or binary dependency changes.
* `jax[cuda12]` installs the CUDA 12-compatible JAX package set.
* GPU execution requires a GPU allocation and compatible host drivers.
* Use compute nodes for dependency resolution, package installation, and environment builds.
* Avoid performing large package installations directly on CSC login nodes.
