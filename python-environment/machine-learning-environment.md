# ML Environment Configuration

Last updated: 8 July 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack optimised for modern machine learning, statistics, and detailed chemical kinetics analysis on CSC supercomputers (**Puhti / Mahti / Roihu**). The setup focuses on a high-throughput **JAX + Equinox + ONNX** development ecosystem.

Instead of deploying traditional Conda or pip environments directly on the parallel filesystem, we use **Tykky** to package the Python stack inside a single-file container image. This design reduces the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

Roihu requires separate Tykky environments for CPU and GPU nodes because Roihu CPU nodes use **x64 / amd64**, while Roihu GPU nodes use **ARM64 / aarch64**. A Tykky container built on one architecture should not be expected to run on the other.

### Why Tykky?

* **Import Performance** — Library initialisation times drop from several minutes to seconds.
* **Reproducibility** — The complete execution stack remains packaged inside a single container image.
* **Startup Latency** — Fast environment startup is valuable for high-volume, short MPI jobs.
* **Isolation** — The Python dependency stack remains separated from the cluster host environment.

### Why uv?

This configuration uses **uv** inside the Tykky build environment to install Python packages.

* **Fast Installation** — uv downloads and installs large scientific Python dependency sets efficiently.
* **Compatible Dependency Selection** — uv resolves compatible direct and transitive package versions during installation.
* **Simple Workflow** — Packages are installed directly from `requirements.in`.
* **Installed-State Record** — The final installed package versions are recorded in `requirements.txt`.

The direct package specifications in `requirements.in` intentionally avoid strict version pins. Each new build may therefore install newer compatible package versions.

The two dependency files have different roles:

```text
requirements.in   Human-maintained direct package specifications
requirements.txt  Exact installed package versions recorded after the build
```

Dependency installation takes place inside the Python 3.12 environment created by Tykky. No external Conda, Miniforge, Mamba, Python module, or virtual environment is required.

---

## Global Configuration

Execute one of the following configuration blocks depending on the target node architecture.

Use the **x64 configuration** for Roihu CPU nodes and other x86_64 systems such as Puhti and Mahti.

Use the **ARM64 configuration** for Roihu GPU nodes.

### x64 Configuration for CPU Nodes

Run this block before building the CPU-node environment:

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export PROJECT_USER_DIR="Harry"             # Your directory under the CSC project
export ENV_NICKNAME="Dumbledore"            # Desired environment name
export ENV_ARCH="x64"                       # x64 / amd64 environment for CPU nodes
# --- USER CONFIGURATION END ---

# Derived paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_$ENV_ARCH"

# Initialise directories without removing existing environments
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "x64 configuration loaded for $CSC_PROJECT."
echo "ENV_PREFIX=$ENV_PREFIX"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
```

### ARM64 Configuration for Roihu GPU Nodes

Run this block before building the Roihu GPU-node environment:

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export PROJECT_USER_DIR="Harry"             # Your directory under the CSC project
export ENV_NICKNAME="Dumbledore"            # Desired environment name
export ENV_ARCH="arm64"                     # ARM64 / aarch64 environment for Roihu GPU nodes
# --- USER CONFIGURATION END ---

# Derived paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_$ENV_ARCH"

# Initialise directories without removing existing environments
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "ARM64 configuration loaded for $CSC_PROJECT."
echo "ENV_PREFIX=$ENV_PREFIX"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
```

The configuration variables represent:

```text
CSC_PROJECT       CSC project ID
PROJECT_USER_DIR  Personal or shared directory under the CSC project
ENV_NICKNAME      Name assigned to the Python environment
ENV_ARCH          Target architecture: x64 or arm64
```

For example:

```bash
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"
export ENV_ARCH="x64"
```

The resulting x64 environment path is:

```text
/scratch/project_xxxxxxx/Harry/Utilities/Python/envs/Dumbledore-3.12-x64
```

The resulting ARM64 environment path is:

```text
/scratch/project_xxxxxxx/Harry/Utilities/Python/envs/Dumbledore-3.12-arm64
```

`Harry` and `Dumbledore` are fictional placeholder values used in this public documentation. Replace them with your actual project directory and preferred environment name.

`PROJECT_USER_DIR` is not necessarily the same as your CSC login username. It identifies the directory located directly under the CSC project scratch path.

**Directory Structure**

```plaintext
/scratch/
└── $CSC_PROJECT/
    └── $PROJECT_USER_DIR/
        └── Utilities/                             # $BASE_SCRATCH
            ├── .tykky_runtime_x64/                # x64 temporary build directory
            ├── .tykky_runtime_arm64/              # ARM64 temporary build directory
            ├── Python4ML.sh
            └── Python/                            # $PYTHON_ROOT
                ├── base4ML.yml
                ├── extra4ML.sh
                ├── update4ML.sh
                ├── requirements.in
                ├── requirements.txt
                └── envs/
                    ├── $ENV_NICKNAME-3.12-x64/    # x64 Tykky environment
                    └── $ENV_NICKNAME-3.12-arm64/  # ARM64 Tykky environment
```

---

## Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| **Python** | 3.12 | Base interpreter created by Tykky |
| **uv** | Latest available during the build | Dependency resolution and installation |
| **NumPy** | Compatible version selected during installation | Core numerical array backend |
| **JAX** | Compatible CUDA 12 release selected during installation | Array programming and automatic differentiation |
| **Equinox** | Compatible version selected during installation | Neural-network and PyTree framework for JAX |
| **ONNX / jax2onnx** | Compatible versions selected during installation | Model export and interoperability |

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
export UV_CONCURRENT_DOWNLOADS=4

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Python 3.12 Tykky build environment
python -m pip install --no-cache-dir uv

# Install the direct dependencies and their compatible transitive dependencies
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.in"

# Record the exact installed package versions
python -m pip freeze > "$PYTHON_ROOT/requirements.txt"

# Remove package caches
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x extra4ML.sh
```

---

## 2. Build the Tykky Container

Build the x64 environment from an x64 CPU node.

Build the ARM64 environment from a Roihu GPU node.

### 2.1 Request an x64 CPU Build Node

Use this for the x64 environment:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

### 2.2 Request an ARM64 Roihu GPU Build Node

Use this for the ARM64 environment:

```bash
sinteractive \
    --account "$CSC_PROJECT" \
    --gpu \
    --cores 36 \
    --time 01:30:00
```

### 2.3 Build the Environment

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

Confirm that the configuration files exist:

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

During the build, Tykky performs these operations:

```text
1. Creates the Python 3.12 base environment.
2. Runs extra4ML.sh inside the environment.
3. Installs uv.
4. Installs the packages listed in requirements.in.
5. Records the installed package versions in requirements.txt.
6. Packages the complete environment into the Tykky image.
```

After the build completes:

```bash
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements.txt"
```

---

## Environment Activation / Loader

Create `$BASE_SCRATCH/Python4ML.sh`:

```bash
cat <<'EOF' > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

# Project configuration
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"

# Derived paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"

# Select the matching Tykky environment for the current node architecture
case "$(uname -m)" in
    x86_64)
        export ENV_ARCH="x64"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        return 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"

# Tykky executable path
export PATH="$ENV_PREFIX/bin:$PATH"

# Prefer the JAX GPU backend by default
export JAX_PLATFORMS="gpu"
EOF
```

Edit the loader and replace `project_xxxxxxx`, `Harry`, and `Dumbledore` with your actual values:

```bash
nano -m "$BASE_SCRATCH/Python4ML.sh"
```

Make the loader executable:

```bash
chmod +x "$BASE_SCRATCH/Python4ML.sh"
```

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

Confirm the selected environment:

```bash
echo "$ENV_PREFIX"
python --version
```

For CPU-only execution:

```bash
export JAX_PLATFORMS="cpu"
```

---

## VS Code Kernel Registration

The kernel should be registered separately on each architecture after loading the matching environment.

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

Create the kernel directory:

```bash
mkdir -p "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-ml-$(uname -m)"
```

Create `kernel.json`:

```bash
cat <<EOF > "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-ml-$(uname -m)/kernel.json"
{
  "argv": [
    "$ENV_PREFIX/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.12 ($ENV_NICKNAME ML $(uname -m))",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF
```

Confirm the registration:

```bash
echo "Jupyter kernel '$ENV_NICKNAME-ml-$(uname -m)' has been registered."
```

List the available kernels:

```bash
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

Use the CPU backend when validating on a CPU node:

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

Inspect the installed package versions:

```bash
python -m pip freeze
```

Inspect the recorded package versions:

```bash
head -n 40 "$PYTHON_ROOT/requirements.txt"
```

---

## Dependency File Workflow

The dependency files serve different purposes:

```text
requirements.in
    Human-maintained direct package specifications.

requirements.txt
    Exact installed package versions recorded after installation.
```

### Add a Package

Open `requirements.in`:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Add the package on its own line.

For example:

```text
psutil
```

Save the file and rebuild or update the environment.

### Remove a Package

Open `requirements.in`:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Remove the package entry.

Perform a complete rebuild to ensure that the removed package and unused transitive dependencies are no longer present.

### Reproduce an Existing Installed Package Set

The generated `requirements.txt` contains the installed versions from the completed build.

To create another environment from those exact versions, temporarily replace this line in `extra4ML.sh`:

```bash
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.in"
```

with:

```bash
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.txt"
```

For normal development builds, continue using `requirements.in`.

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

export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Tykky update environment
python -m pip install --no-cache-dir uv

# Install the current direct dependency set
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.in"

# Record the updated installed package versions
python -m pip freeze > "$PYTHON_ROOT/requirements.txt"

# Remove package caches
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x "$PYTHON_ROOT/update4ML.sh"
```

### 3. Apply the Update

Request the same architecture as the environment being updated.

For x64:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

For ARM64:

```bash
sinteractive \
    --account "$CSC_PROJECT" \
    --gpu \
    --cores 36 \
    --time 01:30:00
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

Load and validate the updated environment:

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

Request the same architecture as the environment being rebuilt.

For x64:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

For ARM64:

```bash
sinteractive \
    --account "$CSC_PROJECT" \
    --gpu \
    --cores 36 \
    --time 01:30:00
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

The rebuild installs the package set defined in `requirements.in` and records the installed versions in `requirements.txt`.

---

## Complete Clean Installation

Run the global configuration block for the target architecture.

Remove the existing environment and temporary build data:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
```

Confirm that these files exist:

```bash
ls -l \
    "$PYTHON_ROOT/base4ML.yml" \
    "$PYTHON_ROOT/requirements.in" \
    "$PYTHON_ROOT/extra4ML.sh"
```

Make the installation script executable:

```bash
chmod +x "$PYTHON_ROOT/extra4ML.sh"
```

Request the same architecture as the environment being installed.

For x64:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

For ARM64:

```bash
sinteractive \
    --account "$CSC_PROJECT" \
    --gpu \
    --cores 36 \
    --time 01:30:00
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

Create the architecture-aware loader:

```bash
cat <<'EOF' > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

# Project configuration
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"

# Derived paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"

# Select the matching Tykky environment for the current node architecture
case "$(uname -m)" in
    x86_64)
        export ENV_ARCH="x64"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        return 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"

# Tykky executable path
export PATH="$ENV_PREFIX/bin:$PATH"

# Prefer the JAX GPU backend by default
export JAX_PLATFORMS="gpu"
EOF
```

Edit the loader and replace `project_xxxxxxx`, `Harry`, and `Dumbledore` with your actual values.

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

Run the normal Tykky build again on the matching architecture.

### `requirements.txt` Is Missing

The file is generated after package installation.

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

### Package Installation Fails

Edit the direct dependency file:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Remove or replace the package identified in the installation error and repeat the build.

### Package Installation Exceeds the Home Quota

Run the global configuration block before starting the build.

Temporary files and caches are redirected to architecture-specific scratch paths:

```text
$BASE_SCRATCH/.tykky_runtime_x64
$BASE_SCRATCH/.tykky_runtime_arm64
```

### Container Architecture Mismatch

If an environment built on an x64 node is used on a Roihu GPU node, execution may fail with an error similar to:

```text
the image's architecture (amd64) could not run on the host's (arm64)
```

Build and use a separate ARM64 environment on the Roihu GPU node.

If an ARM64 environment is used on an x64 CPU node, build and use the x64 environment instead.

### JAX Reports No GPU

For CPU execution:

```bash
export JAX_PLATFORMS="cpu"
```

For GPU execution, run inside a GPU allocation and use the ARM64 environment on Roihu GPU nodes.

### Import Errors After an Update

Remove the environment and perform a complete rebuild:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Then run the normal Tykky build on the matching architecture.

### Compiler Linkage Errors

Remove the current environment and rebuild it from the configuration files:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Load Tykky and repeat the build.

---

## Notes

* The environment uses Python 3.12.
* Roihu CPU nodes use x64 / amd64.
* Roihu GPU nodes use ARM64 / aarch64.
* Build a separate Tykky environment for each architecture that you need to use.
* Do not expect an x64 Tykky container to run on Roihu GPU nodes.
* Do not expect an ARM64 Tykky container to run on Roihu CPU nodes.
* `Harry` and `Dumbledore` are fictional placeholder values used in the public documentation.
* Replace `Harry` with the actual personal or shared directory under the CSC project.
* Replace `Dumbledore` with the preferred environment nickname.
* `PROJECT_USER_DIR` is not necessarily the same as the CSC login username.
* Dependency installation runs inside the Tykky Python 3.12 build environment.
* No external Conda, Miniforge, Mamba, Python module, or virtual environment is required.
* `requirements.in` contains the direct dependency specifications.
* `requirements.txt` records the package versions installed during the build.
* New builds from `requirements.in` may install newer compatible versions.
* Use the recorded `requirements.txt` when an exact package set must be reproduced.
* `jax[cuda12]` installs the CUDA 12-compatible JAX package set.
* GPU execution requires a GPU allocation and compatible host drivers.
* Use compute nodes for package installation and environment builds.
* Avoid performing large package installations directly on CSC login nodes.
