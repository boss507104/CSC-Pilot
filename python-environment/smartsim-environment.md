# SmartSim Environment Configuration

Last updated: 3 July 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack containing **SmartSim 0.8.0 + SmartRedis 0.6.1** on CSC supercomputers (**Puhti / Mahti / Roihu**). The setup focuses on coupling **JAX + Equinox + ONNX** models with parallel OpenFOAM solvers.

Instead of deploying traditional Conda or pip environments directly on the parallel filesystem, we use **Tykky** to package the Python stack inside a single-file container image. This design reduces the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

### Why Tykky?

* **Import Performance** — Library initialisation times drop from several minutes to seconds.
* **Reproducibility** — The complete Python execution stack remains packaged inside a single container image.
* **Startup Latency** — Fast environment startup is valuable for high-volume, short MPI jobs.
* **Isolation** — The Python dependency stack remains separated from the cluster host environment.

### Why uv?

This configuration uses **uv** inside the Tykky build environment to resolve and install Python packages.

* **Fast Resolution** — uv resolves large scientific Python dependency trees efficiently.
* **Compatible Dependency Selection** — uv selects mutually compatible direct and transitive package versions.
* **Simple Workflow** — Packages are installed directly from `requirements.in`.
* **Installed-State Record** — The final installed package versions are recorded in `requirements.txt`.
* **Post-Installation Validation** — `uv pip check` verifies the installed dependency relationships.
* **Explicit Copy Mode** — `--link-mode=copy` avoids unsupported hardlink operations between the uv cache and the temporary Tykky environment.

The direct package specifications in `requirements.in` contain the SmartSim-specific compatibility constraints:

```text
Python       3.11
SmartSim     0.8.0
SmartRedis   0.6.1-compatible source
JAX          0.6.2
ONNX         1.17.0
NumPy        < 2.0.0
protobuf     3.20.3
CMake        < 3.30.0
```

The dependency files have different roles:

```text
requirements.in
    Human-maintained direct package specifications and compatibility constraints.

requirements.txt
    Installed package versions recorded after a successful Tykky build.
```

The `requirements.txt` file is an installed-state snapshot rather than a separately compiled lockfile.

Dependency resolution and installation take place inside the Python 3.11 environment created by Tykky. No external Conda, Miniforge, Mamba, Python module, resolver environment, or virtual environment is required.

The SmartRedis Python client and the SmartRedis native C++/Fortran library are built separately:

```text
SmartRedis Python client
    Installed inside the Tykky Python environment.

SmartRedis native library
    Built outside Tykky for direct linkage with OpenFOAM, C++, and Fortran solvers.
```

The SmartSim database build installs its own ONNX-related Python dependencies. The installation script therefore reapplies `requirements.in` after `smart build` to restore the required `onnx==1.17.0` version before running the final dependency check.

This configuration forms part of the [CSC Environment Helpers Framework](https://github.com/boss507104/CSCEnvironmentHelpers). Production examples for coupling SmartSim, SmartRedis, OpenFOAM, JAX, and ONNX are maintained in the [SmartSim4CSC repository](https://github.com/boss507104/SmartSim4CSC).

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
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.11"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis"
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
        └── Utilities/                             # $BASE_SCRATCH
            ├── .tykky_runtime/                    # $TMP_BUILD_DIR
            ├── Python4SmartSim.sh
            ├── SmartRedis/                        # $SMARTREDIS_DIR
            │   ├── build/
            │   └── install/
            │       ├── include/
            │       ├── lib64/
            │       └── share/
            └── Python/                            # $PYTHON_ROOT
                ├── base4SmartSim.yml
                ├── extra4SmartSim.sh
                ├── update4SmartSim.sh
                ├── requirements.in
                ├── requirements.txt
                └── envs/
                    └── $ENV_NICKNAME-3.11/        # $ENV_PREFIX
```

---

## Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| **Python** | 3.11 | Base interpreter created by Tykky |
| **uv** | Latest available during the build | Dependency resolution, installation, and validation |
| **SmartSim** | 0.8.0 | Orchestration framework and database lifecycle management |
| **SmartRedis** | 0.6.1-compatible source | Python client and native C++/Fortran client library |
| **JAX** | 0.6.2 with CUDA 12 support | Array programming and automatic differentiation |
| **ONNX** | 1.17.0 | Compatible with `jax2onnx` and the pinned protobuf stack |
| **NumPy** | `< 2.0.0` | Compatibility constraint required by the SmartSim stack |
| **protobuf** | 3.20.3 | Compatibility layer used by SmartSim and ONNX tooling |
| **CMake** | `< 3.30.0` | SmartRedis and SmartSim native build compatibility |

---

## Installation Steps

### 1. Create the Configuration Files

Create the Python configuration directory:

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
```

### 1.1 Create the Base Conda Specification

Create `base4SmartSim.yml`:

```bash
nano -m base4SmartSim.yml
```

Insert:

```yaml
channels:
  - conda-forge
  - nodefaults

dependencies:
  - python=3.11
  - pip
  - git
  - compilers
  - cmake<3.30.0
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
numpy<2.0.0
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
jax[cuda12]==0.6.2
diffrax
equinox
jaxtyping
jax2onnx
jaxopt
einops
lineax
onnx==1.17.0
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
treeple
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

# --- Physics, CFD & SmartSim ---
cantera
foamlib
meshio
protobuf==3.20.3
smartsim==0.8.0

# --- Mathematical Tools ---
numba
pint
ruptures
sympy
tensorly

# --- Custom Utilities ---
DataGraph @ git+https://github.com/boss507104/DataGraph.git#subdirectory=DataGraph
eqx_io @ git+https://github.com/boss507104/CSC-HPC-Guide.git#subdirectory=utilities/eqx4smartredis

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

Create `extra4SmartSim.sh`:

```bash
nano -m extra4SmartSim.sh
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

# Limit simultaneous downloads on the CSC network
export UV_CONCURRENT_DOWNLOADS=4

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Python 3.11 Tykky build environment
python -m pip install --no-cache-dir uv

# Resolve and install the requested Python dependency set
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# Clone and install the patched SmartRedis Python client
rm -rf "$CW_BUILD_TMPDIR/SmartRedis"

git clone \
    https://github.com/boss507104/SmartRedis.git \
    "$CW_BUILD_TMPDIR/SmartRedis"

cd "$CW_BUILD_TMPDIR/SmartRedis"

# Add the fixed-width integer header required by newer compilers
grep -q '#include <cstdint>' src/cpp/tensorpack.cpp || \
    sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp

# Preserve the Tykky compiler flags
OLD_CFLAGS="${CFLAGS-}"
OLD_CXXFLAGS="${CXXFLAGS-}"
OLD_CPPFLAGS="${CPPFLAGS-}"
OLD_LDFLAGS="${LDFLAGS-}"

unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

# Install the SmartRedis Python client
python -m pip install --no-cache-dir .

# Restore the Tykky compiler flags
export CFLAGS="$OLD_CFLAGS"
export CXXFLAGS="$OLD_CXXFLAGS"
export CPPFLAGS="$OLD_CPPFLAGS"
export LDFLAGS="$OLD_LDFLAGS"

# Build the SmartSim database dependencies without unused ML backends
export USE_SYSTEMD=no

env \
    CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart clobber

env \
    CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart build \
        --device cpu \
        --skip-torch \
        --skip-tensorflow

# Restore packages modified by the SmartSim database build
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# Verify the final installed dependency relationships
uv pip check

# Record the installed package versions
python -m pip list \
    --format=freeze \
    | grep -v '^smartredis==' \
    | sort \
    > "$PYTHON_ROOT/requirements.txt"

# Remove temporary source and package caches
rm -rf "$CW_BUILD_TMPDIR/SmartRedis"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

The second `uv pip install` restores packages changed by `smart build`. In particular, the SmartSim ONNX backend may install `onnx==1.15.0`, while `jax2onnx` requires:

```text
onnx>=1.17.0,<1.19.0
```

The explicit `onnx==1.17.0` constraint restores a version compatible with both `jax2onnx` and the pinned `protobuf==3.20.3` stack.

The locally patched SmartRedis Python client is deliberately excluded from `requirements.txt`. It is installed separately from the SmartRedis source repository during every build.

Make the script executable:

```bash
chmod +x extra4SmartSim.sh
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

Confirm that the configuration files exist:

```bash
ls -l \
    "$PYTHON_ROOT/base4SmartSim.yml" \
    "$PYTHON_ROOT/extra4SmartSim.sh" \
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
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

During the build, Tykky performs these operations:

```text
1. Creates the Python 3.11 base environment.
2. Runs extra4SmartSim.sh inside the environment.
3. Installs uv.
4. Resolves and installs the packages listed in requirements.in.
5. Installs the patched SmartRedis Python client.
6. Builds the SmartSim database dependencies.
7. Restores the package versions specified in requirements.in.
8. Verifies the final installed dependency relationships.
9. Records the installed package versions in requirements.txt.
10. Packages the completed environment into the Tykky image.
```

After the build completes:

```bash
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements.txt"
```

Inspect the critical installed versions:

```bash
grep -E \
    '^(jax|numpy|onnx|protobuf|smartsim)==' \
    "$PYTHON_ROOT/requirements.txt"
```

The expected ONNX version is:

```text
onnx==1.17.0
```

---

## 3. Build the SmartRedis Native Library

The SmartRedis Python client installed inside the Tykky environment is not sufficient for OpenFOAM, C++, or Fortran solver linkage.

Build the native SmartRedis C++, C, and Fortran libraries separately on a compute node using the compiler modules available on the target CSC system.

Start from a clean module environment:

```bash
module purge
```

Example for Roihu:

```bash
module load gcc/13.4.0
module load cmake/3.26.5
```

Example for Mahti:

```bash
module load gcc/13.1.0
module load cmake/3.28.6
module load git
```

Clone the SmartRedis source:

```bash
cd "$BASE_SCRATCH"

rm -rf "$SMARTREDIS_DIR"

git clone \
    https://github.com/boss507104/SmartRedis.git \
    "$SMARTREDIS_DIR"

cd "$SMARTREDIS_DIR"
```

Apply the compiler compatibility patch:

```bash
grep -q '#include <cstdint>' src/cpp/tensorpack.cpp || \
    sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp
```

Remove previous native build artefacts:

```bash
rm -rf build install
```

Build the C++, C, and Fortran libraries:

```bash
env \
    -u CFLAGS \
    -u CXXFLAGS \
    -u CPPFLAGS \
    -u LDFLAGS \
    -u CC \
    -u CXX \
    -u FC \
    CC=gcc \
    CXX=g++ \
    FC=gfortran \
    make lib-with-fortran
```

Verify the native installation:

```bash
find "$SMARTREDIS_DIR/install" \
    -maxdepth 3 \
    -type f \
    | sort
```

Inspect the library directory:

```bash
# If lib64 does not exist on the target system, replace lib64 with lib.
ls -la "$SMARTREDIS_DIR/install/lib64"
```

Verify the Fortran shared library:

```bash
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library installed successfully."
```

Inspect its dynamic dependencies:

```bash
ldd "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so"
```

---

## Environment Activation / Loader

Create `$BASE_SCRATCH/Python4SmartSim.sh`:

```bash
cat <<EOF > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash

# Compiler runtime
module load gcc/13.4.0

# Environment paths
export ENV_PREFIX="$ENV_PREFIX"
export SMARTREDIS_DIR="$SMARTREDIS_DIR"

# Tykky executable path
export PATH="\$ENV_PREFIX/bin:\$PATH"

# SmartRedis native libraries
# If lib64 does not exist on the target system, replace lib64 with lib.
export LD_LIBRARY_PATH="\$SMARTREDIS_DIR/install/lib64:\${LD_LIBRARY_PATH:-}"

# SmartRedis CMake package files
export CMAKE_PREFIX_PATH="\$SMARTREDIS_DIR/install:\${CMAKE_PREFIX_PATH:-}"

# SmartSim database startup tolerance
export SMARTSIM_DB_FILE_PARSE_TRIALS=600

# Prefer the JAX GPU backend
export JAX_PLATFORMS="gpu"
EOF
```

Replace `gcc/13.4.0` with the GCC module used to build SmartRedis on the target CSC system.

Make the loader executable:

```bash
chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"
```

Load the environment:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
```

Confirm the Python version:

```bash
python --version
```

Verify the SmartRedis native library path:

```bash
echo "$LD_LIBRARY_PATH"
```

Verify the SmartRedis CMake package path:

```bash
echo "$CMAKE_PREFIX_PATH"
```

For CPU-only execution:

```bash
export JAX_PLATFORMS="cpu"
```

---

## VS Code Kernel Registration

Create the kernel directory:

```bash
mkdir -p "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-smartsim"
```

Create `kernel.json`:

```bash
cat <<EOF > "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-smartsim/kernel.json"
{
  "argv": [
    "$ENV_PREFIX/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.11 ($ENV_NICKNAME Tykky SmartSim)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF
```

Confirm the registration:

```bash
echo "Jupyter kernel '$ENV_NICKNAME-smartsim' has been registered."
```

List the available kernels:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
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
source "$BASE_SCRATCH/Python4SmartSim.sh"
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
from smartsim._core.config import CONFIG

print(f'Python:      {sys.version.split()[0]}')
print(f'SmartSim:    {version(\"smartsim\")}')
print(f'SmartRedis:  {version(\"smartredis\")}')
print(f'JAX:         {jax.__version__}')
print(f'Equinox:     {eqx.__version__}')
print(f'jax2onnx:    {version(\"jax2onnx\")}')
print(f'ONNX:        {version(\"onnx\")}')
print(f'NumPy:       {np.__version__}')
print(f'protobuf:    {version(\"protobuf\")}')
print(f'Devices:     {jax.devices()}')
print(f'DB Exec:     {CONFIG.database_exe}')
"
```

Verify the scientific and SmartSim packages:

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
import smartredis
import smartsim
import xarray

print('Core SmartSim, ML, and scientific packages imported successfully.')
"
```

Verify the installed dependency relationships:

```bash
uv pip check
```

Run the SmartSim integrity diagnostic:

```bash
smart validate --device cpu
```

Missing PyTorch or TensorFlow backends are expected because the SmartSim database build deliberately excludes them.

Inspect the installed package versions:

```bash
python -m pip list --format=freeze
```

Inspect the recorded package snapshot:

```bash
head -n 40 "$PYTHON_ROOT/requirements.txt"
```

Verify the native SmartRedis libraries:

```bash
# If lib64 does not exist on the target system, replace lib64 with lib.
ls -la "$SMARTREDIS_DIR/install/lib64"
```

Verify the Fortran shared library:

```bash
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library is available."
```

Inspect native linkage:

```bash
ldd "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so"
```

Verify the SmartRedis CMake package files:

```bash
find "$SMARTREDIS_DIR/install/share/cmake" \
    -maxdepth 3 \
    -type f \
    | sort
```

To validate the complete data path, run a JAX to ONNX to SmartRedis graph submission test on a compute node.

---

## Dependency File Workflow

The dependency files serve different purposes:

```text
requirements.in
    Human-maintained direct dependencies and SmartSim compatibility constraints.

requirements.txt
    Installed package versions recorded after a successful build.
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

Perform a complete rebuild so that the removed package and unused transitive dependencies are no longer included.

### Preserve the Compatibility Constraints

Keep the following entries unless the complete stack is deliberately revalidated:

```text
numpy<2.0.0
jax[cuda12]==0.6.2
onnx==1.17.0
protobuf==3.20.3
smartsim==0.8.0
```

The Python and CMake constraints remain in `base4SmartSim.yml`:

```text
python=3.11
cmake<3.30.0
```

### Reproduce an Existing Installed Package Set

A previously generated `requirements.txt` records the package versions installed in that environment.

To reproduce those versions, temporarily replace both dependency installation commands in `extra4SmartSim.sh`:

```bash
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"
```

with:

```bash
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.txt"
```

The patched SmartRedis Python client must still be installed separately from its source repository.

For ordinary development builds, continue using `requirements.in`.

---

## Adding or Updating Python Packages

### 1. Edit the Direct Dependencies

Edit:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Preserve the SmartSim compatibility constraints.

### 2. Create the Update Script

Create `update4SmartSim.sh`:

```bash
nano -m "$PYTHON_ROOT/update4SmartSim.sh"
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

# Limit simultaneous downloads on the CSC network
export UV_CONCURRENT_DOWNLOADS=4

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Tykky update environment
python -m pip install --no-cache-dir uv

# Resolve and install the requested dependency set
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# Reinstall the patched SmartRedis Python client
rm -rf "$CW_BUILD_TMPDIR/SmartRedis"

git clone \
    https://github.com/boss507104/SmartRedis.git \
    "$CW_BUILD_TMPDIR/SmartRedis"

cd "$CW_BUILD_TMPDIR/SmartRedis"

grep -q '#include <cstdint>' src/cpp/tensorpack.cpp || \
    sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp

OLD_CFLAGS="${CFLAGS-}"
OLD_CXXFLAGS="${CXXFLAGS-}"
OLD_CPPFLAGS="${CPPFLAGS-}"
OLD_LDFLAGS="${LDFLAGS-}"

unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

python -m pip install --no-cache-dir .

export CFLAGS="$OLD_CFLAGS"
export CXXFLAGS="$OLD_CXXFLAGS"
export CPPFLAGS="$OLD_CPPFLAGS"
export LDFLAGS="$OLD_LDFLAGS"

# Rebuild the SmartSim database dependencies
export USE_SYSTEMD=no

env \
    CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart clobber

env \
    CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart build \
        --device cpu \
        --skip-torch \
        --skip-tensorflow

# Restore packages modified by the SmartSim database build
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# Verify the final installed dependency relationships
uv pip check

# Record the updated installed package versions
python -m pip list \
    --format=freeze \
    | grep -v '^smartredis==' \
    | sort \
    > "$PYTHON_ROOT/requirements.txt"

# Remove temporary source and package caches
rm -rf "$CW_BUILD_TMPDIR/SmartRedis"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x "$PYTHON_ROOT/update4SmartSim.sh"
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
    --post-install "$PYTHON_ROOT/update4SmartSim.sh" \
    "$ENV_PREFIX"
```

Load and validate the updated environment:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
export JAX_PLATFORMS="cpu"

python --version
uv pip check
```

Updating the Tykky Python environment does not rebuild the separate native SmartRedis library under `$SMARTREDIS_DIR`.

Rebuild the native library separately when its source, compiler, or ABI changes.

---

## Rebuilding the Complete Environment

### 1. Remove the Tykky Environment

```bash
rm -rf "$ENV_PREFIX"
```

### 2. Clear the Temporary Build Directory

```bash
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

### 3. Request a Compute Node

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

### 4. Load Tykky

```bash
module purge
module load tykky
```

### 5. Configure the Build Directory

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
```

### 6. Verify the Configuration Files

```bash
ls -l \
    "$PYTHON_ROOT/base4SmartSim.yml" \
    "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/requirements.in"
```

### 7. Rebuild the Tykky Environment

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

The rebuild resolves the dependency set from `requirements.in`, restores packages modified by `smart build`, validates the final environment, and records the installed versions in `requirements.txt`.

### 8. Rebuild the SmartRedis Native Library

Rebuild `$SMARTREDIS_DIR` when:

* the SmartRedis source changes;
* the compiler module changes;
* the target CSC system changes;
* the C++, C, or Fortran ABI changes;
* native linkage errors appear.

---

## Complete Clean Installation

Run the global configuration block.

Remove the existing Python environment, temporary build data, and native SmartRedis installation:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
rm -rf "$SMARTREDIS_DIR"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
```

Confirm that the Python configuration files exist:

```bash
ls -l \
    "$PYTHON_ROOT/base4SmartSim.yml" \
    "$PYTHON_ROOT/requirements.in" \
    "$PYTHON_ROOT/extra4SmartSim.sh"
```

Make the post-installation script executable:

```bash
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"
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

Build the Tykky environment:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

Build the SmartRedis native library using the compiler modules for the target CSC system.

Create the loader:

```bash
cat <<EOF > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash

module load gcc/13.4.0

export ENV_PREFIX="$ENV_PREFIX"
export SMARTREDIS_DIR="$SMARTREDIS_DIR"

export PATH="\$ENV_PREFIX/bin:\$PATH"

# If lib64 does not exist on the target system, replace lib64 with lib.
export LD_LIBRARY_PATH="\$SMARTREDIS_DIR/install/lib64:\${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="\$SMARTREDIS_DIR/install:\${CMAKE_PREFIX_PATH:-}"

export SMARTSIM_DB_FILE_PARSE_TRIALS=600
export JAX_PLATFORMS="gpu"
EOF

chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"
```

Load and validate:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
export JAX_PLATFORMS="cpu"

python --version

python -c "
import jax
import equinox
import numpy
import onnx
import smartredis
import smartsim

print('SmartSim environment is ready.')
print(jax.devices())
"

ls -la "$SMARTREDIS_DIR/install/lib64"
```

---

## Troubleshooting

### Total Environment Reset

Remove the Tykky environment and temporary build directory:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Repeat the normal Tykky build.

Remove the native SmartRedis installation only when a native rebuild is required:

```bash
rm -rf "$SMARTREDIS_DIR"
```

### `requirements.txt` Is Missing

The file is generated only after the Python packages, SmartRedis Python client, SmartSim database dependencies, dependency restoration, and final compatibility check have completed successfully.

Run a complete Tykky build:

```bash
module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

### Package Resolution Fails

Open the direct dependency file:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Preserve these compatibility constraints:

```text
numpy<2.0.0
jax[cuda12]==0.6.2
onnx==1.17.0
protobuf==3.20.3
smartsim==0.8.0
```

Repeat the build after correcting the conflicting dependency.

### `jax2onnx` Reports an Incompatible ONNX Version

The incompatible state appears as:

```text
jax2onnx requires onnx>=1.17.0,<1.19.0
but onnx 1.15.0 is installed
```

The SmartSim database build may install `onnx==1.15.0` for its ONNX Runtime backend.

Keep this constraint in `requirements.in`:

```text
onnx==1.17.0
```

Ensure that `extra4SmartSim.sh` reapplies the dependency file after `smart build`:

```bash
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check
```

Do not remove the final `uv pip check`. It confirms that the restored environment is internally consistent.

### uv Reports a Hardlink Warning

The uv cache and temporary Tykky Python environment reside on different filesystems.

Use explicit copy mode:

```bash
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"
```

This avoids the failed hardlink attempt and suppresses the fallback warning.

### Package Downloads Are Slow

Large binary packages and CUDA dependencies may require substantial downloads.

The installation script limits uv to four simultaneous downloads:

```bash
export UV_CONCURRENT_DOWNLOADS=4
```

The total build time still depends on the network connection between the CSC compute node and the external package servers.

### Package Installation Exceeds the Home Quota

Run the global configuration block before starting the build.

Temporary files and package caches are redirected to:

```text
$BASE_SCRATCH/.tykky_runtime
```

### JAX Reports No GPU

For CPU execution:

```bash
export JAX_PLATFORMS="cpu"
```

For GPU execution, run inside a GPU allocation.

### SmartSim Cannot Locate the Database Executable

Inspect the SmartSim configuration:

```bash
python -c "
from smartsim._core.config import CONFIG
print(CONFIG.database_exe)
"
```

Rebuild the database dependencies:

```bash
export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-torch \
    --skip-tensorflow
```

After rebuilding the database dependencies, restore the requested Python package set:

```bash
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check
```

### SmartRedis Native Library Cannot Be Found

Inspect the native installation:

```bash
# If lib64 does not exist on the target system, replace lib64 with lib.
ls -la "$SMARTREDIS_DIR/install/lib64"
```

Verify the runtime library path:

```bash
echo "$LD_LIBRARY_PATH"
```

Reload the environment:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
```

Verify the Fortran shared library:

```bash
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library is available."
```

### SmartRedis Shared Library Dependencies Are Missing

Inspect the Fortran shared library:

```bash
ldd "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so"
```

Any entry ending in `not found` indicates a missing compiler runtime or dependent shared library.

Load the same GCC module used for the native build:

```bash
module load gcc/13.4.0
```

Reload the environment:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
```

### SmartRedis Compiler Errors

Verify the loaded compiler and CMake modules:

```bash
module list
gcc --version
gfortran --version
cmake --version
```

Confirm that the `<cstdint>` patch exists:

```bash
grep -n '#include <cstdint>' \
    "$SMARTREDIS_DIR/src/cpp/tensorpack.cpp"
```

Remove previous build artefacts:

```bash
cd "$SMARTREDIS_DIR"
rm -rf build install
```

Repeat the native build.

### SmartSim Reports Incompatible Pointer Errors

Rebuild the SmartSim database dependencies using:

```bash
env \
    CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart build \
        --device cpu \
        --skip-torch \
        --skip-tensorflow
```

Restore the Python dependency set afterwards:

```bash
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check
```

### Import Errors After an Update

Verify the installed dependency relationships:

```bash
uv pip check
```

Inspect the installed versions:

```bash
python -m pip list --format=freeze
```

Compare them with:

```bash
cat "$PYTHON_ROOT/requirements.txt"
```

Perform a complete rebuild instead of stacking additional incremental updates when the environment becomes inconsistent.

---

## SmartSim Deployment Track

This environment provides the software foundation for coupled multi-physics simulations in which parallel solvers exchange tensors and machine-learning models through SmartRedis.

Typical workflows include:

* running the SmartSim Orchestrator on node-local storage;
* launching OpenFOAM solvers through Slurm;
* tracing Equinox models into ONNX graphs;
* uploading ONNX models to SmartRedis;
* evaluating the models during solver execution;
* exchanging distributed CFD fields through the Redis database;
* linking external C++ or Fortran solvers against the native SmartRedis client.

The complete production architecture, Slurm templates, database placement strategies, and model-injection examples are maintained in the [SmartSim4CSC reference repository](https://github.com/boss507104/SmartSim4CSC).

---

## Notes

* The environment uses Python 3.11.
* `Harry` and `Dumbledore` are fictional placeholder values used in this public documentation.
* Replace `Harry` with the actual personal or shared directory under the CSC project.
* Replace `Dumbledore` with the preferred environment nickname.
* `PROJECT_USER_DIR` is not necessarily the same as the CSC login username.
* Dependency resolution and installation run inside the Tykky Python 3.11 environment.
* No external Conda, Miniforge, Mamba, Python module, resolver environment, or virtual environment is required.
* `requirements.in` contains the direct dependencies and SmartSim compatibility constraints.
* `requirements.txt` records the package versions installed during the build.
* `requirements.txt` is an installed-state snapshot rather than a separately compiled lockfile.
* New builds from `requirements.in` may install newer compatible versions of unconstrained packages.
* Use a previously generated `requirements.txt` when the same installed package versions must be reconstructed.
* The patched SmartRedis Python client is installed separately from its source repository.
* The SmartRedis Python client is excluded from `requirements.txt`.
* Preserve the SmartSim, JAX, ONNX, NumPy, protobuf, Python, and CMake compatibility constraints.
* `onnx==1.17.0` satisfies the `jax2onnx` requirement while remaining compatible with the pinned protobuf stack.
* The SmartSim database build may temporarily install `onnx==1.15.0`.
* The dependency set must therefore be reapplied after `smart build`.
* The final `uv pip check` must run only after the dependency restoration step.
* Every `uv pip install` uses `--link-mode=copy` because the uv cache and Tykky target environment reside on different filesystems.
* The SmartRedis Python client and SmartRedis native libraries serve different purposes.
* The native SmartRedis libraries are expected under `install/lib64`.
* If the target system installs native libraries under `install/lib`, replace `lib64` with `lib` in the loader and validation commands.
* `CMAKE_PREFIX_PATH` points to the SmartRedis installation prefix for downstream CMake projects.
* The native SmartRedis library must be rebuilt when its compiler, source, target system, or ABI changes.
* `UV_CONCURRENT_DOWNLOADS=4` limits simultaneous external package downloads.
* Missing PyTorch and TensorFlow backends in `smart validate` are intentional.
* GPU execution requires a GPU allocation and compatible host drivers.
* Use compute nodes for package installation, SmartSim database compilation, SmartRedis native compilation, and computational workloads.
* Avoid performing large package installations or native builds directly on CSC login nodes.
* Prefer a complete rebuild over repeated incremental updates when the dependency set changes substantially.
