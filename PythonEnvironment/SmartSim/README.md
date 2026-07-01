# SmartSim Environment Configuration

Last updated: 25 June 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack containing **SmartSim 0.8.0 + SmartRedis 0.6.1** on CSC supercomputers (**Puhti / Mahti**). The setup focuses heavily on coupling **JAX + Equinox + ONNX** models with parallel OpenFOAM solvers.

Instead of deploying traditional conda or pip environments, we leverage **Tykky** to wrap the Python stack inside a single-file container image. This design addresses the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

### Why Tykky?

* **Import Performance** — Library initialisation times drop from several minutes down to seconds.
* **Reproducibility** — The entire execution stack remains frozen within a single, immutable file image.
* **Startup Latency** — Immediate execution startup proves critical when running high-volume, short MPI jobs.
* **Isolation** — The complete Python dependency stack remains strictly separated from cluster host system modules.

> [!NOTE]
> This configuration forms an integral module of the [CSC Environment Helpers Framework](https://www.google.com/search?q=https://github.com/boss507104/CSCEnvironmentHelpers). For a complete reference architecture, including comprehensive production examples running via SLURM on local node spaces or across **Mahti** compute blocks, consult the original development repository at [SmartSim4CSC](https://github.com/boss507104/SmartSim4CSC/blob/main/README.md).

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
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.11"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis"
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
            ├── SmartRedis/                 # $SMARTREDIS_DIR
            └── Python/                     # $PYTHON_ROOT
                └── envs/
                    └── $ENV_NICKNAME-3.11/  # $ENV_PREFIX

```

> [!TIP]
> Move your active development folders into **your own Utilities directory** located on the parallel scratch filesystem. Verify that you have explicitly initialised this path via `mkdir -p` before starting the compilation toolchains.

---

## Dependency Overview

| Package | Version | Purpose / Constraint |
| --- | --- | --- |
| **Python** | 3.11 | Tykky container engine foundation base layer. |
| **smartsim** | 0.8.0 | Orchestrator infrastructure framework management engine. |
| **smartredis** | 0.6.1 | Client runtime (also built from source for Fortran linkers). |
| **jax[cuda12]** | 0.6.2 | Main array manipulation and automatic differentiation backend. |
| **protobuf** | 3.20.3 | Shared exchange format layer between SmartSim and ONNX runtimes. |
| **numpy** | 1.26.4 | Strict restriction $< 2.0.0$ enforced by the SmartSim orchestrator. |

All secondary python packages resolve automatically to their latest compatible releases using `pip-compile`.

---

## Installation Steps

### 1. Create Configuration Files

#### Base Conda Specification

Navigate to your working directory and generate the initial recipe layout:

```bash
cd $PYTHON_ROOT
nano -m base4SmartSim.yml

```

**Insert the following block into `base4SmartSim.yml**`

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

#### Heavy-Lifting Post-Installation Script

```bash
nano -m extra4SmartSim.sh

```

**Insert the following block into `extra4SmartSim.sh**`

```bash
#!/bin/bash
set -e

pip install --no-cache-dir pip-tools setuptools

cat <<'IN' > requirements.in
# --- Core Math & Data ---
numpy==1.26.4
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
scikit-learn==1.5.2
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

# --- Physics & CFD ---
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
eqx_io @ git+https://github.com/boss507104/CSC-Pilot.git#subdirectory=Utilities/eqx4smartredis

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

# Fetch and patch SmartSim client source directly to prevent explicit pointer mismatches
cd /tmp
rm -rf SmartRedis
git clone https://github.com/boss507104/SmartRedis.git
cd SmartRedis
sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp

OLD_CFLAGS="${CFLAGS-}"
OLD_CXXFLAGS="${CXXFLAGS-}"
OLD_CPPFLAGS="${CPPFLAGS-}"
OLD_LDFLAGS="${LDFLAGS-}"

unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
pip install --no-cache-dir .

export CFLAGS="$OLD_CFLAGS"
export CXXFLAGS="$OLD_CXXFLAGS"
export CPPFLAGS="$OLD_CPPFLAGS"
export LDFLAGS="$OLD_LDFLAGS"

cd /
rm -rf /tmp/SmartRedis

export USE_SYSTEMD=no
env CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart clobber

env CFLAGS="-Wno-incompatible-pointer-types" \
    CXXFLAGS="-Wno-incompatible-pointer-types" \
    USE_SYSTEMD=no \
    smart build --device cpu --skip-torch --skip-tensorflow

```

```bash
chmod +x extra4SmartSim.sh

```

### 2. Build Tykky Container

Request an interactive session on a test node to execute the container packaging tools:

```bash
srun --account=$CSC_PROJECT --partition=test --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --time=01:00:00 --pty bash

```

If network limits delay package downloads, switch to `--partition=medium` to lengthen runtime.

```bash
module load tykky
export TMPDIR=$TMP_BUILD_DIR

conda-containerize new \
    --prefix $ENV_PREFIX \
    --post-install $PYTHON_ROOT/extra4SmartSim.sh \
    $PYTHON_ROOT/base4SmartSim.yml

```

### 3. Build SmartRedis Native Library (Fortran/C++)

The client communication layer requires native compilation using standard cluster software modules:

```bash
module purge
module load gcc/13.1.0 cmake/3.28.6 git

cd "$BASE_SCRATCH"
rm -rf SmartRedis
git clone https://github.com/boss507104/SmartRedis.git
cd SmartRedis
sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp

rm -rf build

env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran

```

---

## Environment Activation / Loader

Save the following initialization template to your scratch workspace utility path at `$BASE_SCRATCH/Python4SmartSim.sh`.

```bash
cat <<EOF > $BASE_SCRATCH/Python4SmartSim.sh
#!/bin/bash
module load gcc/13.1.0

# Paths
export ENV_PREFIX="$ENV_PREFIX"
export SMARTREDIS_DIR="$SMARTREDIS_DIR"

# Tykky container bin path
export PATH="\$ENV_PREFIX/bin:\$PATH"

# SmartRedis native libs
export LD_LIBRARY_PATH="\$SMARTREDIS_DIR/install/lib:\$LD_LIBRARY_PATH"

# SmartSim & JAX tuning
export SMARTSIM_DB_FILE_PARSE_TRIALS=600
export JAX_PLATFORMS="gpu"
EOF

```

```bash
chmod +x $BASE_SCRATCH/Python4SmartSim.sh

```

Load your runtime stack during production job preparation steps: `source $BASE_SCRATCH/Python4SmartSim.sh`

---

## VS Code Kernel Registration

Run this block to expose your single-file container image to Jupyter interfaces or remote VS Code connections:

```bash
mkdir -p ~/.local/share/jupyter/kernels/$ENV_NICKNAME-smartsim

cat <<EOF > ~/.local/share/jupyter/kernels/$ENV_NICKNAME-smartsim/kernel.json
{
 "argv": [
  "$ENV_PREFIX/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python-3.11 ($ENV_NICKNAME Tykky-SmartSim)",
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
source $BASE_SCRATCH/Python4SmartSim.sh
jupyter kernelspec list

# Erase deprecated entries if required
jupyter kernelspec uninstall -f <kernel_name>

```

---

## Validation

Verify engine compatibility directly from your compute terminal:

```bash
source $BASE_SCRATCH/Python4SmartSim.sh

```

```bash
python -c "
import jax, equinox as eqx, jax2onnx
from smartsim._core.config import CONFIG
from importlib.metadata import version

print(f'JAX:        {jax.__version__}')
print(f'Equinox:    {eqx.__version__}')
print(f'jax2onnx:   {version(\"jax2onnx\")}')
print(f'DB Exec:    {CONFIG.database_exe}')
"

```

To validate advanced runtime interaction, run a complete JAX $\rightarrow$ ONNX $\rightarrow$ SmartRedis graph submission notebook.

**Standard Integrity Diagnostic Routine:**

```bash
smart validate --device cpu

```

Missing PyTorch or TensorFlow backends are normal and intentional.

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

* **Total State Purge** — Erase the deployment prefix target directory using `rm -rf $ENV_PREFIX` and repeat from step 3.
* **Sluggish Updates** — Consolidate scattered pip invocations inside a singular update file script layer.
* **Compiler Linkage Faults** — Verify host compiler environments explicitly using `module load gcc/13.1.0 cmake/3.28.6`.
* **Incompatible Pointer Exceptions** — Ensure strict compiler masking flags (`-Wno-incompatible-pointer-types`) are set during step 1.


## SmartSim Deployment Track (Co-Processing Ecosystem)

This module builds an isolated container architecture optimised for coupled multi-physics simulations, specifically interfacing parallel CFD solvers (e.g., OpenFOAM) with machine learning inference backends.

> [!NOTE]
> This installer sets up the core Python environment running **SmartSim 0.8.0** and **SmartRedis 0.6.1** under Python 3.11, adhering strictly to the required `numpy < 2.0.0` constraints.

### Production Blueprints & Advanced Tutorials

While this folder provides the setup files required to compile and containerise the environment via Tykky, the complete reference architecture, SLURM templates, and machine learning graph injection models are maintained in a dedicated validation repository.

For production-grade deployment strategies, please consult:
🔗 **[SmartSim4CSC Reference Repository](https://github.com/boss507104/SmartSim4CSC)**

#### Key Features Described in the Reference Repository:
* **HPC Topology Optimisation:** Detailed blueprints for executing the Orchestrator within local node boundaries (utilising node-local `nvme` scratch space) or scaling across distributed **Mahti** compute nodes via high-speed interconnects.
* **Physics-Informed Deep Learning:** Practical workflows demonstrating how to trace and serialise **JAX + Equinox** models into the immutable ONNX format, followed by real-time parallel graph submission and evaluation inside running OpenFOAM solver steps.
* **SLURM Batch Manifests:** Production-ready `sbatch` profiles tailored for the CSC environment, minimising data-transfer latency between numerical grids and the Redis in-memory database.
