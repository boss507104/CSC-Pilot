# ML Environment Configuration

Last updated: 15 July 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack optimised for machine learning, statistics, symbolic regression, and chemical kinetics analysis on CSC supercomputers (**Puhti / Mahti / Roihu**). The core stack is **JAX + Equinox + ONNX + PySR (JuliaCall)**.

Instead of Conda/pip environments directly on the parallel filesystem, we use **Tykky** to package the whole Python stack into a single-file container image, avoiding Lustre metadata slowdowns from thousands of small file imports.

Roihu needs **two separate Tykky environments**:

| Track | Applies to | CPU arch |
|---|---|---|
| `x64` | Roihu CPU nodes, Puhti, Mahti | x86_64 / amd64 |
| `arm64` | Roihu GPU nodes | aarch64 / ARM64 |

A container built for one architecture will not run on the other.

**Why Tykky:** near-instant imports, a single reproducible image, fast startup for short/high-volume jobs, and isolation from the host environment.

**Why uv:** fast resolution/installation of a large scientific stack. Direct specs live in `requirements.in` (unpinned on purpose, so builds pick up newer compatible versions); the exact installed set is recorded in `requirements.txt`.

```text
requirements.in   Human-maintained direct package specifications
requirements.txt  Exact installed package versions recorded after the build
```

No external Conda, Miniforge, Mamba, module-based Python, or venv is needed — everything happens inside the Tykky Python 3.12 build.

PySR uses `juliacall`. Because Tykky's `--post-install` script runs *inside* the activated build environment, Julia + its packages can be resolved and precompiled once, at build time, and shipped inside the container image.

---

## Build Flow

```text
Choose target architecture
  |
  +-- x64  (Roihu CPU / Puhti / Mahti)  --> Global Config (x64)  --> build envs/$ENV_NICKNAME-3.12-x64
  |
  +-- arm64 (Roihu GPU)                 --> Global Config (arm64) --> build envs/$ENV_NICKNAME-3.12-arm64

After the required track(s) are built:
  Create Python4ML.sh  -->  source Python4ML.sh  -->  loader picks x64/arm64 from `uname -m`
```

Skip the `arm64` track entirely if you never use Roihu GPU nodes.

---

## 1. Global Configuration

Run **one** of these blocks depending on the node you're on. Everything below (`$PYTHON_ROOT`, `$ENV_PREFIX`, etc.) depends on it.

> `Harry` and `Dumbledore` are fictional placeholders used throughout this doc. Replace them — and `project_xxxxxxx` — with your real values **everywhere they appear**, including inside `Python4ML.sh`.

### 1.1 x64 (Roihu CPU / Puhti / Mahti)

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export PROJECT_USER_DIR="Harry"             # Your directory under the CSC project
export ENV_NICKNAME="Dumbledore"            # Desired environment name
export ENV_ARCH="x64"
# --- USER CONFIGURATION END ---

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "ENV_ARCH=$ENV_ARCH"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
```

### 1.2 arm64 (Roihu GPU)

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"
export ENV_ARCH="arm64"
# --- USER CONFIGURATION END ---

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "ENV_ARCH=$ENV_ARCH"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
```

**Directory layout produced by this config:**

```text
/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities/     # $BASE_SCRATCH
├── .tykky_runtime_x64/
├── .tykky_runtime_arm64/
├── Python4ML.sh
└── Python/                                            # $PYTHON_ROOT
    ├── base4ML.yml
    ├── extra4ML.sh
    ├── update4ML.sh
    ├── requirements.in
    ├── requirements.txt
    └── envs/
        ├── $ENV_NICKNAME-3.12-x64/
        └── $ENV_NICKNAME-3.12-arm64/
```

---

## 2. Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| Python | 3.12 | Base interpreter |
| uv | latest at build time | Resolution & installation |
| NumPy | resolved | Numerical backend |
| JAX | CUDA 12 build, resolved | Autodiff / array programming |
| Equinox | resolved | NN / PyTree framework for JAX |
| ONNX / jax2onnx | resolved | Model export |
| PySR / julia (JuliaCall) | resolved | Symbolic regression |

---

## 3. Create the Configuration Files

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
```

### 3.1 `base4ML.yml`

```bash
cat <<'EOF' > "$PYTHON_ROOT/base4ML.yml"
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
EOF
```

### 3.2 `requirements.in`

```bash
cat <<'EOF' > "$PYTHON_ROOT/requirements.in"
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
shap
tensorboard
wandb
xgboost

# --- Symbolic Regression & Julia ---
pysr
julia

# --- Hyperparameter Optimisation ---
optuna
optuna-dashboard

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

# --- Data Version Control ---
dvc

# --- Custom Utilities ---
DataGraph @ git+https://github.com/boss507104/DataGraph.git#subdirectory=DataGraph

# --- Notebook Execution ---
ipykernel
ipywidgets
IPython
nbconvert
papermill

# --- Visualisation & UI ---
cmocean
colorcet
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
EOF
```

### 3.3 `extra4ML.sh` (post-install, runs *inside* the build)

This installs the stack via `uv`, then resolves/precompiles PySR's Julia dependency so it's baked into the image.

```bash
cat <<'EOF' > "$PYTHON_ROOT/extra4ML.sh"
#!/bin/bash
set -e

: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"

export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Always derive the Julia paths from the *actual* Python sys.prefix inside
# the container, not from $ENV_PREFIX — Tykky's wrapper path and the real
# in-container prefix are not the same thing.
PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"
mkdir -p "$JULIA_DEPOT_PATH" "$PYTHON_JULIAPKG_PROJECT"

python -m pip install --no-cache-dir uv

uv pip install --requirements "$PYTHON_ROOT/requirements.in"

python - <<'PY'
import juliapkg
juliapkg.resolve()
print(f"Julia executable: {juliapkg.executable()}")
print(f"Julia project:    {juliapkg.project()}")
PY

python - <<'PY'
import pysr
print(f"PySR version: {pysr.__version__}")
PY

python - <<'PY'
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
subprocess.run(
    [julia, f"--project={project}", "-e",
     "using Pkg; Pkg.instantiate(); Pkg.precompile(); "
     "using PythonCall; using SymbolicRegression"],
    check=True,
)
PY

python -m pip freeze > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

python - <<'PY' > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
print(f"Julia executable: {julia}")
print(f"Julia project: {project}\n")
subprocess.run(
    [julia, f"--project={project}", "-e",
     "using InteractiveUtils; versioninfo(); using Pkg; Pkg.status()"],
    check=True,
)
PY

rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/extra4ML.sh"
```

---

## 4. Request a Build Node

> **Tip — downloads:** `uv pip install` and Julia's package resolution both need outbound internet access. If your compute-node allocation has restricted/unstable external network access and the build fails or stalls specifically at the download stage, try running the download-heavy steps on the **login node** first (or the whole build, if your project's policy allows it), and reserve `srun`/`sinteractive` for genuinely compute-heavy steps. Check your own cluster's login-node usage policy before doing large builds there.

This block is reused for the initial build, updates, and rebuilds — just re-run it whenever you need a fresh allocation.

**x64:**

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

**arm64 (Roihu GPU):**

```bash
sinteractive \
    --account "$CSC_PROJECT" \
    --gpu \
    --cores 36 \
    --time 01:30:00
```

If the environment variables from Section 1 aren't inherited into the new shell, re-run the matching Global Configuration block after the allocation starts.

---

## 5. Build the Tykky Environment

```bash
module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

Build order: create the base env → run `extra4ML.sh` → install `uv` → install from `requirements.in` → record `requirements.txt` → resolve/precompile Julia+PySR → package everything into the Tykky image.

Check the result:

```bash
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
ls -lh "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
```

Build the other architecture separately, using its own Global Configuration (Section 1) and matching node (Section 4).

---

## 6. Loader — `Python4ML.sh`

```bash
cat <<'EOF' > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"

case "$(uname -m)" in
    x86_64)
        export ENV_ARCH="x64"
        export KERNEL_ARCH="x86_64"
        export JAX_PLATFORMS="cpu"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        export KERNEL_ARCH="aarch64"
        export JAX_PLATFORMS="cuda"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        return 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export PATH="$ENV_PREFIX/bin:$PATH"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "Environment not found for $ENV_ARCH: $ENV_PREFIX"
    return 1
fi

export PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"
export PYTHON_JULIAPKG_OFFLINE="yes"
export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK:-auto}"

export JUPYTER_KERNEL_NAME="$ENV_NICKNAME-ml-$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.12 ($ENV_NICKNAME ML $KERNEL_ARCH)"
export XDG_DATA_HOME="$HOME/.local/share/$KERNEL_ARCH"
export JUPYTER_KERNEL_DIR="$XDG_DATA_HOME/jupyter/kernels/$JUPYTER_KERNEL_NAME"

echo "ENV_ARCH=$ENV_ARCH"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"
EOF
chmod +x "$BASE_SCRATCH/Python4ML.sh"
```

Replace `project_xxxxxxx`, `Harry`, `Dumbledore` inside the file with your real values, then load it:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
uname -m; echo "$ENV_ARCH"; echo "$ENV_PREFIX"
which python; python --version
```

---

## 7. Register the Jupyter Kernel

Run once **per architecture**, after sourcing the loader on that architecture:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": ["$ENV_PREFIX/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": { "debugger": true }
}
EOF

cat "$JUPYTER_KERNEL_DIR/kernel.json"
jupyter kernelspec list
```

Remove an obsolete kernel: `jupyter kernelspec uninstall -f <kernel_name>`.

Then in VS Code: **Command Palette → Developer: Reload Window**.

---

## 8. Validate the Environment

```bash
source "$BASE_SCRATCH/Python4ML.sh"

python - <<'PY'
import sys, dvc, equinox, jax, nbconvert, numpy, optuna, papermill, pysr, shap

print(f"Python:      {sys.version.split()[0]}")
print(f"JAX:         {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
print(f"Equinox:     {equinox.__version__}")
print(f"NumPy:       {numpy.__version__}")
print(f"PySR:        {pysr.__version__}")
print(f"Papermill:   {papermill.__version__}")
print(f"nbconvert:   {nbconvert.__version__}")
print(f"DVC:         {dvc.__version__}")
print(f"SHAP:        {shap.__version__}")
PY
```

Scientific stack import check:

```bash
python -c "
import cantera, h5py, matplotlib, onnx, optax, pandas, scipy, sklearn, xarray
print('Core ML and scientific packages imported successfully.')
"
```

**JuliaCall** (should not download or install anything — that already happened at build time):

```bash
python - <<'PY'
import juliapkg
from juliacall import Main as jl
print(f"Julia executable: {juliapkg.executable()}")
print(f"Julia version:    {jl.VERSION}")
print(f"Julia threads:    {jl.Threads.nthreads()}")
PY
```

**PySR** end-to-end fit:

```bash
python - <<'PY'
import numpy as np
from pysr import PySRRegressor

X = np.linspace(-2.0, 2.0, 100)[:, None]
y = X[:, 0]**2 + 2.0 * X[:, 0] + 1.0

model = PySRRegressor(niterations=5, populations=2, population_size=20,
                       binary_operators=["+", "-", "*"], progress=False, verbosity=0)
model.fit(X, y)
print(model.sympy())
PY
```

**Papermill / nbconvert / Optuna Dashboard:**

```bash
papermill --version
jupyter nbconvert --version
optuna-dashboard --version
```

Inspect installed vs. recorded versions:

```bash
python -m pip freeze
head -n 40 "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
```

---

## 9. Dependency File Workflow

```text
requirements.in   Human-maintained direct package specifications
requirements.txt  Exact installed versions recorded after a completed build
```

**Add a package** — append it to `requirements.in`, then rebuild/update (Section 10 or 11):

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

**Remove a package** — delete its line, then do a full rebuild (Section 11) so unused transitive dependencies are also dropped.

**Reproduce an exact installed set** — temporarily point the install line in `extra4ML.sh` at `requirements-$ENV_ARCH.txt` instead of `requirements.in`, rebuild, then switch it back for normal development.

---

## 10. Updating the Environment

```bash
cat <<'EOF' > "$PYTHON_ROOT/update4ML.sh"
#!/bin/bash
set -e

: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"
: "${ENV_ARCH:?ENV_ARCH is not set}"

export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"

python -m pip install --no-cache-dir uv
uv pip install --requirements "$PYTHON_ROOT/requirements.in"

python - <<'PY'
import juliapkg, pysr
juliapkg.resolve()
print(f"PySR version: {pysr.__version__}")
print(f"Julia executable: {juliapkg.executable()}")
PY

python - <<'PY'
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
subprocess.run([julia, f"--project={project}", "-e",
                "using Pkg; Pkg.instantiate(); Pkg.precompile()"], check=True)
PY

python -m pip freeze > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/update4ML.sh"
```

Request the **same architecture** node (Section 4), load Tykky, and apply:

```bash
module purge
module load tykky
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize update \
    --post-install "$PYTHON_ROOT/update4ML.sh" \
    "$ENV_PREFIX"
```

Reload and check:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
python --version
python -m pip freeze
```

---

## 11. Rebuild / Clean Reinstall

Same for both a routine rebuild and a from-scratch install — the only difference is whether `$PYTHON_ROOT`'s files already exist.

```bash
# 1) Run the matching Global Configuration block (Section 1) first.

# 2) Confirm targets
echo "ENV_ARCH=$ENV_ARCH"; echo "ENV_PREFIX=$ENV_PREFIX"; echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"

# 3) Wipe old env + build cache
rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

# 4) Confirm config files exist (Section 3), then chmod the post-install script
ls -l "$PYTHON_ROOT/base4ML.yml" "$PYTHON_ROOT/requirements.in" "$PYTHON_ROOT/extra4ML.sh"
chmod +x "$PYTHON_ROOT/extra4ML.sh"
```

Request a node (Section 4), then build (Section 5):

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

If `Python4ML.sh` doesn't exist yet, create it (Section 6).

---

## 12. Troubleshooting

**Total reset:**
```bash
rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```
Then rebuild per Section 11 on the matching architecture.

**`requirements-$ENV_ARCH.txt` missing** — it's only generated after a completed install; run a full build (Section 5).

**Package install fails** — edit `requirements.in`, remove/replace the offending package, rebuild.

**Home quota exceeded during build** — make sure Section 1's Global Configuration ran first; caches redirect to `$BASE_SCRATCH/.tykky_runtime_x64` / `_arm64`, not `$HOME`.

**Architecture mismatch** (e.g. `the image's architecture (amd64) could not run on the host's (arm64)`) — build and use the matching-architecture environment; there is no cross-architecture container.

**JAX reports no GPU** — the loader sets `JAX_PLATFORMS=cpu` on x64 and `JAX_PLATFORMS=cuda` on arm64 automatically. Avoid manually setting `JAX_PLATFORMS=gpu`; it can trigger backend discovery paths that aren't valid on the target system. For GPU, you also need an actual GPU allocation.

**Import errors after an update / compiler linkage errors** — remove `$ENV_PREFIX` and `$TMP_BUILD_DIR`, and do a full rebuild (Section 11) rather than another `update`.

---

## Notes

* Python 3.12, built separately per architecture (x64, arm64) — never mix containers across architectures.
* `Harry` / `Dumbledore` / `project_xxxxxxx` are fictional placeholders — replace consistently in **every** script, including `Python4ML.sh`.
* `PROJECT_USER_DIR` is not necessarily your CSC login username; it's just the directory under the project's scratch space.
* `requirements.in` = direct specs (unpinned by design); `requirements.txt` / `requirements-$ENV_ARCH.txt` = exact installed versions. Use the latter when an exact set must be reproduced.
* `jax[cuda12]` installs the CUDA 12–compatible JAX build.
* PySR's Julia dependency is resolved and precompiled **at build time** inside `extra4ML.sh` / `update4ML.sh`; `PYTHON_JULIAPKG_OFFLINE=yes` at runtime prevents any accidental re-download.
* Use compute nodes for the actual container build; consider the login node only for download-heavy steps if a compute allocation's network access is the bottleneck — check your project's policy first.
