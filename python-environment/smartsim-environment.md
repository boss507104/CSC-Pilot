# SmartSim Environment Configuration

Last updated: 24 July 2026

> [!TIP]
> ## One-Command Installation
>
> Copy `smartsim-python.sh` to `/scratch/project_xxxxxxx/PROJECT_USER_DIR/`, then run:
>
> ```bash
> chmod +x smartsim-python.sh
> ./smartsim-python.sh
> ```

---

## Overview & Motivation

This folder deploys a **unified ML + SmartSim/SmartRedis stack on CSC's Roihu supercomputer only** — **JAX + Equinox + TensorFlow + PyTorch + ONNX + SmartSim + SmartRedis**, with **PySR (JuliaCall)** available as an **optional** add-on, all in one environment. The same SmartSim-CSC checkout also contains an independently maintained OpenFOAM integration for OpenFOAM.com v2412 on Roihu x86_64 CPU nodes. Puhti and Mahti are no longer targets of this guide.

The target architecture is **auto-detected** from the node you run the installer on:

| Host (`uname -m`) | `ENV_ARCH` | SmartSim-CSC profile |
| --- | --- | --- |
| `x86_64` (Roihu CPU) | `x64` | `linux-x64-cpu` |
| `aarch64` (Roihu GPU) | `arm64` | `linux-arm64-gpu` |

The OpenFOAM integration is currently validated only on `x86_64` Roihu CPU nodes with OpenFOAM.com v2412, GCC 15.2.0, and OpenMPI 5.0.10.

SmartSim and SmartRedis no longer come from two separate CSC forks. Both are now sourced from a single **[SmartSim-CSC](https://github.com/PentagonToy/SmartSim-CSC)** monorepo, which also bundles the RedisAI source and a `stack.toml` that defines build profiles:

```text
Component      Version
SmartSim-CSC   1.0.0
SmartSim       1.0.0+csc
SmartRedis     1.0.0+csc
RedisAI        1.2.7
```

The officially validated profile is `linux-x64-cpu`:

```text
[profiles.linux-x64-cpu]
device = "cpu"
backends = ["onnxruntime", "jax"]
```

`linux-arm64-cpu` is validated the same way. The `linux-arm64-gpu` profile has also been validated on a Roihu GH200 compute node with CUDA 12.9. The environment may be installed on the ARM64 GPU login node, but GPU runtime validation and GPU workloads require an allocated GPU compute node.

Only **ONNX Runtime** and **JAX** are built as RedisAI backends by this stack — there is no RedisAI TensorFlow or LibTorch backend here. TensorFlow and PyTorch are still installed as regular Python packages (usable directly in Python), just not exposed through `set_model`/`run_model`.

**PySR/Julia is optional**, unrelated to the SmartSim-CSC stack itself. A single `INSTALL_PYSR` toggle, asked once per architecture and persisted to `install-options-$ENV_ARCH.sh`, controls whether:

* `pysr`/`julia` are added to `requirements.in`,
* the Julia resolve/precompile step runs inside the Tykky build,
* the one-time writable Julia runtime copy (Section 5.1) is prepared,
* the loader (Section 7) configures `PYTHON_JULIAPKG_PROJECT`/`JULIA_DEPOT_PATH`.

We use **Tykky** to package the whole Python stack into a single-file container image, avoiding Lustre metadata slowdowns from thousands of small file imports. A Tykky container built for one architecture will not run on the other. The **native SmartRedis library** (Section 6, needed for OpenFOAM/C++/Fortran linkage) is now built from `components/smartredis` inside the SmartSim-CSC checkout rather than from a separate git clone, and is unaffected by the PySR toggle. On Roihu x86_64 CPU nodes, this native library can then be used to build the bundled OpenFOAM v2412 integration (Section 6.1).

**Why uv:** fast resolution/installation, plus `uv pip check` to validate the final dependency graph. `--link-mode=copy` is used throughout since the uv cache and the Tykky build environment live on different filesystems.

```text
Python           3.12
SmartSim-CSC     1.0.0 (PentagonToy/SmartSim-CSC, pinned ref)
SmartSim         1.0.0+csc
SmartRedis       1.0.0+csc
RedisAI          1.2.7, backends: onnxruntime + jax (per stack.toml)
OpenFOAM         v2412 integration validated on Roihu x86_64 CPU
JAX              installed by SmartSim-CSC's install.sh, not requirements.in
TensorFlow       2.18.1  (Python package only, not a RedisAI backend)
PyTorch          2.7.1   (Python package only, not a RedisAI backend)
ONNX             resolved (+ ONNX Runtime, tf2onnx, skl2onnx)
PySR / Julia     OPTIONAL (INSTALL_PYSR=yes/no)
NumPy            >= 2.0
```

```text
requirements.in                  Human-maintained direct package specs; no smartsim/smartredis/jax; pysr/julia only if INSTALL_PYSR=yes
requirements-$ENV_ARCH.txt       Installed-state snapshot recorded after a successful build
julia-environment-$ENV_ARCH.txt  Julia toolchain/package status; placeholder if INSTALL_PYSR=no
install-options-$ENV_ARCH.sh     Persists INSTALL_PYSR across sessions
runtime-$ENV_ARCH.sh             GCC/CUDA modules + PySR-enabled flag, read by the loader every source
```

Build order: install `uv` → install `requirements.in` (TensorFlow, PyTorch, ONNX, `pysr`/`julia` if enabled) → resolve/precompile Julia+PySR (if enabled) → checkout pinned `SmartSim-CSC` ref → run its `scripts/install.sh` for the detected profile (installs SmartSim + SmartRedis, builds Redis + RedisAI + selected backends, runs `smart validate`) → `uv pip check` → prepare the writable Julia runtime once, if enabled → build the native SmartRedis library from `components/smartredis` and record its GCC/CUDA modules + PySR-enabled flag → optionally build the OpenFOAM v2412 integration on Roihu x86_64 CPU.

---

## Build Flow

```text
Set identity once (Section 0)
  |
  v
Run the installer on the target Roihu node — architecture is auto-detected
  |
  +-- Roihu CPU node (x86_64) -> ENV_ARCH=x64, PROFILE=linux-x64-cpu
  |     --> choose INSTALL_PYSR=yes/no, persist to install-options-x64.sh
  |     --> install requirements.in (pysr/julia only if enabled)
  |     --> [if enabled] resolve/precompile Julia + PySR
  |     --> checkout SmartSim-CSC @ pinned ref
  |     --> run scripts/install.sh PROFILE=linux-x64-cpu (SmartSim+SmartRedis+RedisAI: onnxruntime+jax)
  |     --> uv pip check
  |     --> [if enabled] prepare writable Julia runtime ONCE
  |     --> build native SmartRedis-$ENV_ARCH library from components/smartredis; record GCC/CUDA modules + PySR flag
  |
  +-- Roihu GPU node (aarch64) -> ENV_ARCH=arm64, PROFILE=linux-arm64-gpu
        same steps as above, using components checked out from SMARTSIM_CSC_REF

After the required track(s) are built:
  Create Python4SmartSim.sh --> source Python4SmartSim.sh
  --> loader picks x64/arm64 + profile from `uname -m`, only sets env vars and
      PATH/LD_LIBRARY_PATH/CMAKE_PREFIX_PATH (idempotent)
  --> loader configures Julia env vars only if this architecture was built with INSTALL_PYSR=yes
  --> Jupyter kernels run through a launcher wrapper that sources the same loader
```

Skip the `arm64` track if you do not need Roihu GPU workloads. Each architecture independently chooses `INSTALL_PYSR=yes` or `no`.

---

## 0. One-Time Identity Configuration

Every script needs three values: your CSC project ID, your directory under that project, and the environment nickname. Set them **once**.

> `Harry`, `Dumbledore`, and `project_xxxxxxx` are fictional placeholders. Fill in real values **only here**.

```bash
mkdir -p "$HOME/.config/csc-hpc"

cat <<'EOF' > "$HOME/.config/csc-hpc/identity.sh"
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"
EOF

chmod 600 "$HOME/.config/csc-hpc/identity.sh"
```

```bash
nano "$HOME/.config/csc-hpc/identity.sh"

source "$HOME/.config/csc-hpc/identity.sh"
echo "CSC_PROJECT=$CSC_PROJECT"
echo "PROJECT_USER_DIR=$PROJECT_USER_DIR"
echo "ENV_NICKNAME=$ENV_NICKNAME"
```

`ENV_ARCH`, `SMARTSIM_CSC_PROFILE`, and `INSTALL_PYSR` are **not** part of this file — the first two are auto-detected from `uname -m` everywhere, and `INSTALL_PYSR` is chosen per build in Section 1.

---

## 1. Global Configuration

Run on the Roihu node matching the architecture you want to build. Architecture and profile are detected automatically; you only choose the PySR toggle.

```bash
source "$HOME/.config/csc-hpc/identity.sh"

case "$(uname -m)" in
    x86_64)  export ENV_ARCH="x64";  export SMARTSIM_CSC_PROFILE="linux-x64-cpu" ;;
    aarch64) export ENV_ARCH="arm64"; export SMARTSIM_CSC_PROFILE="linux-arm64-gpu" ;;
    *) echo "Unsupported Roihu architecture: $(uname -m)"; return 1 ;;
esac

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTSIM_CSC_REPO="https://github.com/PentagonToy/SmartSim-CSC.git"
export SMARTSIM_CSC_REF="3d4749d"
export SMARTSIM_CSC_DIR="$PYTHON_ROOT/src/SmartSim-CSC"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

read -r -p "Install PySR (symbolic regression) with its Julia toolchain for $ENV_ARCH? [Y/n]: " _A
case "$_A" in n|N|no|NO) export INSTALL_PYSR="no" ;; *) export INSTALL_PYSR="yes" ;; esac
unset _A

cat <<EOF > "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
export INSTALL_PYSR="$INSTALL_PYSR"
EOF
chmod 600 "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

echo "ENV_ARCH=$ENV_ARCH  PROFILE=$SMARTSIM_CSC_PROFILE  INSTALL_PYSR=$INSTALL_PYSR"
echo "SMARTSIM_CSC_REF=$SMARTSIM_CSC_REF"
```

> `SMARTSIM_CSC_REF` currently defaults to the validated commit `3d4749d`, which includes the OpenFOAM v2412 full-build integration. After these changes are merged and released, replace the commit with the corresponding release tag.

**Directory layout:**

```text
/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities/          # $BASE_SCRATCH
├── .tykky_runtime_smartsim_x64/
├── .tykky_runtime_smartsim_arm64/
├── .julia_env_runtime_x64/       # ONCE after Tykky build (5.1) — only if INSTALL_PYSR=yes
├── .julia_env_runtime_arm64/
├── .julia_depot_runtime_x64/
├── .julia_depot_runtime_arm64/
├── Python4SmartSim.sh
├── SmartRedis-x64/                                  # native install, $SMARTREDIS_DIR
├── SmartRedis-arm64/
├── OpenFOAM/
│   └── OpenFOAM-v2412/                  # x86_64 OpenFOAM user build/install area
└── Python/                                           # $PYTHON_BASE
    └── PythonSmartSim/                               # $PYTHON_ROOT
        ├── base4SmartSim.yml
        ├── extra4SmartSim.sh
        ├── update4SmartSim.sh
        ├── requirements.in
        ├── requirements-x64.txt
        ├── requirements-arm64.txt
        ├── julia-environment-x64.txt
        ├── julia-environment-arm64.txt
        ├── install-options-x64.sh
        ├── install-options-arm64.sh
        ├── runtime-x64.sh
        ├── runtime-arm64.sh
        ├── jupyter-kernel-x64.sh
        ├── jupyter-kernel-arm64.sh
        ├── src/SmartSim-CSC/                # $SMARTSIM_CSC_DIR — pinned monorepo checkout
        └── envs/
```

---

## 2. Dependency Overview

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.12 | Base interpreter |
| uv | latest at build | Resolution, installation, `uv pip check` |
| SmartSim-CSC | 1.0.0, pinned ref | Monorepo providing SmartSim, SmartRedis, RedisAI sources + `scripts/install.sh` |
| SmartSim | 1.0.0+csc | Orchestration; Redis, RedisAI, JAX worker lifecycle |
| SmartRedis | 1.0.0+csc | Python client + native C++/Fortran library; direct JAX/Equinox registration |
| RedisAI | 1.2.7 | Backends selected per profile: `onnxruntime`, `jax` |
| OpenFOAM | v2412 | Optional x86_64 integration built from `components/openfoam-smartsim` |
| JAX / Equinox / distrax / distreqx | installed by `install.sh` | Autodiff / training / inference |
| TensorFlow | 2.18.1 | Python framework only — not a RedisAI backend here |
| PyTorch | 2.7.1 | Python framework only — not a RedisAI backend here |
| ONNX / ONNX Runtime | resolved | Model interchange + Python-side runtime |
| PySR / julia (JuliaCall) | **optional** | Symbolic regression; only if `INSTALL_PYSR=yes` |
| NumPy | `>= 2.0` | — |

---

## 3. Create the Configuration Files

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh" 2>/dev/null || true
: "${INSTALL_PYSR:=yes}"
```

### 3.1 `base4SmartSim.yml`

```bash
cat <<'EOF' > "$PYTHON_ROOT/base4SmartSim.yml"
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

`smartsim`, `smartredis`, and `jax` itself are **not** listed here — SmartSim-CSC's `install.sh` owns all three. `pysr`/`julia` are appended only if `INSTALL_PYSR=yes`.

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

# --- JAX Ecosystem (jax itself installed by SmartSim-CSC install.sh) ---
diffrax
distrax
distreqx
equinox
jaxtyping
jax2onnx
jaxopt
einops
lineax
optax
optimistix
sympy2jax

# --- TensorFlow / PyTorch / ONNX ---
tensorflow==2.18.1
torch==2.7.1
onnx
onnxruntime
tf2onnx
skl2onnx

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
treeple
wandb
xgboost

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
DataGraph @ git+https://github.com/PentagonToy/DataGraph.git#subdirectory=DataGraph

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
pydantic
PyYAML

# --- Profiling & Logging ---
loguru
pyinstrument

# --- HPC / Slurm ---
submitit

# --- System & Development ---
kneed
natsort
pytest
tabulate
typing-extensions
EOF

if [ "$INSTALL_PYSR" = "yes" ]; then
    cat <<'EOF' >> "$PYTHON_ROOT/requirements.in"

# --- Symbolic Regression & Julia ---
pysr
julia
EOF
fi
```

### 3.3 `extra4SmartSim.sh` (post-install, runs *inside* the build)

Installs the ML package set, conditionally resolves/precompiles PySR's Julia dependency, checks out the pinned `SmartSim-CSC` ref, and hands off to its own `scripts/install.sh` for the detected profile.

```bash
cat <<'EOF' > "$PYTHON_ROOT/extra4SmartSim.sh"
#!/bin/bash
set -e

: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"
: "${ENV_ARCH:?ENV_ARCH is not set}"
: "${SMARTSIM_CSC_REPO:?SMARTSIM_CSC_REPO is not set}"
: "${SMARTSIM_CSC_REF:?SMARTSIM_CSC_REF is not set}"
: "${SMARTSIM_CSC_DIR:?SMARTSIM_CSC_DIR is not set}"
: "${SMARTSIM_CSC_PROFILE:?SMARTSIM_CSC_PROFILE is not set}"
: "${INSTALL_PYSR:=yes}"

export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

python -m pip install --no-cache-dir uv

uv pip install --link-mode=copy --requirements "$PYTHON_ROOT/requirements.in"

if [ "$INSTALL_PYSR" = "yes" ]; then
    PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
    export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
    export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"
    mkdir -p "$JULIA_DEPOT_PATH" "$PYTHON_JULIAPKG_PROJECT"

    python - <<'PY'
import juliapkg
juliapkg.resolve()
print(f"Julia executable: {juliapkg.executable()}")
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
fi

# --- Install the unified SmartSim-CSC stack ---
mkdir -p "$(dirname "$SMARTSIM_CSC_DIR")"

if [ -d "$SMARTSIM_CSC_DIR/.git" ]; then
    git -C "$SMARTSIM_CSC_DIR" fetch --tags origin
else
    rm -rf "$SMARTSIM_CSC_DIR"
    git clone "$SMARTSIM_CSC_REPO" "$SMARTSIM_CSC_DIR"
fi

git -C "$SMARTSIM_CSC_DIR" checkout --detach --force "$SMARTSIM_CSC_REF"
git -C "$SMARTSIM_CSC_DIR" clean -ffdx

export USE_SYSTEMD=no
export PYTHONNOUSERSITE=1

PYTHON="$(command -v python)" \
SMART="$(dirname "$(command -v python)")/smart" \
PROFILE="$SMARTSIM_CSC_PROFILE" \
PYTHONNOUSERSITE=1 \
    "$SMARTSIM_CSC_DIR/scripts/install.sh"

# Restore the user-managed ML environment after SmartSim-CSC installation.
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

python -m pip list --format=freeze \
    | grep -v '^smartredis==' | grep -v '^smartsim==' | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

if [ "$INSTALL_PYSR" = "yes" ]; then
    python - <<'PY' > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
print(f"Julia executable: {julia}\nJulia project: {project}\n")
subprocess.run([julia, f"--project={project}", "-e",
    "using InteractiveUtils; versioninfo(); using Pkg; Pkg.status()"], check=True)
PY
else
    echo "PySR/Julia was not installed (INSTALL_PYSR=no)." \
        > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
fi

rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"
```

The installer expects `PYTHON`, `SMART`, and `PROFILE` to be provided as environment variables; it reads component paths and the backend list from `stack.toml`, builds Redis + RedisAI + the selected backends, checks Python dependencies, verifies build artifacts, and runs `smart validate` internally.

---

## 4. Request a Build Node

Both the SmartSim-CSC checkout/install and Julia's package resolution (if `INSTALL_PYSR=yes`) need outbound network access. Try download-heavy steps on the login node first if a compute allocation restricts networking.

**Roihu CPU:**

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --time=01:30:00 --pty bash
```

**Roihu GPU:**

```bash
sinteractive --account "$CSC_PROJECT" --gpu --cores 36 --time 01:30:00
```

If Section 1's variables aren't inherited, re-run the Global Configuration block on this node.

---

## 5. Build the Tykky Environment

```bash
module purge
module load tykky
module load "$GCC_MODULE"

if [ -n "$CUDA_MODULE" ]; then
    module load "$CUDA_MODULE"
fi

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
export INSTALL_PYSR
export SMARTSIM_CSC_REPO SMARTSIM_CSC_REF SMARTSIM_CSC_DIR SMARTSIM_CSC_PROFILE

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

```bash
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
"$ENV_PREFIX/bin/python" -m pip list --format=freeze \
    | grep -E '^(jax|numpy|tensorflow|torch|onnx|onnxruntime|smartsim|smartredis)=='
```

### 5.1 Prepare the writable PySR / Julia runtime (once, only if `INSTALL_PYSR=yes`)

```bash
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

if [ "$INSTALL_PYSR" = "yes" ]; then
    PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"
    JULIA_ENV_SOURCE="$PYTHON_PREFIX/julia_env"
    JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
    JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

    [ -d "$JULIA_ENV_SOURCE" ] || { echo "ERROR: $JULIA_ENV_SOURCE not found"; exit 1; }

    rm -rf "$JULIA_ENV_RUNTIME"
    cp -a "$JULIA_ENV_SOURCE" "$JULIA_ENV_RUNTIME"
    mkdir -p "$JULIA_DEPOT_RUNTIME"
else
    rm -rf "$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH" "$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
fi
```

Build the other architecture separately (Section 1 + Section 4); its `INSTALL_PYSR` choice is independent.

---

## 6. Build the SmartRedis Native Library

Needed on **both** architectures for OpenFOAM/C++/Fortran linkage. The source now comes from the already-checked-out SmartSim-CSC monorepo, not a separate clone.

```bash
if [ "$ENV_ARCH" = "x64" ]; then
    GCC_MODULE="gcc/13.4.0"
    CMAKE_MODULE="cmake/3.26.5"
    CUDA_MODULE=""
else
    GCC_MODULE="gcc/14.3.0"
    CMAKE_MODULE="cmake/3.31.11"
    CUDA_MODULE="cuda/12.9.1"
fi

module purge
module load "$GCC_MODULE"
module load "$CMAKE_MODULE"

if [ -n "$CUDA_MODULE" ]; then
    module load "$CUDA_MODULE"
fi
```

Record the runtime modules and PySR flag:

```bash
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

cat <<EOF > "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
export SMARTSIM_GCC_MODULE="$GCC_MODULE"
export SMARTSIM_CUDA_MODULE="$CUDA_MODULE"
export SMARTSIM_PYSR_ENABLED="$INSTALL_PYSR"
EOF
chmod 600 "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
```

Copy the component out of the checkout and build:

```bash
[ -d "$SMARTSIM_CSC_DIR/components/smartredis" ] || { echo "SmartRedis source not found in checkout"; exit 1; }

rm -rf "$SMARTREDIS_DIR"
mkdir -p "$SMARTREDIS_DIR"
cp -a "$SMARTSIM_CSC_DIR/components/smartredis/." "$SMARTREDIS_DIR/"

cd "$SMARTREDIS_DIR"
rm -rf build install

env \
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran
```

Verify:

```bash
find "$SMARTREDIS_DIR/install" -maxdepth 3 -type f | sort

# If lib64 doesn't exist, use lib.
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library installed successfully."
ldd "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so"
```


### 6.1 Build the OpenFOAM v2412 Integration

This step is currently supported and validated on Roihu x86_64 CPU nodes only. It uses the native SmartRedis library built in Section 6.

The OpenFOAM module stack uses GCC 15.2.0, while the SmartRedis native library was built with GCC 13.4.0. The tested order is therefore:

```bash
module --force purge
source "$BASE_SCRATCH/Python4SmartSim.sh"

module load gcc/15.2.0
module load openmpi/5.0.10
module load openfoam/2412
```

Do not source `Python4SmartSim.sh` again after loading OpenFOAM, because the loader would restore the SmartRedis build compiler module.

Set the OpenFOAM user installation directory and build the bundled integration:

```bash
export FOAM_USER_DIR="$BASE_SCRATCH/OpenFOAM/OpenFOAM-v2412"

cd "$SMARTSIM_CSC_DIR"
./scripts/openfoam/build-openfoam-v2412.sh
```

The build installs the following libraries and executables:

```text
$FOAM_USER_LIBBIN/libsmartRedisClient.so
$FOAM_USER_LIBBIN/libsmartredisFunctionObjects.so
$FOAM_USER_LIBBIN/libsmartSimMotionSolvers.so

$FOAM_USER_APPBIN/foamSmartSimSvd
$FOAM_USER_APPBIN/foamSmartSimSvdDBAPI
$FOAM_USER_APPBIN/svdToFoam
```

Verify that the executables start and that no shared libraries are missing:

```bash
export PATH="$FOAM_USER_APPBIN:$PATH"

foamSmartSimSvd -help | tail -4
foamSmartSimSvdDBAPI -help | tail -4
svdToFoam -help | tail -4

ldd "$FOAM_USER_APPBIN/foamSmartSimSvdDBAPI"     | grep -E 'not found|smartredis|smartRedisClient'
```

Validated runtime results on Roihu x86_64 include:

- `foamSmartSimSvd`: one pressure tensor with shape `(400, 50)`
- `foamSmartSimSvdDBAPI`: 50 timestep datasets, each containing a pressure tensor with shape `(400, 1)`
- metadata dataset value `NTimes=50`
- SmartRedis C++ → SmartSim database → Python SmartRedis round trip

The function-object and mesh-motion libraries build and link successfully; their dedicated runtime tests remain pending.


---

## 7. Loader — `Python4SmartSim.sh`

Must be *sourced*, not executed; safe to re-source. It auto-detects architecture/profile, validates the Tykky environment and SmartRedis install, loads the recorded GCC and CUDA modules if needed, prepends `PATH`/`LD_LIBRARY_PATH`/`CMAKE_PREFIX_PATH` idempotently, and configures Julia/PySR vars only if this architecture was built with `INSTALL_PYSR=yes`.

```bash
cat <<'EOF' > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This file must be sourced, not executed:"
    echo "    source ${BASH_SOURCE[0]}"
    exit 1
fi

IDENTITY_FILE="$HOME/.config/csc-hpc/identity.sh"
[ -f "$IDENTITY_FILE" ] || { echo "Identity file not found: $IDENTITY_FILE"; return 1; }
source "$IDENTITY_FILE"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export SMARTSIM_CSC_DIR="$PYTHON_ROOT/src/SmartSim-CSC"

case "$(uname -m)" in
    x86_64)
        export ENV_ARCH="x64" KERNEL_ARCH="x86_64" JAX_PLATFORMS="cpu"
        export SMARTSIM_CSC_PROFILE="linux-x64-cpu"
        ;;
    aarch64)
        export ENV_ARCH="arm64" KERNEL_ARCH="aarch64" JAX_PLATFORMS="cuda"
        export SMARTSIM_CSC_PROFILE="linux-arm64-gpu"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"; return 1 ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"

[ -x "$ENV_PREFIX/bin/python" ] || { echo "Python environment not found: $ENV_PREFIX"; return 1; }
[ -d "$SMARTREDIS_DIR/install" ] || { echo "SmartRedis installation not found: $SMARTREDIS_DIR/install"; return 1; }

RUNTIME_CONFIG="$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
[ -f "$RUNTIME_CONFIG" ] && source "$RUNTIME_CONFIG"
: "${SMARTSIM_PYSR_ENABLED:=yes}"

if [ -n "${SMARTSIM_GCC_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module is-loaded "$SMARTSIM_GCC_MODULE" 2>/dev/null || module load "$SMARTSIM_GCC_MODULE"
fi

if [ -n "${SMARTSIM_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module is-loaded "$SMARTSIM_CUDA_MODULE" 2>/dev/null || module load "$SMARTSIM_CUDA_MODULE"
fi

path_prepend() {
    local v="$1" d="$2" c="${!1-}"
    case ":$c:" in *":$d:"*) ;; *) printf -v "$v" '%s' "$d${c:+:$c}"; export "$v" ;; esac
}

if [ -d "$SMARTREDIS_DIR/install/lib64" ]; then
    export SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib64"
elif [ -d "$SMARTREDIS_DIR/install/lib" ]; then
    export SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib"
else
    echo "SmartRedis library directory not found."; return 1
fi

path_prepend PATH "$ENV_PREFIX/bin"
path_prepend LD_LIBRARY_PATH "$SMARTREDIS_LIB_DIR"
path_prepend CMAKE_PREFIX_PATH "$SMARTREDIS_DIR/install"

export SMARTSIM_DB_FILE_PARSE_TRIALS=600
export PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"

if [ "$SMARTSIM_PYSR_ENABLED" = "yes" ]; then
    export JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
    export JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
    [ -d "$JULIA_ENV_RUNTIME" ] || { echo "Writable Julia environment not found: $JULIA_ENV_RUNTIME"; return 1; }
    mkdir -p "$JULIA_DEPOT_RUNTIME"
    export PYTHON_JULIAPKG_PROJECT="$JULIA_ENV_RUNTIME"
    export JULIA_DEPOT_PATH="$JULIA_DEPOT_RUNTIME:$PYTHON_PREFIX/julia_depot"
    export PYTHON_JULIAPKG_OFFLINE="yes"
    export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK:-auto}"
    unset PYTHON_JULIACALL_EXE PYTHON_JULIACALL_PROJECT
else
    unset JULIA_ENV_RUNTIME JULIA_DEPOT_RUNTIME
    unset PYTHON_JULIAPKG_PROJECT JULIA_DEPOT_PATH PYTHON_JULIAPKG_OFFLINE
    unset PYTHON_JULIACALL_THREADS PYTHON_JULIACALL_EXE PYTHON_JULIACALL_PROJECT
fi

export JUPYTER_KERNEL_NAME="$ENV_NICKNAME-smartsim-$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.12 ($ENV_NICKNAME SmartSim $KERNEL_ARCH)"
export JUPYTER_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$JUPYTER_KERNEL_NAME"

if [ "${SMARTSIM_ENV_QUIET:-0}" != "1" ]; then
    echo "SmartSim Python environment loaded"
    echo "ENV_ARCH=$ENV_ARCH  PROFILE=$SMARTSIM_CSC_PROFILE  JAX_PLATFORMS=$JAX_PLATFORMS"
    echo "SMARTSIM_PYSR_ENABLED=$SMARTSIM_PYSR_ENABLED"
fi

unset -f path_prepend
EOF
chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"
```

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
python --version
```

`SMARTSIM_ENV_QUIET=1` suppresses the status banner (used by the Jupyter kernel launcher).

---

## 8. Register the Jupyter Kernel

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"

JUPYTER_KERNEL_LAUNCHER="$PYTHON_ROOT/jupyter-kernel-$ENV_ARCH.sh"

cat <<EOF > "$JUPYTER_KERNEL_LAUNCHER"
#!/bin/bash
export SMARTSIM_ENV_QUIET=1
source "$BASE_SCRATCH/Python4SmartSim.sh" || exit 1
unset SMARTSIM_ENV_QUIET
exec "$ENV_PREFIX/bin/python" -m ipykernel_launcher "\$@"
EOF
chmod +x "$JUPYTER_KERNEL_LAUNCHER"

mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": ["$JUPYTER_KERNEL_LAUNCHER", "-f", "{connection_file}"],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": { "debugger": true }
}
EOF

jupyter kernelspec list
```

Remove an obsolete kernel: `jupyter kernelspec uninstall -f <kernel_name>`.

---

## 9. Validate the Environment

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"

python -c "
import sys, numpy, jax, equinox, tensorflow, torch, onnx, onnxruntime, smartsim, smartredis
print(f'Python:     {sys.version.split()[0]}')
print(f'NumPy:      {numpy.__version__}')
print(f'JAX:        {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}')
print(f'TensorFlow: {tensorflow.__version__}')
print(f'PyTorch:    {torch.__version__}')
print(f'SmartSim:   {smartsim.__version__}')
print(f'SmartRedis: {smartredis.__version__}')
"

uv pip check
smart validate --device cpu
python "$SMARTSIM_CSC_DIR/scripts/check_versions.py"
python "$SMARTSIM_CSC_DIR/scripts/stack_config.py" --profile "$SMARTSIM_CSC_PROFILE"
```

`smart validate` should report the ONNX Runtime and JAX backends only (per `stack.toml`) — there is no TensorFlow or LibTorch RedisAI backend in this stack.

**PySR / JuliaCall** (only if `SMARTSIM_PYSR_ENABLED=yes`):

```bash
if [ "${SMARTSIM_PYSR_ENABLED:-no}" = "yes" ]; then
    python - <<'PY'
import juliapkg, pysr
from juliacall import Main as jl
print(f"PySR version:     {pysr.__version__}")
print(f"Julia version:    {jl.VERSION}")
PY
fi
```

Native library check:

```bash
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library is available."
```

---

## 10. Dependency File Workflow

* Edit `requirements.in`, then rebuild/update (Section 11 or 12). Removing a package needs a full rebuild.
* `jax` itself is intentionally **not** in `requirements.in` — it's installed by SmartSim-CSC's `install.sh` for the selected profile.
* `tensorflow`/`torch` stay pinned; ONNX/ONNX Runtime/NumPy resolve at build time. They are plain Python packages here, not RedisAI backends, so their versions don't need to match RedisAI's own build.
* Flipping `INSTALL_PYSR` requires regenerating `requirements.in` (Section 3.2) and a **full rebuild** (Section 12); `conda-containerize update` alone is not reliable for adding/removing the Julia toolchain.

---

## 11. Updating the Environment

Regular package updates **must not** touch the SmartSim-CSC stack — `smartsim`/`smartredis` are owned by the pinned checkout, not by `uv`/pip.

```bash
cat <<'EOF' > "$PYTHON_ROOT/update4SmartSim.sh"
#!/bin/bash
set -e

: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"
: "${ENV_ARCH:?ENV_ARCH is not set}"
: "${INSTALL_PYSR:=yes}"

export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

python -m pip install --no-cache-dir uv
uv pip install --link-mode=copy --requirements "$PYTHON_ROOT/requirements.in"

UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"
if [ -s "$UPDATE_REQUEST" ]; then
    mapfile -t PKGS < "$UPDATE_REQUEST"
    uv pip install --link-mode=copy --upgrade "${PKGS[@]}"
fi

if [ "$INSTALL_PYSR" = "yes" ]; then
    PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
    export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
    export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"
    python - <<'PY'
import juliapkg, pysr
juliapkg.resolve()
print(f"PySR version: {pysr.__version__}")
PY
    python - <<'PY'
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
subprocess.run([julia, f"--project={project}", "-e",
    "using Pkg; Pkg.instantiate(); Pkg.precompile()"], check=True)
PY
fi

uv pip check

python -m pip list --format=freeze \
    | grep -v '^smartredis==' | grep -v '^smartsim==' | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

rm -f "$UPDATE_REQUEST"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/update4SmartSim.sh"
```

`smartsim-update` updates ordinary Python packages only. It refuses `smartsim`, `smartredis`, `jax`, `jaxlib`, `jax-cuda12-plugin`, and `jax-cuda12-pjrt` because those packages are managed by SmartSim-CSC. It also refuses `pysr`/`julia` when `INSTALL_PYSR=no` for that architecture:

```bash
smartsim-update pydantic
smartsim-update loguru pyinstrument
```

**To update the SmartSim-CSC stack itself** (new SmartSim/SmartRedis/RedisAI version), pin a new `SMARTSIM_CSC_REF` in Section 1 and rerun the **full installer** (Section 5) for that architecture — `smartsim-update` intentionally cannot do this.

Updating also does not refresh the writable Julia runtime copy — re-run Section 5.1 afterward if Julia dependencies changed.

---

## 12. Rebuild / Clean Reinstall

```bash
# Re-run Section 1 first (re-detects arch/profile, re-asks INSTALL_PYSR)
rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
rm -rf "$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH" "$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
# For a full clean install, also: rm -rf "$SMARTREDIS_DIR" "$SMARTSIM_CSC_DIR"
```

If `INSTALL_PYSR` changed, regenerate `requirements.in` (Section 3.2). Build (Section 5) → Julia runtime prep (Section 5.1) → native SmartRedis build (Section 6, only if `$SMARTREDIS_DIR`/`$SMARTSIM_CSC_DIR` were removed).

---

## 13. Troubleshooting

**`SmartSim-CSC` checkout keeps re-cloning** — `extra4SmartSim.sh` only clones if `$SMARTSIM_CSC_DIR/.git` is missing; otherwise it fetches and checks out the pinned `SMARTSIM_CSC_REF` in place.

**`scripts/install.sh` fails on profile lookup** — confirm `SMARTSIM_CSC_PROFILE` matches an entry in `$SMARTSIM_CSC_DIR/stack.toml`; run `python scripts/stack_config.py --profile <name>` inside the checkout to debug.

**`linux-arm64-gpu` fails or behaves unexpectedly** — confirm that the pinned `SMARTSIM_CSC_REF` contains the profile, load the required CUDA module, and run `smart validate --device gpu` inside a GPU allocation.

**PySR still installed/missing despite the toggle** — check `install-options-$ENV_ARCH.sh`, confirm `requirements.in` was regenerated after changing it, and that Section 5 exported `INSTALL_PYSR` before calling `conda-containerize new`.

**`OSError: Read-only file system` importing `pysr`** — the one-time Julia copy (Section 5.1) was never run or was deleted; re-run it, then re-source the loader.

**Native SmartRedis build fails with missing source** — confirm `$SMARTSIM_CSC_DIR/components/smartredis` exists; if not, the monorepo checkout (Section 5's `extra4SmartSim.sh` step) didn't complete.

**`smart validate` reports fewer backends than expected** — only `onnxruntime` and `jax` are built by this stack; TensorFlow/LibTorch RedisAI backends are not part of `stack.toml` here.

**`PATH`/`LD_LIBRARY_PATH` grows on repeated sourcing** — shouldn't happen; the loader's `path_prepend` checks for existing entries.

**Home quota exceeded during build** — caches redirect to `$BASE_SCRATCH/.tykky_runtime_smartsim_*`, not `$HOME`.

**Architecture mismatch** — Tykky containers, native SmartRedis, and the SmartSim-CSC profile are all architecture-specific; never mix `x64` and `arm64` artifacts.

**OpenFOAM build reports that v2412 is not loaded** — load `gcc/15.2.0`, `openmpi/5.0.10`, and `openfoam/2412` after sourcing `Python4SmartSim.sh`.

**OpenFOAM build links against the wrong compiler runtime** — use the exact module order from Section 6.1 and do not source `Python4SmartSim.sh` again afterward.

**OpenFOAM executable reports a missing `libsmartredis.so`** — confirm that `$SMARTREDIS_DIR/install/lib64` exists and remains in `LD_LIBRARY_PATH`.

**Identity file not found** — go back to Section 0.

---

## 14. SmartSim Deployment Track

```text
Roihu CPU node                                Roihu GPU node
└─ SmartSim Orchestrator                       └─ SmartSim Orchestrator
   (Redis + RedisAI: ONNX Runtime + JAX)          (Redis + RedisAI: ONNX Runtime + JAX)
   └─ tensor/metric storage                        └─ JAX/Equinox(/PySR) training & inference
                                                       + tensor/metric storage
```

TensorFlow and PyTorch remain available as plain Python libraries for training/inference outside RedisAI; only ONNX and JAX/Equinox models can be registered and executed directly inside RedisAI via `set_model`/`run_model`:

```python
from smartredis import Client
import jax.numpy as jnp

client = Client(address="localhost:6379", cluster=False)
x = jnp.asarray(client.get_tensor("training_data"))
result = jax_function(x)
client.put_tensor("result", result)
```

---

## Notes

* This guide now targets **Roihu only**; Puhti and Mahti sections have been removed. Architecture is auto-detected from `uname -m`.
* SmartSim and SmartRedis come from the **[SmartSim-CSC](https://github.com/PentagonToy/SmartSim-CSC)** monorepo (pinned via `SMARTSIM_CSC_REF`), not from separate forks or PyPI.
* The `linux-x64-cpu`, `linux-arm64-cpu`, and `linux-arm64-gpu` profiles are validated. The GPU profile was tested on a Roihu GH200 compute node with CUDA 12.9; runtime validation and workloads require a GPU allocation.
* Only `onnxruntime` and `jax` RedisAI backends are built; TensorFlow and PyTorch remain regular Python packages, not RedisAI backends.
* **PySR/Julia is optional**, controlled per architecture by `INSTALL_PYSR`, independent of the SmartSim-CSC stack itself.
* The native SmartRedis library (Section 6) is copied from `components/smartredis` inside the SmartSim-CSC checkout and built per architecture, independent of `install.sh`/RedisAI and the PySR toggle.
* The bundled OpenFOAM integration is currently validated on Roihu x86_64 with OpenFOAM.com v2412. ARM64/GH200 OpenFOAM support has not yet been validated.
* `smartsim-update` never touches the SmartSim-CSC stack; bumping `SMARTSIM_CSC_REF` and re-running the full installer is the only supported way to update SmartSim/SmartRedis/RedisAI.
