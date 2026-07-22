# SmartSim Environment Configuration

Last updated: 22 July 2026

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

This folder deploys a **unified ML + SmartSim/SmartRedis stack** on CSC supercomputers (**Puhti / Mahti / Roihu**) — **JAX + Equinox + TensorFlow + PyTorch + ONNX + SmartSim + SmartRedis**, with **PySR (JuliaCall)** available as an **optional** add-on, all in one environment.

This environment is now a **superset of the previously separate `PythonML` stack**. Every package and workflow that used to live in the standalone ML environment — including PySR's Julia toolchain, if selected — is included here, built and validated on both `x64` and `arm64`. The earlier guidance to keep the ML and SmartSim stacks strictly separate no longer applies once PySR/Julia support is folded in this way; a standalone `PythonML/` environment is not needed if you use this stack.

**PySR/Julia is optional.** Some users never run symbolic regression and would rather skip the extra build time, disk usage, and Julia toolchain entirely. A single `INSTALL_PYSR` toggle, set once per architecture in Section 1 and persisted to `install-options-$ENV_ARCH.sh`, controls whether:

* `pysr`/`julia` are added to `requirements.in` at all,
* the Julia resolve/precompile step runs inside the Tykky build,
* the one-time writable Julia runtime copy (Section 5.1) is prepared,
* the loader (Section 7) configures `PYTHON_JULIAPKG_PROJECT`/`JULIA_DEPOT_PATH` at all.

Everything else in the stack — SmartSim, SmartRedis, JAX, TensorFlow, PyTorch, ONNX, RedisAI backends — is unaffected by this toggle and always installs.

SmartRedis installs from the CSC-maintained `v1.0.0-csc` release, while SmartSim installs from `v1.0.3-csc`. These releases already contain:

* **Python 3.12 support**
* **NumPy 2.x compatibility**
* **Linux ARM64 platform support** — upstream SmartSim only shipped a `Darwin+ARM64+CPU` platform config; the fork adds `Linux+ARM64+CPU` and the missing `aarch64`→`arm64` architecture-string mapping
* **RedisAI TensorFlow backend on Linux ARM64**
* **RedisAI ONNX Runtime backend on Linux ARM64**
* **RedisAI LibTorch backend on Linux ARM64**
* **SmartRedis compiler/source fixes**
* **RedisAI JAX backend with direct JAX/Equinox model registration**
* **Automatic JAX worker lifecycle management**
* **JAX output shape and `float32`/`float64`/`int32`/`int64` dtype support**
* **Polymorphic JAX batch dimensions through `jax_shape_specs`**

**No post-install source patching is required.** `smart build` runs identically on `x64` and `arm64`, building the RedisAI TensorFlow, ONNX Runtime, LibTorch, and JAX backends automatically.

When `INSTALL_PYSR=yes`, **PySR's Julia dependency is resolved and precompiled at build time**, exactly as it was in the standalone ML stack: `juliapkg` fetches Julia and the `PythonCall`/`SymbolicRegression` packages inside the Tykky build. Immediately after the Tykky build succeeds, the installer copies that packaged Julia *project* **once** into a writable scratch location (the Tykky image itself is read-only and `juliapkg` needs to write a lock file there); the Julia *depot* directory is created alongside it. Sourcing the loader afterwards only points environment variables at these already-prepared directories — it no longer copies or deletes anything. `PYTHON_JULIAPKG_OFFLINE=yes` at runtime prevents any accidental re-download. When `INSTALL_PYSR=no`, none of this happens — no Julia download, no `julia_env`/`julia_depot` directories, no runtime copy, and the loader skips the Julia block entirely.

RedisAI model execution is available for TensorFlow, ONNX, PyTorch (via LibTorch), and JAX/Equinox models. JAX or Equinox callables can be registered directly with `set_model(..., backend="JAX", example_inputs=..., jax_shape_specs=...)` and executed through `run_model`; SmartSim manages the persistent JAX worker automatically.

We use **Tykky** to package the whole Python stack into a single-file container image, avoiding Lustre metadata slowdowns from thousands of small file imports.

A Tykky container built for one architecture will not run on the other. The **SmartRedis native library** (Section 6, used for OpenFOAM/C++/Fortran linkage) is built separately per architecture via CMake, independent of the `smart build` step described above, and is unaffected by the PySR toggle.

**Why Tykky:** near-instant imports, a single reproducible image, fast startup, isolation from the host environment.

**Why uv:** fast resolution/installation, plus `uv pip check` to validate the final dependency graph. `--link-mode=copy` is used throughout since the uv cache and the Tykky build environment live on different filesystems.

```text
Python        3.12
SmartSim      1.0.1+csc (PentagonToy/SmartSim @ v1.0.3-csc)
SmartRedis    1.0.0+csc (PentagonToy/SmartRedis @ v1.0.0-csc)
JAX           resolved at build time (CUDA 12 on arm64)
TensorFlow    2.18.1
PyTorch       2.7.1
ONNX          resolved (+ ONNX Runtime, tf2onnx, skl2onnx)
PySR / Julia  OPTIONAL (INSTALL_PYSR=yes/no); resolved + precompiled at build time when enabled
NumPy         >= 2.0
protobuf      resolved by uv (no longer hard-pinned)
CMake         resolved (no longer pinned < 3.30.0)
RedisAI       TensorFlow + ONNX Runtime + LibTorch + JAX backends, built on both x64 and arm64
```

```text
requirements.in                  Human-maintained direct package specifications; pysr/julia present only if INSTALL_PYSR=yes
requirements-$ENV_ARCH.txt       Installed-state snapshot recorded after a successful build (excludes SmartSim/SmartRedis)
julia-environment-$ENV_ARCH.txt  Julia toolchain + package status; written only if INSTALL_PYSR=yes, otherwise a placeholder
install-options-$ENV_ARCH.sh     Persists INSTALL_PYSR across sessions
runtime-$ENV_ARCH.sh             GCC module + PySR-enabled flag recorded at build time, read by the loader on every source
```

Build order: install `uv` → install the full `requirements.in` set (TensorFlow, PyTorch, ONNX, and `pysr`/`julia` only if `INSTALL_PYSR=yes`) → resolve/precompile Julia+PySR (if enabled) → install SmartRedis (fork) → install SmartSim (fork) → `smart build` (Redis + RedisAI, all backends) → restore `requirements.in` → `uv pip check` → **prepare the writable Julia runtime once, if enabled** → build the native SmartRedis library and record its GCC module + PySR-enabled flag.

Part of the [CSC Environment Helpers Framework](https://github.com/PentagonToy/CSCEnvironmentHelpers). Production examples live in [SmartSim4CSC](https://github.com/PentagonToy/SmartSim4CSC).

---

## Build Flow

```text
Set identity once (Section 0)
  |
  v
Choose target architecture + PySR/Julia toggle (Section 1)
  |
  +-- x64  (Roihu CPU / Puhti / Mahti)
  |     Global Config (x64) --> choose INSTALL_PYSR=yes/no, persist to install-options-x64.sh
  |     --> install full requirements.in (pysr/julia included only if INSTALL_PYSR=yes)
  |     --> [if INSTALL_PYSR=yes] resolve/precompile Julia + PySR
  |     --> install SmartRedis v1.0.0-csc + SmartSim v1.0.3-csc
  |     --> build Tykky env (Redis + RedisAI TensorFlow/ONNX Runtime/LibTorch/JAX backends)
  |     --> [if INSTALL_PYSR=yes] prepare writable Julia runtime ONCE
  |     --> build SmartRedis-x64 native library; record GCC module + PySR flag used
  |
  +-- arm64 (Roihu GPU)
        Global Config (arm64) --> choose INSTALL_PYSR=yes/no, persist to install-options-arm64.sh
        --> install full requirements.in (pysr/julia included only if INSTALL_PYSR=yes)
        --> [if INSTALL_PYSR=yes] resolve/precompile Julia + PySR
        --> install SmartRedis v1.0.0-csc + SmartSim v1.0.3-csc
        --> build Tykky env (Redis + RedisAI TensorFlow/ONNX Runtime/LibTorch/JAX backends)
        --> [if INSTALL_PYSR=yes] prepare writable Julia runtime ONCE
        --> build SmartRedis-arm64 native library; record GCC module + PySR flag used
        --> also runs JAX/Equinox/TensorFlow/PyTorch(/PySR) training and inference locally

After the required track(s) are built:
  Create Python4SmartSim.sh --> source Python4SmartSim.sh
  --> loader picks x64/arm64 from `uname -m`, only sets environment variables
      and PATH/LD_LIBRARY_PATH/CMAKE_PREFIX_PATH (idempotent — safe to re-source)
  --> loader configures Julia env vars only if this architecture was built with INSTALL_PYSR=yes
  --> Jupyter kernels run through a launcher wrapper that sources the same loader
```

Skip the `arm64` track entirely if you never run workloads on Roihu GPU nodes against this stack. Each architecture can independently choose `INSTALL_PYSR=yes` or `no` — for example, PySR could be skipped on `arm64` while kept on `x64`.

---

## 0. One-Time Identity Configuration

Every script needs three values: your CSC project ID, your directory under that project, and the environment nickname. Set them **once** in a file under `$HOME`.

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

Edit with real values, then verify:

```bash
nano "$HOME/.config/csc-hpc/identity.sh"

source "$HOME/.config/csc-hpc/identity.sh"
echo "CSC_PROJECT=$CSC_PROJECT"
echo "PROJECT_USER_DIR=$PROJECT_USER_DIR"
echo "ENV_NICKNAME=$ENV_NICKNAME"
```

`ENV_ARCH` and `INSTALL_PYSR` are **not** part of this file — they're chosen per build in Section 1, and `ENV_ARCH` is auto-detected via `uname -m` in the loader / `smartsim-update`.

---

## 1. Global Configuration

Run **one** block per node. `ENV_ARCH` differs, and each architecture asks separately whether to install PySR/Julia.

### 1.1 x64 (Roihu CPU / Puhti / Mahti)

```bash
source "$HOME/.config/csc-hpc/identity.sh"
export ENV_ARCH="x64"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

# --- PySR / Julia toggle (per architecture) ---
read -r -p "Install PySR (symbolic regression) with its Julia toolchain for $ENV_ARCH? [Y/n]: " _INSTALL_PYSR_ANSWER
case "$_INSTALL_PYSR_ANSWER" in
    n|N|no|NO) export INSTALL_PYSR="no" ;;
    *)         export INSTALL_PYSR="yes" ;;
esac
unset _INSTALL_PYSR_ANSWER

cat <<EOF > "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
export INSTALL_PYSR="$INSTALL_PYSR"
EOF
chmod 600 "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

echo "ENV_ARCH=$ENV_ARCH"
echo "PYTHON_ROOT=$PYTHON_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
echo "INSTALL_PYSR=$INSTALL_PYSR"
```

### 1.2 arm64 (Roihu GPU)

```bash
source "$HOME/.config/csc-hpc/identity.sh"
export ENV_ARCH="arm64"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

# --- PySR / Julia toggle (per architecture) ---
read -r -p "Install PySR (symbolic regression) with its Julia toolchain for $ENV_ARCH? [Y/n]: " _INSTALL_PYSR_ANSWER
case "$_INSTALL_PYSR_ANSWER" in
    n|N|no|NO) export INSTALL_PYSR="no" ;;
    *)         export INSTALL_PYSR="yes" ;;
esac
unset _INSTALL_PYSR_ANSWER

cat <<EOF > "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
export INSTALL_PYSR="$INSTALL_PYSR"
EOF
chmod 600 "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

echo "ENV_ARCH=$ENV_ARCH"
echo "PYTHON_ROOT=$PYTHON_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
echo "INSTALL_PYSR=$INSTALL_PYSR"
```

Already know you want (or don't want) PySR and don't want the prompt? Skip the `read` line and just `export INSTALL_PYSR="yes"` or `export INSTALL_PYSR="no"` directly before writing `install-options-$ENV_ARCH.sh`.

**Directory layout:**

```text
/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities/          # $BASE_SCRATCH
├── .tykky_runtime_smartsim_x64/
├── .tykky_runtime_smartsim_arm64/
├── .julia_env_runtime_x64/       # created ONCE, right after the Tykky build (Section 5.1) — ONLY if INSTALL_PYSR=yes
├── .julia_env_runtime_arm64/     # created ONCE, right after the Tykky build (Section 5.1) — ONLY if INSTALL_PYSR=yes
├── .julia_depot_runtime_x64/     # created ONCE, right after the Tykky build (Section 5.1) — ONLY if INSTALL_PYSR=yes
├── .julia_depot_runtime_arm64/   # created ONCE, right after the Tykky build (Section 5.1) — ONLY if INSTALL_PYSR=yes
├── Python4SmartSim.sh
├── SmartRedis-x64/                                  # $SMARTREDIS_DIR (x64)
├── SmartRedis-arm64/                                # $SMARTREDIS_DIR (arm64)
└── Python/                                           # $PYTHON_BASE
    └── PythonSmartSim/                               # $PYTHON_ROOT
        ├── base4SmartSim.yml
        ├── extra4SmartSim.sh
        ├── update4SmartSim.sh
        ├── requirements.in
        ├── requirements-x64.txt
        ├── requirements-arm64.txt
        ├── julia-environment-x64.txt        # placeholder text if INSTALL_PYSR=no for x64
        ├── julia-environment-arm64.txt       # placeholder text if INSTALL_PYSR=no for arm64
        ├── install-options-x64.sh            # persists INSTALL_PYSR for x64
        ├── install-options-arm64.sh          # persists INSTALL_PYSR for arm64
        ├── runtime-x64.sh          # records GCC module + PySR-enabled flag for x64
        ├── runtime-arm64.sh        # records GCC module + PySR-enabled flag for arm64
        ├── jupyter-kernel-x64.sh   # kernel launcher wrapper (Section 8)
        ├── jupyter-kernel-arm64.sh # kernel launcher wrapper (Section 8)
        └── envs/
```

The `.julia_env_runtime_*` / `.julia_depot_runtime_*` directories, `runtime-$ENV_ARCH.sh`, and `install-options-$ENV_ARCH.sh` are created **once**, at build time (Sections 1, 5.1, and 6) — not by the loader. Sourcing `Python4SmartSim.sh` only ever reads them; it never copies, deletes, or recreates them. Re-run the installer for a given architecture if these directories are missing or out of date.

---

## 2. Dependency Overview

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.12 | Base interpreter |
| uv | latest at build | Resolution, installation, `uv pip check` |
| SmartSim | `1.0.1+csc` (`PentagonToy/SmartSim @ v1.0.3-csc`) | Orchestration; Redis, RedisAI, and JAX worker lifecycle |
| SmartRedis | `1.0.0+csc` (`PentagonToy/SmartRedis @ v1.0.0-csc`) | Python client with direct JAX/Equinox registration + native C++/Fortran library |
| JAX / Equinox / distrax / distreqx | resolved at build time; CUDA 12 on arm64 | Autodiff / training / inference / probabilistic modelling |
| TensorFlow | 2.18.1 | Python framework + source for the RedisAI TensorFlow backend |
| PyTorch | 2.7.1 | Python framework; RedisAI executes via the LibTorch backend |
| ONNX / ONNX Runtime | resolved at build time | Model interchange + Python-side ONNX Runtime |
| PySR / julia (JuliaCall) | **optional**, resolved | Symbolic regression; only installed and precompiled if `INSTALL_PYSR=yes` |
| shap | resolved | Model explainability |
| dvc | resolved | Data version control |
| nbconvert / papermill | resolved | Notebook execution and export |
| optuna / optuna-dashboard | resolved | Hyperparameter optimisation + web UI |
| NumPy | `>= 2.0` | No longer pinned below 2.0 |
| pydantic / loguru / pyinstrument | resolved | Config validation, structured logging, profiling |
| RedisAI backends | TensorFlow + ONNX Runtime + LibTorch + JAX, built on **both** architectures | JAX/Equinox and conventional ML model execution through SmartRedis |

---

## 3. Create the Configuration Files

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"

# Restore INSTALL_PYSR if this is a fresh shell.
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh" 2>/dev/null || true
: "${INSTALL_PYSR:=yes}"
echo "INSTALL_PYSR=$INSTALL_PYSR"
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

Do **not** add `smartsim` or `smartredis` here — both install separately from the fork in `extra4SmartSim.sh`. `pysr`/`julia` are appended conditionally, based on `INSTALL_PYSR`, rather than always being present.

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
    echo "Added pysr/julia to requirements.in (INSTALL_PYSR=yes)."
else
    echo "Skipped pysr/julia in requirements.in (INSTALL_PYSR=no)."
fi
```

`tensorflow==2.18.1` and `torch==2.7.1` remain pinned. `jax[cuda12]`, ONNX, ONNX Runtime, `protobuf`, and `numpy` are deliberately left unpinned and resolve to the newest compatible versions at build time. Python-side framework versions do not need to exactly match the RedisAI backend versions built by `smart build`, since they're separate runtime components — validate exported models against the corresponding RedisAI backend (Section 13) rather than assuming version equality. When `INSTALL_PYSR=yes`, `pysr`/`julia` also require the resolve/precompile step below — adding them to `requirements.in` alone is not sufficient.

### 3.3 `extra4SmartSim.sh` (post-install, runs *inside* the build)

This installs the full package set, conditionally resolves and precompiles PySR's Julia dependency (identical to the standalone ML stack's process, only when `INSTALL_PYSR=yes`), then installs and builds SmartSim/SmartRedis with all RedisAI backends. `smart build` runs identically on both architectures, with no runtime patching. `INSTALL_PYSR` is exported before the Tykky build is invoked (Section 5), so it's inherited here the same way `ENV_ARCH`/`PYTHON_ROOT`/`CW_BUILD_TMPDIR` already are.

```bash
cat <<'EOF' > "$PYTHON_ROOT/extra4SmartSim.sh"
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

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

if [ "$INSTALL_PYSR" = "yes" ]; then
    echo "INSTALL_PYSR=yes — resolving and precompiling PySR's Julia dependency..."

    # Always derive the Julia paths from the *actual* Python sys.prefix inside
    # the container, not from $ENV_PREFIX — Tykky's wrapper path and the real
    # in-container prefix are not the same thing.
    PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
    export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
    export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"
    mkdir -p "$JULIA_DEPOT_PATH" "$PYTHON_JULIAPKG_PROJECT"

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
else
    echo "INSTALL_PYSR=no — skipping PySR/Julia resolve and precompile."
fi

# --- CSC SmartRedis and SmartSim releases (both architectures) ---
# SmartRedis v1.0.0-csc and SmartSim v1.0.3-csc include the CSC
# platform/compiler fixes, RedisAI JAX backend, direct JAX/Equinox
# registration, polymorphic JAX batch support, and ARM64 runtimes.
uv pip install \
    --link-mode=copy \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@v1.0.0-csc"

uv pip install \
    --link-mode=copy \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@v1.0.3-csc"

# --- Build the Orchestrator (Redis + RedisAI backends) — both architectures ---
export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages

# Restore packages potentially disturbed by the build above
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

# Record installed versions; SmartSim/SmartRedis excluded (installed
# fresh from source every build, never replayed).
python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

if [ "$INSTALL_PYSR" = "yes" ]; then
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
else
    echo "PySR/Julia was not installed (INSTALL_PYSR=no)." \
        > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
fi

rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"
```

If `smart build` reports incompatible-pointer-type compile errors on some GCC versions, retry with `CFLAGS="-Wno-incompatible-pointer-types" CXXFLAGS="-Wno-incompatible-pointer-types"` prefixed to the `smart clobber`/`smart build` lines — see Section 13.

---

## 4. Request a Build Node

> **Tip — downloads:** the SmartRedis/SmartSim fork installs, `smart build`'s automatic backend downloads (TensorFlow, ONNX Runtime, LibTorch from GitHub Releases), and — if `INSTALL_PYSR=yes` — Julia's package resolution all need outbound internet access. If a compute allocation's network is restricted, try the download-heavy steps on the login node first.

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

If Section 1's variables aren't inherited, re-run the matching Global Configuration block (this also re-asks the PySR toggle — answer the same way to stay consistent, or `source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"` instead to reuse the previous answer without being asked again).

---

## 5. Build the Tykky Environment

```bash
module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

# Restore the PySR toggle in case this is a fresh shell on the build node.
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
export INSTALL_PYSR

ls -l \
    "$PYTHON_ROOT/base4SmartSim.yml" \
    "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/requirements.in"

echo "Building with INSTALL_PYSR=$INSTALL_PYSR"

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

Check the result:

```bash
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
ls -lh "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"

"$ENV_PREFIX/bin/python" -m pip list --format=freeze \
    | grep -E '^(jax|numpy|tensorflow|torch|onnx|onnxruntime|smartsim|smartredis)=='

# Only meaningful if INSTALL_PYSR=yes:
"$ENV_PREFIX/bin/python" -m pip list --format=freeze | grep -E '^(pysr|julia)=='
```

### 5.1 Prepare the writable PySR / Julia runtime (once, only if `INSTALL_PYSR=yes`)

The Tykky image is read-only, but `juliapkg` needs to write a lock file into the Julia project directory the first time it's used. Rather than doing this copy on every `source` (as in earlier versions of this guide), it now happens **once**, right after the Tykky build succeeds — and only if PySR was actually installed:

```bash
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

if [ "$INSTALL_PYSR" = "yes" ]; then
    echo "Preparing writable PySR / Julia runtime..."

    PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"
    JULIA_ENV_SOURCE="$PYTHON_PREFIX/julia_env"
    JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
    JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

    if [ ! -d "$JULIA_ENV_SOURCE" ]; then
        echo "ERROR: Packaged Julia environment was not found: $JULIA_ENV_SOURCE"
        echo "       Was this Tykky environment actually built with INSTALL_PYSR=yes?"
        exit 1
    fi

    rm -rf "$JULIA_ENV_RUNTIME"
    cp -a "$JULIA_ENV_SOURCE" "$JULIA_ENV_RUNTIME"
    mkdir -p "$JULIA_DEPOT_RUNTIME"

    echo "Julia environment: $JULIA_ENV_RUNTIME"
    echo "Julia depot:       $JULIA_DEPOT_RUNTIME"
else
    echo "INSTALL_PYSR=no — skipping writable Julia runtime preparation."
    rm -rf "$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH" "$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
fi
```

Rerun this block (and only this block) if the Tykky environment is rebuilt for this architecture — the packaged Julia project may have changed.

Build the other architecture separately (Section 1 + Section 4). Its `INSTALL_PYSR` choice is independent.

---

## 6. Build the SmartRedis Native Library

Needed on **both** architectures for OpenFOAM/C++/Fortran linkage — this is a separate CMake build, unrelated to `smart build`/RedisAI or the Julia toolchain, and unaffected by `INSTALL_PYSR`.

Request a node (Section 4), then set the GCC module for your target system and load compilers:

```bash
module purge
```

Compilers, e.g. Roihu CPU:

```bash
# Roihu CPU
export GCC_MODULE="gcc/13.4.0"
module load "$GCC_MODULE"
module load cmake/3.26.5
```

Roihu GPU:

```bash
# Roihu GPU
export GCC_MODULE="gcc/13.4.0"
module load "$GCC_MODULE"
module load cmake/3.31.11
```

or Mahti:

```bash
export GCC_MODULE="gcc/13.1.0"
module load "$GCC_MODULE"
module load cmake/3.28.6
module load git
```

Record the GCC module and the PySR-enabled flag for the loader so it never has to guess:

```bash
source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

cat <<EOF > "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
export SMARTSIM_GCC_MODULE="$GCC_MODULE"
export SMARTSIM_PYSR_ENABLED="$INSTALL_PYSR"
EOF
chmod 600 "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
```

Clone and build:

```bash
cd "$BASE_SCRATCH"
rm -rf "$SMARTREDIS_DIR"

git clone \
    --branch v1.0.0-csc \
    https://github.com/PentagonToy/SmartRedis.git \
    "$SMARTREDIS_DIR"

cd "$SMARTREDIS_DIR"
rm -rf build install
```

```bash
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
ls -la "$SMARTREDIS_DIR/install/lib64"
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library installed successfully."
ldd "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so"
```

---

## 7. Loader — `Python4SmartSim.sh`

The loader is a **pure environment loader** — it must be *sourced*, not executed, and re-sourcing it repeatedly in the same shell is safe. It:

* validates that the Tykky environment and SmartRedis install exist;
* reads `runtime-$ENV_ARCH.sh` to learn which GCC module the native SmartRedis library was built against (loading it only if not already loaded, via `module is-loaded`), **and** whether this architecture was built with `INSTALL_PYSR=yes`;
* prepends to `PATH`, `LD_LIBRARY_PATH`, and `CMAKE_PREFIX_PATH` through a small `path_prepend` helper that skips directories already present, so re-sourcing never creates duplicate entries;
* configures Julia/PySR environment variables **only if** `SMARTSIM_PYSR_ENABLED=yes` was recorded at build time — otherwise it skips that block entirely, with no error;
* never copies, deletes, or installs anything.

```bash
cat <<'EOF' > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash
#
# SmartSim Python environment loader
#
# Usage:
#   source /scratch/<project>/<user>/Utilities/Python4SmartSim.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This file must be sourced, not executed:"
    echo "    source ${BASH_SOURCE[0]}"
    exit 1
fi

IDENTITY_FILE="$HOME/.config/csc-hpc/identity.sh"

if [ ! -f "$IDENTITY_FILE" ]; then
    echo "Identity file not found: $IDENTITY_FILE"
    return 1
fi

source "$IDENTITY_FILE"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"

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
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "Python environment not found:"
    echo "    $ENV_PREFIX"
    return 1
fi

if [ ! -d "$SMARTREDIS_DIR/install" ]; then
    echo "SmartRedis installation not found:"
    echo "    $SMARTREDIS_DIR/install"
    return 1
fi

RUNTIME_CONFIG="$PYTHON_ROOT/runtime-$ENV_ARCH.sh"

if [ -f "$RUNTIME_CONFIG" ]; then
    source "$RUNTIME_CONFIG"
fi

# Default to "yes" for backward compatibility with environments built before
# this flag existed (they always installed PySR/Julia).
: "${SMARTSIM_PYSR_ENABLED:=yes}"

if [ -n "${SMARTSIM_GCC_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module is-loaded "$SMARTSIM_GCC_MODULE" 2>/dev/null ||
        module load "$SMARTSIM_GCC_MODULE"
fi

path_prepend() {
    local variable_name="$1"
    local directory="$2"
    local current_value="${!variable_name-}"

    case ":$current_value:" in
        *":$directory:"*)
            ;;
        *)
            printf -v "$variable_name" '%s' \
                "$directory${current_value:+:$current_value}"
            export "$variable_name"
            ;;
    esac
}

if [ -d "$SMARTREDIS_DIR/install/lib64" ]; then
    export SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib64"
elif [ -d "$SMARTREDIS_DIR/install/lib" ]; then
    export SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib"
else
    echo "SmartRedis library directory not found."
    return 1
fi

path_prepend PATH "$ENV_PREFIX/bin"
path_prepend LD_LIBRARY_PATH "$SMARTREDIS_LIB_DIR"
path_prepend CMAKE_PREFIX_PATH "$SMARTREDIS_DIR/install"

export SMARTSIM_DB_FILE_PARSE_TRIALS=600

export PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"

if [ "$SMARTSIM_PYSR_ENABLED" = "yes" ]; then
    # PySR / Julia runtime paths — prepared ONCE at build time (Sections 5.1/6);
    # this loader only points environment variables at them.
    export JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
    export JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

    if [ ! -d "$JULIA_ENV_RUNTIME" ]; then
        echo "Writable Julia environment not found:"
        echo "    $JULIA_ENV_RUNTIME"
        echo "Run the SmartSim installer again for $ENV_ARCH (Section 5.1)."
        return 1
    fi

    mkdir -p "$JULIA_DEPOT_RUNTIME"

    export PYTHON_JULIAPKG_PROJECT="$JULIA_ENV_RUNTIME"
    export JULIA_DEPOT_PATH="$JULIA_DEPOT_RUNTIME:$PYTHON_PREFIX/julia_depot"
    export PYTHON_JULIAPKG_OFFLINE="yes"
    export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK:-auto}"

    unset PYTHON_JULIACALL_EXE
    unset PYTHON_JULIACALL_PROJECT
else
    # This architecture was built with INSTALL_PYSR=no — make sure no stale
    # Julia environment variables leak in from a previous session.
    unset JULIA_ENV_RUNTIME JULIA_DEPOT_RUNTIME
    unset PYTHON_JULIAPKG_PROJECT JULIA_DEPOT_PATH PYTHON_JULIAPKG_OFFLINE
    unset PYTHON_JULIACALL_THREADS PYTHON_JULIACALL_EXE PYTHON_JULIACALL_PROJECT
fi

export JUPYTER_KERNEL_NAME="$ENV_NICKNAME-smartsim-$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.12 ($ENV_NICKNAME SmartSim $KERNEL_ARCH)"
export JUPYTER_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$JUPYTER_KERNEL_NAME"

if [ "${SMARTSIM_ENV_QUIET:-0}" != "1" ]; then
    echo "SmartSim Python environment loaded"
    echo "ENV_ARCH=$ENV_ARCH"
    echo "ENV_PREFIX=$ENV_PREFIX"
    echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
    echo "JAX_PLATFORMS=$JAX_PLATFORMS"
    echo "SMARTSIM_PYSR_ENABLED=$SMARTSIM_PYSR_ENABLED"
    if [ "$SMARTSIM_PYSR_ENABLED" = "yes" ]; then
        echo "PYTHON_JULIAPKG_PROJECT=$PYTHON_JULIAPKG_PROJECT"
    fi
fi

unset -f path_prepend
EOF
chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"
```

Load it:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
echo "$PYTHON_ROOT"; echo "$ENV_PREFIX"; echo "$SMARTREDIS_DIR"
python --version
```

Confirm re-sourcing doesn't duplicate `PATH`:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
source "$BASE_SCRATCH/Python4SmartSim.sh"
echo "$PATH" | tr ':' '\n' | grep PythonSmartSim
```

The environment path should appear exactly once.

`SMARTSIM_ENV_QUIET=1` suppresses the status banner — used internally by the Jupyter kernel launcher (Section 8).

---

## 8. Register the Jupyter Kernel

Run once per architecture. Rather than baking `ENV_PREFIX` and Julia/JAX environment variables directly into `kernel.json`, the kernel runs through a small launcher wrapper that sources the exact same loader used interactively — so a notebook kernel always matches a terminal session, including whether PySR/Julia is even configured.

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"

# --- Kernel launcher wrapper ---
JUPYTER_KERNEL_LAUNCHER="$PYTHON_ROOT/jupyter-kernel-$ENV_ARCH.sh"

cat <<EOF > "$JUPYTER_KERNEL_LAUNCHER"
#!/bin/bash
export SMARTSIM_ENV_QUIET=1
source "$BASE_SCRATCH/Python4SmartSim.sh" || exit 1
unset SMARTSIM_ENV_QUIET
exec "$ENV_PREFIX/bin/python" -m ipykernel_launcher "\$@"
EOF
chmod +x "$JUPYTER_KERNEL_LAUNCHER"

# --- kernel.json pointing at the wrapper ---
mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": [
    "$JUPYTER_KERNEL_LAUNCHER",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF

jupyter kernelspec list
```

Remove an obsolete kernel: `jupyter kernelspec uninstall -f <kernel_name>`. In VS Code: **Command Palette → Developer: Reload Window**.

---

## 9. Validate the Environment

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"

python -c "
import sys
import numpy, jax, equinox, tensorflow, torch, onnx, onnxruntime
import smartsim, smartredis

print(f'Python:       {sys.version.split()[0]}')
print(f'NumPy:        {numpy.__version__}')
print(f'JAX:          {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}')
print(f'Equinox:      {equinox.__version__}')
print(f'TensorFlow:   {tensorflow.__version__}')
print(f'PyTorch:      {torch.__version__}')
print(f'ONNX:         {onnx.__version__}')
print(f'ONNXRuntime:  {onnxruntime.__version__}')
print(f'SmartSim:     {smartsim.__version__}')
print(f'SmartRedis:   {smartredis.__version__}')
"
```

```bash
uv pip check
smart validate --device cpu
```

`smart validate` should report TensorFlow, ONNX Runtime, LibTorch, and JAX backends as available on both architectures — regardless of the PySR toggle.

**PySR / JuliaCall** (only meaningful if this architecture was built with `INSTALL_PYSR=yes`; should not download or install anything at this point — that already happened at build time):

```bash
if [ "${SMARTSIM_PYSR_ENABLED:-no}" = "yes" ]; then
    python - <<'PY'
import juliapkg
import pysr
from juliacall import Main as jl
print(f"PySR version:     {pysr.__version__}")
print(f"Julia executable: {juliapkg.executable()}")
print(f"Julia version:    {jl.VERSION}")
PY
else
    echo "PySR/Julia was not installed for $ENV_ARCH (INSTALL_PYSR=no) — nothing to validate here."
fi
```

Native library check (both architectures):

```bash
ls -la "$SMARTREDIS_DIR/install/lib64"
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library is available."
```

---

## 10. Dependency File Workflow

```text
requirements.in            Human-maintained direct dependencies (not SmartSim/SmartRedis themselves); pysr/julia present only if INSTALL_PYSR=yes
requirements-$ENV_ARCH.txt Installed-state snapshot (excludes SmartSim/SmartRedis)
```

**Add/remove a package** — edit `requirements.in`, then rebuild/update (Section 11 or 12). Removing a package needs a full rebuild to drop unused transitive deps.

**Keep `jax[cuda12]` unpinned** to pick up the newest compatible JAX release; `tensorflow` and `torch` remain pinned in `requirements.in`, while ONNX/ONNX Runtime/`protobuf`/`numpy` resolve at build time. Python-side TensorFlow/PyTorch/ONNX Runtime versions are independent of the RedisAI backend binaries `smart build` produces — exact equality isn't required, but validate exported models with `set_model`/`run_model` before production use.

**Turning PySR on or off later** — edit `install-options-$ENV_ARCH.sh` to flip `INSTALL_PYSR`, then re-run Section 3.2 to regenerate `requirements.in` with/without the `pysr`/`julia` lines, and do a **full rebuild** (Section 12). A `conda-containerize update` alone is not reliable for adding or removing the Julia toolchain, since Julia's own dependency resolution and precompilation only happen cleanly in a fresh build.

**Reproduce an exact installed set** — temporarily point `extra4SmartSim.sh`'s two `uv pip install --requirements ...` lines at `requirements-$ENV_ARCH.txt`, rebuild, then switch back.

---

## 11. Updating the Environment

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

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"
if [ -s "$UPDATE_REQUEST" ]; then
    mapfile -t UPDATE_PACKAGES < "$UPDATE_REQUEST"
    uv pip install --link-mode=copy --upgrade "${UPDATE_PACKAGES[@]}"
fi

if [ "$INSTALL_PYSR" = "yes" ]; then
    # Keep the packaged Julia environment ready for PySR
    PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
    export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
    export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"

    python - <<'PY'
import juliapkg
import pysr
juliapkg.resolve()
print(f"PySR version:     {pysr.__version__}")
print(f"Julia executable: {juliapkg.executable()}")
PY

    python - <<'PY'
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
subprocess.run(
    [julia, f"--project={project}", "-e",
     "using Pkg; Pkg.instantiate(); Pkg.precompile()"],
    check=True,
)
PY
else
    echo "INSTALL_PYSR=no — skipping Julia/PySR maintenance during update."
fi

uv pip install \
    --link-mode=copy \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@v1.0.0-csc"

uv pip install \
    --link-mode=copy \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@v1.0.3-csc"

export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

rm -f "$UPDATE_REQUEST"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF

chmod +x "$PYTHON_ROOT/update4SmartSim.sh"
```

Create `smartsim-update`:

```bash
mkdir -p "$HOME/bin"

cat <<'EOF' > "$HOME/bin/smartsim-update"
#!/bin/bash -l
set -e

if [ "$#" -eq 0 ]; then
    echo "Usage: smartsim-update <package> [package ...]"
    exit 1
fi

if [ ! -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "Identity file not found: $HOME/.config/csc-hpc/identity.sh"
    exit 1
fi

source "$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"

case "$(uname -m)" in
    x86_64) export ENV_ARCH="x64" ;;
    aarch64) export ENV_ARCH="arm64" ;;
    *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"
export UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ -f "$PYTHON_ROOT/install-options-$ENV_ARCH.sh" ]; then
    source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
fi
export INSTALL_PYSR="${INSTALL_PYSR:-yes}"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "Environment not found: $ENV_PREFIX"; exit 1
fi

if [ ! -f "$PYTHON_ROOT/requirements.in" ]; then
    echo "requirements.in not found: $PYTHON_ROOT/requirements.in"; exit 1
fi

for package in "$@"; do
    package_name="$(printf '%s\n' "$package" | sed -E 's/\[.*//; s/[<>=!~].*//')"
    case "$package_name" in
        smartsim|smartredis)
            echo "$package_name is managed separately and must not be added to requirements.in."
            exit 1
            ;;
        pysr|julia)
            if [ "$INSTALL_PYSR" != "yes" ]; then
                echo "$package_name requires INSTALL_PYSR=yes for this architecture."
                echo "Edit $PYTHON_ROOT/install-options-$ENV_ARCH.sh and do a full rebuild (Section 12) instead."
                exit 1
            fi
            ;;
    esac
done

printf '%s\n' "$@" > "$UPDATE_REQUEST"

python - "$PYTHON_ROOT/requirements.in" "$@" <<'PY'
import re, sys
from pathlib import Path

requirements_file = Path(sys.argv[1])
requested = sys.argv[2:]
lines = requirements_file.read_text().splitlines()

def package_name(spec):
    return re.split(r"[\[<>=!~]", spec, maxsplit=1)[0].strip().lower()

for spec in requested:
    name = package_name(spec)
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or " @ " in stripped:
            continue
        if package_name(stripped) == name:
            lines[index] = spec
            replaced = True
            print(f"Updated requirement: {spec}")
            break
    if not replaced:
        lines.append(spec)
        print(f"Added requirement: {spec}")

requirements_file.write_text("\n".join(lines) + "\n")
PY

module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize update \
    --post-install "$PYTHON_ROOT/update4SmartSim.sh" \
    "$ENV_PREFIX"

echo "Update completed. Recorded packages: $PYTHON_ROOT/requirements-$ENV_ARCH.txt"
EOF

chmod +x "$HOME/bin/smartsim-update"
```

```bash
grep -qxF 'export PATH="$HOME/bin:$PATH"' ~/.bashrc || \
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Apply on the matching architecture node:

```bash
smartsim-update pydantic
smartsim-update loguru pyinstrument
```

`smartsim-update` refuses `pysr`/`julia` if this architecture was built with `INSTALL_PYSR=no` — see Section 10 for how to switch it on (full rebuild required).

Updating does **not** rebuild the native SmartRedis library, and does **not** refresh the writable Julia runtime copy under `.julia_env_runtime_$ENV_ARCH` — since `conda-containerize update` repacks the Julia project inside the container, re-run the copy step in Section 5.1 afterwards if PySR's Julia dependencies changed (only relevant when `INSTALL_PYSR=yes`).

---

## 12. Rebuild / Clean Reinstall

```bash
# 1) Run the matching Global Configuration block (Section 1) first — this
#    will re-ask (or let you reuse) the INSTALL_PYSR choice for this architecture.
echo "ENV_ARCH=$ENV_ARCH"; echo "ENV_PREFIX=$ENV_PREFIX"; echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "INSTALL_PYSR=$INSTALL_PYSR"

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
# Also clear the writable Julia runtime copy, since it's derived from this build:
rm -rf "$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH" "$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
# For a full clean install, also: rm -rf "$SMARTREDIS_DIR"

ls -l "$PYTHON_ROOT/base4SmartSim.yml" "$PYTHON_ROOT/requirements.in" "$PYTHON_ROOT/extra4SmartSim.sh"
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"
```

If you changed `INSTALL_PYSR` since the last build, also regenerate `requirements.in` (Section 3.2) before rebuilding, so `pysr`/`julia` are added or removed accordingly.

Request a node (Section 4), then build (Section 5), followed immediately by the one-time Julia runtime preparation (Section 5.1, a no-op if `INSTALL_PYSR=no`). If you removed `$SMARTREDIS_DIR`, rebuild it too (Section 6), which also rewrites `runtime-$ENV_ARCH.sh`.

---

## 13. Troubleshooting

**Total reset:**
```bash
rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```
Rebuild per Section 12.

**`requirements-$ENV_ARCH.txt` missing** — only written after a successful build; run Section 5.

**PySR still gets installed even though I chose "no"** — check `$PYTHON_ROOT/install-options-$ENV_ARCH.sh`; confirm `INSTALL_PYSR="no"` there, that `requirements.in` (Section 3.2) was regenerated *after* setting it, and that Section 5 sourced the same file (`export INSTALL_PYSR` line) before calling `conda-containerize new`.

**I want to add PySR after skipping it** — edit `install-options-$ENV_ARCH.sh` to `INSTALL_PYSR="yes"`, regenerate `requirements.in` (Section 3.2), and do a full rebuild (Section 12) followed by Section 5.1. `smartsim-update` deliberately refuses to add `pysr`/`julia` for this reason.

**PySR tries to download Julia/packages at runtime instead of using the precompiled build** — confirm `$JULIA_ENV_RUNTIME` (Section 5.1) exists and that `PYTHON_JULIAPKG_OFFLINE=yes` is set: `python -c "import os; print(os.environ.get('PYTHON_JULIAPKG_OFFLINE'))"` should print `yes`. This only applies if `INSTALL_PYSR=yes` for this architecture.

**`OSError: Read-only file system` when importing `pysr`** — the one-time Julia-project copy (Section 5.1) was never run for this architecture, or its target directory was deleted. Re-run Section 5.1, then re-source `Python4SmartSim.sh`; the loader itself no longer performs any copy and will refuse to load if `$JULIA_ENV_RUNTIME` is missing while `SMARTSIM_PYSR_ENABLED=yes`.

**`import pysr` fails with `ModuleNotFoundError`** — this architecture was likely built with `INSTALL_PYSR=no`. Check `runtime-$ENV_ARCH.sh` for `SMARTSIM_PYSR_ENABLED`; if it says `no`, that's expected — see "I want to add PySR after skipping it" above.

**"This file must be sourced, not executed" when running the loader** — run it with `source Python4SmartSim.sh`, not `./Python4SmartSim.sh` or `bash Python4SmartSim.sh`.

**`PATH`/`LD_LIBRARY_PATH` grows every time the loader is sourced** — this should no longer happen; the loader's `path_prepend` helper checks for existing entries before prepending. If it does happen, confirm you're using the updated loader from Section 7, not an older version.

**GitHub release installation fails** — confirm outbound network access from the build node and verify that the SmartRedis `v1.0.0-csc` and SmartSim `v1.0.3-csc` tags are accessible.

**`smart build` reports incompatible-pointer-type compile errors** — retry with `CFLAGS="-Wno-incompatible-pointer-types" CXXFLAGS="-Wno-incompatible-pointer-types"` prefixed to `smart clobber`/`smart build`.

**`smart build` rejects `--skip-python-packages`** — run `smart build --help` inside the build environment to get the ground-truth flags for whatever version is installed.

**TensorFlow/PyTorch/ONNX Runtime model compatibility with RedisAI** — Python package versions and RedisAI backend versions are separate and don't need to match exactly. Validate actual exported TensorFlow, TorchScript, and ONNX models with `set_model`/`run_model`. If a model uses operators/formats the RedisAI backend doesn't support, re-export with an older compatible opset or framework format.

**uv hardlink warning** — expected; `--link-mode=copy` handles it.

**Home quota exceeded during build** — caches redirect to `$BASE_SCRATCH/.tykky_runtime_smartsim_*`, not `$HOME`.

**Architecture mismatch** — build and use the matching-architecture Tykky environment; no cross-architecture container.

**JAX reports no GPU** — loader sets `JAX_PLATFORMS` automatically; avoid `JAX_PLATFORMS=gpu`.

**SmartRedis native library not found** — check `$LD_LIBRARY_PATH`, confirm files under `install/lib64` (or `lib`), re-source the loader.

**Jupyter kernel doesn't see the same environment as the terminal** — confirm `kernel.json`'s `argv` points at `jupyter-kernel-$ENV_ARCH.sh` (Section 8), not directly at `$ENV_PREFIX/bin/python`; the wrapper sources the loader before launching `ipykernel_launcher`.

**Import errors after an update** — run `uv pip check`; prefer a full rebuild (Section 12) over stacking updates.

**Identity file not found** — go back to Section 0.

---

## 14. SmartSim Deployment Track

Each architecture runs its own local Orchestrator, and can also run JAX/Equinox/TensorFlow/PyTorch(/PySR, if installed) workloads locally:

```text
x64 CPU node                              arm64 GPU node
└─ SmartSim Orchestrator                  └─ SmartSim Orchestrator
   (Redis + RedisAI: TF/ONNX/LibTorch)       (Redis + RedisAI: TF/ONNX/LibTorch)
   └─ tensor/weight/metric storage            └─ JAX/Equinox/TF/PyTorch(/PySR) training & inference
                                                  + tensor/weight/metric storage
```

RedisAI model execution is available but optional. TensorFlow, ONNX, and PyTorch (via LibTorch) models can be executed directly inside RedisAI with `set_model`/`run_model`; the primary workflow may still run JAX/Equinox(/PySR) inference in external Python workers, with SmartRedis carrying tensors either way:

```python
from smartredis import Client
import jax.numpy as jnp

client = Client(address="localhost:6379", cluster=False)
x = jnp.asarray(client.get_tensor("training_data"))
result = jax_function(x)          # runs on the GPU
client.put_tensor("result", result)
```

Other typical workflows: launching OpenFOAM solvers + Python producers/consumers through Slurm; linking external C++/Fortran solvers against the native SmartRedis client; symbolic regression on simulation output with PySR (if installed); validating producer/consumer config with `pydantic`, logging with `loguru`, profiling with `pyinstrument`; DVC-tracked datasets; Papermill-driven notebook pipelines.

Full production architecture and Slurm templates: [SmartSim4CSC](https://github.com/PentagonToy/SmartSim4CSC).

---

## Notes

* Python 3.12, built separately per architecture — never mix containers across architectures.
* **This environment is now a superset of the previously standalone ML stack.** It includes everything the ML environment had — including PySR/Julia when selected — plus SmartSim/SmartRedis and RedisAI's TensorFlow/ONNX Runtime/LibTorch backends. A separate `PythonML/` environment is not needed if you use this stack.
* **PySR/Julia is optional**, controlled per architecture by `INSTALL_PYSR` (Section 1), persisted to `install-options-$ENV_ARCH.sh`, and recorded again in `runtime-$ENV_ARCH.sh` as `SMARTSIM_PYSR_ENABLED` for the loader to read. Skipping it saves build time and disk space for users who never run symbolic regression; switching it on later requires a full rebuild (Section 10/12), not just an update.
* SmartSim and SmartRedis install from the CSC releases (`PentagonToy/SmartSim @ v1.0.3-csc`, `PentagonToy/SmartRedis @ v1.0.0-csc`), not PyPI — no runtime patching remains in `extra4SmartSim.sh` / `update4SmartSim.sh`, and neither is affected by the PySR toggle.
* When `INSTALL_PYSR=yes`, **PySR's Julia dependency is resolved and precompiled at build time**, exactly as in the standalone ML stack, and the writable runtime copy of the Julia project is created **once**, immediately after a successful Tykky build (Section 5.1) — not on every `source`. The Julia depot (precompiled packages) stays read-only and is layered in via `JULIA_DEPOT_PATH`. `PYTHON_JULIAPKG_OFFLINE=yes` prevents any runtime re-download. None of this happens when `INSTALL_PYSR=no`.
* The loader (`Python4SmartSim.sh`) is a **pure loader**: it must be sourced (not executed), is idempotent across repeated sourcing thanks to a `path_prepend` helper, only loads the GCC module recorded in `runtime-$ENV_ARCH.sh` if it isn't already loaded, and only configures Julia/PySR variables if `SMARTSIM_PYSR_ENABLED=yes` was recorded for this architecture.
* Jupyter kernels run through a **launcher wrapper** (`jupyter-kernel-$ENV_ARCH.sh`) that sources the loader before starting `ipykernel_launcher`, so notebook kernels — including under VS Code — always match an interactive terminal session, PySR toggle included, rather than duplicating environment variables inside `kernel.json`.
* `smart build` runs identically on both architectures and builds all three RedisAI backends by default; `--skip-python-packages` is used because TensorFlow/PyTorch/ONNX Python packages are already managed via `requirements.in`.
* `jax[cuda12]`, `onnx`, `onnxruntime`, `numpy`, and `protobuf` are intentionally unpinned and resolve to the newest compatible versions at build time. The exact installed versions are recorded in `requirements-$ENV_ARCH.txt` — validate the resolved environment with `uv pip check`.
* Placeholders (`Harry`/`Dumbledore`/`project_xxxxxxx`) are set once in the identity file (Section 0).
* `requirements.in` = direct deps, not SmartSim/SmartRedis themselves; `requirements-$ENV_ARCH.txt` = installed-state snapshot, not a lockfile.
* Every `uv pip install` uses `--link-mode=copy`.
* The native SmartRedis library (Section 6) is unrelated to `smart build`/RedisAI and the Julia toolchain, and is built on both architectures as usual regardless of the PySR toggle. The GCC module needed for it varies by target system (Roihu, Mahti, Puhti) and is recorded once in `runtime-$ENV_ARCH.sh` at build time, alongside the PySR-enabled flag, rather than guessed from hostname at every `source`.
