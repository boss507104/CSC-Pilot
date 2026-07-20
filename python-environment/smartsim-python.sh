#!/bin/bash
# smartsim-python.sh
# Interactive installer for the unified Python 3.12 ML + SmartSim/SmartRedis
# Tykky environment, native SmartRedis library, smartsim-update command,
# PySR/Julia runtime setup, and Jupyter kernel registration.
#
# Intended location:
#   /scratch/$CSC_PROJECT/$PROJECT_USER_DIR/smartsim-python.sh
#
# Run once per architecture:
#   - x64:   Roihu CPU / Puhti / Mahti x86_64 node
#   - arm64: Roihu GPU aarch64 node
#
# SmartSim and SmartRedis install from the CSC-maintained csc-develop forks.
# These forks already include Python 3.12 support, NumPy 2.x compatibility,
# Linux ARM64 support, RedisAI TensorFlow/ONNX Runtime/LibTorch backends on
# Linux ARM64, and the required SmartRedis compiler/source fixes.
#
# No post-install source patching is performed.
#
# This script performs installation only. It skips the manual validation,
# dependency-workflow notes, troubleshooting, and deployment examples from
# the full guide.

set -e

echo "=================================================================="
echo " Unified ML + SmartSim Environment Installer"
echo "=================================================================="
echo
echo "WARNING: This script runs the Tykky build and native SmartRedis"
echo "compilation on the current node. Use a node with the target"
echo "architecture and sufficient resources."
echo

# ------------------------------------------------------------------
# Project number: double verification
# ------------------------------------------------------------------
prompt_project_number() {
    local first second

    while true; do
        read -r -p "Type project number: " first
        read -r -p "Type project number (verification): " second

        if [ -z "$first" ]; then
            echo "Project number cannot be empty."
            echo
            continue
        fi

        if [ "$first" != "$second" ]; then
            echo "Project numbers did not match. Try again."
            echo
            continue
        fi

        RAW_PROJECT="$first"
        return
    done
}

# ------------------------------------------------------------------
# Single-entry prompt
# ------------------------------------------------------------------
prompt_value() {
    local prompt_text="$1"
    local result_variable="$2"
    local value

    while true; do
        read -r -p "${prompt_text}: " value

        if [ -z "$value" ]; then
            echo "Value cannot be empty."
            echo
            continue
        fi

        printf -v "$result_variable" '%s' "$value"
        return
    done
}

# ------------------------------------------------------------------
# Architecture prompt
# ------------------------------------------------------------------
prompt_architecture() {
    local value

    while true; do
        read -r -p "Type node or architecture (cpu / gpu / x64 / arm64): " value
        value="$(echo "$value" | tr '[:upper:]' '[:lower:]' | xargs)"

        case "$value" in
            cpu|x64)
                ENV_ARCH="x64"
                return
                ;;
            gpu|arm64)
                ENV_ARCH="arm64"
                return
                ;;
            *)
                echo "Invalid choice. Enter cpu, gpu, x64, or arm64."
                echo
                ;;
        esac
    done
}

# ------------------------------------------------------------------
# Target system prompt
# ------------------------------------------------------------------
prompt_system() {
    local value

    while true; do
        read -r -p "Type target system (roihu / mahti / puhti): " value
        value="$(echo "$value" | tr '[:upper:]' '[:lower:]' | xargs)"

        case "$value" in
            roihu|mahti|puhti)
                TARGET_SYSTEM="$value"
                return
                ;;
            *)
                echo "Invalid choice. Enter roihu, mahti, or puhti."
                echo
                ;;
        esac
    done
}

# ------------------------------------------------------------------
# 1. Collect configuration
# ------------------------------------------------------------------
echo "--- Project identity ---"
prompt_project_number

if [[ "$RAW_PROJECT" == project_* ]]; then
    CSC_PROJECT="$RAW_PROJECT"
else
    CSC_PROJECT="project_${RAW_PROJECT}"
fi

prompt_value "Type project user directory name" PROJECT_USER_DIR
prompt_value "Type environment nickname" ENV_NICKNAME

echo
echo "--- Target architecture ---"
prompt_architecture

echo
echo "--- Target system ---"
prompt_system

case "$TARGET_SYSTEM" in
    roihu)
        GCC_MODULE="gcc/13.4.0"
        if [ "$ENV_ARCH" = "x64" ]; then
            CMAKE_MODULE="cmake/3.26.5"
        else
            CMAKE_MODULE="cmake/3.31.11"
        fi
        LOAD_GIT_MODULE="no"
        ;;
    mahti)
        GCC_MODULE="gcc/13.1.0"
        CMAKE_MODULE="cmake/3.28.6"
        LOAD_GIT_MODULE="yes"
        ;;
    puhti)
        echo
        echo "The guide does not define fixed Puhti compiler modules."
        prompt_value "Type GCC module, for example gcc/13.1.0" GCC_MODULE
        prompt_value "Type CMake module, for example cmake/3.28.6" CMAKE_MODULE
        LOAD_GIT_MODULE="yes"
        ;;
esac

echo
echo "--- Configuration ---"
echo "CSC_PROJECT       = $CSC_PROJECT"
echo "PROJECT_USER_DIR  = $PROJECT_USER_DIR"
echo "ENV_NICKNAME      = $ENV_NICKNAME"
echo "ENV_ARCH          = $ENV_ARCH"
echo "TARGET_SYSTEM     = $TARGET_SYSTEM"
echo "GCC_MODULE        = $GCC_MODULE"
echo "CMAKE_MODULE      = $CMAKE_MODULE"
echo "Python            = 3.12"
echo "SmartSim fork     = PentagonToy/SmartSim @ csc-develop"
echo "SmartRedis fork   = PentagonToy/SmartRedis @ csc-develop"
echo "RedisAI backends  = TensorFlow + ONNX Runtime + LibTorch"
echo "PySR / Julia      = resolved and precompiled during build"
echo

HOST_ARCH="$(uname -m)"

if [ "$ENV_ARCH" = "arm64" ] && [ "$HOST_ARCH" != "aarch64" ]; then
    echo "WARNING: arm64 was selected, but this host reports '$HOST_ARCH'."
    echo "The generated environment and native library would use the host"
    echo "architecture rather than ARM64."
    echo
fi

if [ "$ENV_ARCH" = "x64" ] && [ "$HOST_ARCH" != "x86_64" ]; then
    echo "WARNING: x64 was selected, but this host reports '$HOST_ARCH'."
    echo "The generated environment and native library would use the host"
    echo "architecture rather than x86_64."
    echo
fi

read -r -p "Proceed with this configuration? [y/N]: " CONFIRM_ALL
case "$CONFIRM_ALL" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
esac
echo

# ------------------------------------------------------------------
# 2. Identity file
# ------------------------------------------------------------------
echo "[1/10] Writing identity file..."

mkdir -p "$HOME/.config/csc-hpc"

cat <<EOF > "$HOME/.config/csc-hpc/identity.sh"
export CSC_PROJECT="$CSC_PROJECT"
export PROJECT_USER_DIR="$PROJECT_USER_DIR"
export ENV_NICKNAME="$ENV_NICKNAME"
EOF

chmod 600 "$HOME/.config/csc-hpc/identity.sh"

echo "      $HOME/.config/csc-hpc/identity.sh"
echo

# ------------------------------------------------------------------
# 3. Global paths
# ------------------------------------------------------------------
echo "[2/10] Setting paths..."

source "$HOME/.config/csc-hpc/identity.sh"

export ENV_ARCH
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "      ENV_ARCH=$ENV_ARCH"
echo "      PYTHON_ROOT=$PYTHON_ROOT"
echo "      ENV_PREFIX=$ENV_PREFIX"
echo "      SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "      TMP_BUILD_DIR=$TMP_BUILD_DIR"
echo

# ------------------------------------------------------------------
# 4. Configuration files
# ------------------------------------------------------------------
echo "[3/10] Creating configuration files..."

mkdir -p "$PYTHON_ROOT"

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
DataGraph @ git+https://github.com/PentagonToy/DataGraph.git#subdirectory=DataGraph
eqx_io @ git+https://github.com/PentagonToy/CSC-HPC-Guide.git#subdirectory=utilities/eqx4smartredis

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

cat <<'EOF' > "$PYTHON_ROOT/extra4SmartSim.sh"
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

python -m pip install --no-cache-dir uv

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# Resolve and precompile PySR's Julia dependency using the actual
# in-container Python prefix.
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
import subprocess
import juliapkg

julia = juliapkg.executable()
project = juliapkg.project()

subprocess.run(
    [
        julia,
        f"--project={project}",
        "-e",
        (
            "using Pkg; "
            "Pkg.instantiate(); "
            "Pkg.precompile(); "
            "using PythonCall; "
            "using SymbolicRegression"
        ),
    ],
    check=True,
)
PY

# Install SmartRedis and SmartSim from the CSC-maintained forks.
uv pip install \
    --link-mode=copy \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@csc-develop"

uv pip install \
    --link-mode=copy \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@csc-develop"

# Build Redis and all RedisAI backends on both architectures.
export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages

# Restore packages potentially changed by smart build.
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

python - <<'PY' > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
import subprocess
import juliapkg

julia = juliapkg.executable()
project = juliapkg.project()

print(f"Julia executable: {julia}")
print(f"Julia project: {project}\n")

subprocess.run(
    [
        julia,
        f"--project={project}",
        "-e",
        "using InteractiveUtils; versioninfo(); using Pkg; Pkg.status()",
    ],
    check=True,
)
PY

rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF

chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"

echo "      Created base4SmartSim.yml"
echo "      Created requirements.in"
echo "      Created extra4SmartSim.sh"
echo

# ------------------------------------------------------------------
# 5. Build Tykky environment
# ------------------------------------------------------------------
echo "[4/10] Building the Tykky environment..."

module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"

echo
echo "      Tykky environment built:"
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
ls -lh "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
echo

# ------------------------------------------------------------------
# 6. Native SmartRedis library
# ------------------------------------------------------------------
echo "[5/10] Loading native-build modules..."

module purge
module load "$GCC_MODULE"
module load "$CMAKE_MODULE"

if [ "$LOAD_GIT_MODULE" = "yes" ]; then
    module load git
fi

echo "      Loaded $GCC_MODULE"
echo "      Loaded $CMAKE_MODULE"
echo

echo "[6/10] Building the native SmartRedis library..."

cd "$BASE_SCRATCH"
rm -rf "$SMARTREDIS_DIR"

git clone \
    --branch csc-develop \
    https://github.com/PentagonToy/SmartRedis.git \
    "$SMARTREDIS_DIR"

cd "$SMARTREDIS_DIR"
rm -rf build install

env \
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran

echo
echo "[7/10] Verifying the native SmartRedis library..."

find "$SMARTREDIS_DIR/install" -maxdepth 3 -type f | sort

if [ -d "$SMARTREDIS_DIR/install/lib64" ]; then
    LIB_DIR="lib64"
else
    LIB_DIR="lib"
fi

echo "      Native library directory: install/$LIB_DIR"
ls -la "$SMARTREDIS_DIR/install/$LIB_DIR"

if [ -f "$SMARTREDIS_DIR/install/$LIB_DIR/libsmartredis-fortran.so" ]; then
    echo "      SmartRedis Fortran library installed successfully."
else
    echo "ERROR: libsmartredis-fortran.so was not found."
    exit 1
fi

ldd "$SMARTREDIS_DIR/install/$LIB_DIR/libsmartredis-fortran.so"
echo

# ------------------------------------------------------------------
# 7. Loader
# ------------------------------------------------------------------
echo "[8/10] Creating loader and update tooling..."

cat <<'EOF' > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash

if [ ! -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "Identity file not found: $HOME/.config/csc-hpc/identity.sh"
    return 1
fi

source "$HOME/.config/csc-hpc/identity.sh"

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

case "${HOSTNAME:-}" in
    *roihu*)
        module load gcc/13.4.0
        ;;
    *mahti*)
        module load gcc/13.1.0
        ;;
    *puhti*)
        echo "Load the GCC module matching the Puhti SmartRedis native build."
        ;;
    *)
        echo "Unrecognised host. Load the GCC module matching the native build."
        ;;
esac

export PATH="$ENV_PREFIX/bin:$PATH"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "Environment not found for $ENV_ARCH: $ENV_PREFIX"
    return 1
fi

if [ -d "$SMARTREDIS_DIR/install/lib64" ]; then
    SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib64"
else
    SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib"
fi

export LD_LIBRARY_PATH="$SMARTREDIS_LIB_DIR:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$SMARTREDIS_DIR/install:${CMAKE_PREFIX_PATH:-}"

export SMARTSIM_DB_FILE_PARSE_TRIALS=600

# PySR / Julia runtime setup.
export PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
export JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

source = Path(sys.prefix) / "julia_env"
target = Path(os.environ["JULIA_ENV_RUNTIME"])

shutil.rmtree(target, ignore_errors=True)
shutil.copytree(source, target)

Path(os.environ["JULIA_DEPOT_RUNTIME"]).mkdir(
    parents=True,
    exist_ok=True,
)
PY

export PYTHON_JULIAPKG_PROJECT="$JULIA_ENV_RUNTIME"
export JULIA_DEPOT_PATH="$JULIA_DEPOT_RUNTIME:$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_OFFLINE="yes"
export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK:-auto}"

unset PYTHON_JULIACALL_EXE
unset PYTHON_JULIACALL_PROJECT

export JUPYTER_KERNEL_NAME="$ENV_NICKNAME-smartsim-$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.12 ($ENV_NICKNAME SmartSim $KERNEL_ARCH)"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share/$KERNEL_ARCH}"
export JUPYTER_KERNEL_DIR="$XDG_DATA_HOME/jupyter/kernels/$JUPYTER_KERNEL_NAME"

echo "ENV_ARCH=$ENV_ARCH"
echo "PYTHON_ROOT=$PYTHON_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"
echo "PYTHON_JULIAPKG_PROJECT=$PYTHON_JULIAPKG_PROJECT"
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
EOF

chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"

echo "      Created $BASE_SCRATCH/Python4SmartSim.sh"

# ------------------------------------------------------------------
# 8. update4SmartSim.sh
# ------------------------------------------------------------------
cat <<'EOF' > "$PYTHON_ROOT/update4SmartSim.sh"
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

python -m pip install --no-cache-dir uv

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ -s "$UPDATE_REQUEST" ]; then
    mapfile -t UPDATE_PACKAGES < "$UPDATE_REQUEST"

    uv pip install \
        --link-mode=copy \
        --upgrade \
        "${UPDATE_PACKAGES[@]}"
fi

# Keep the packaged Julia environment ready for PySR.
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
import subprocess
import juliapkg

julia = juliapkg.executable()
project = juliapkg.project()

subprocess.run(
    [
        julia,
        f"--project={project}",
        "-e",
        "using Pkg; Pkg.instantiate(); Pkg.precompile()",
    ],
    check=True,
)
PY

uv pip install \
    --link-mode=copy \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@csc-develop"

uv pip install \
    --link-mode=copy \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@csc-develop"

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

echo "      Created $PYTHON_ROOT/update4SmartSim.sh"

# ------------------------------------------------------------------
# 9. smartsim-update
# ------------------------------------------------------------------
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
    x86_64)
        export ENV_ARCH="x64"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"
export UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "Environment not found: $ENV_PREFIX"
    exit 1
fi

if [ ! -f "$PYTHON_ROOT/requirements.in" ]; then
    echo "requirements.in not found: $PYTHON_ROOT/requirements.in"
    exit 1
fi

for package in "$@"; do
    package_name="$(printf '%s\n' "$package" | sed -E 's/\[.*//; s/[<>=!~].*//')"

    case "$package_name" in
        smartsim|smartredis)
            echo "$package_name is managed separately and must not be added to requirements.in."
            exit 1
            ;;
    esac
done

printf '%s\n' "$@" > "$UPDATE_REQUEST"

python - "$PYTHON_ROOT/requirements.in" "$@" <<'PY'
import re
import sys
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

echo "Update completed."
echo "Recorded packages: $PYTHON_ROOT/requirements-$ENV_ARCH.txt"
EOF

chmod +x "$HOME/bin/smartsim-update"

grep -qxF 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" || \
    echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"

echo "      Created $HOME/bin/smartsim-update"
echo

# ------------------------------------------------------------------
# 10. Jupyter kernel
# ------------------------------------------------------------------
echo "[9/10] Registering the Jupyter kernel..."

source "$BASE_SCRATCH/Python4SmartSim.sh"

mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": ["$ENV_PREFIX/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": { "debugger": true },
  "env": {
    "JAX_PLATFORMS": "$JAX_PLATFORMS",
    "PYTHON_JULIAPKG_PROJECT": "$PYTHON_JULIAPKG_PROJECT",
    "JULIA_DEPOT_PATH": "$JULIA_DEPOT_PATH",
    "PYTHON_JULIAPKG_OFFLINE": "yes",
    "PYTHON_JULIACALL_THREADS": "auto"
  }
}
EOF

echo "      Created $JUPYTER_KERNEL_DIR/kernel.json"
echo "      Registered $JUPYTER_KERNEL_NAME"
echo

if command -v jupyter >/dev/null 2>&1; then
    jupyter kernelspec list 2>/dev/null || true
fi

echo "[10/10] Installation complete."
echo "=================================================================="
echo
echo "Load the environment:"
echo "    source \"$BASE_SCRATCH/Python4SmartSim.sh\""
echo
echo "Update or add packages:"
echo "    smartsim-update pydantic"
echo "    smartsim-update loguru pyinstrument"
echo
echo "The environment uses Python 3.12 and includes:"
echo "    JAX + Equinox"
echo "    TensorFlow 2.18.1"
echo "    PyTorch 2.7.1"
echo "    ONNX + ONNX Runtime"
echo "    PySR + JuliaCall"
echo "    SmartSim + SmartRedis CSC forks"
echo "    RedisAI TensorFlow + ONNX Runtime + LibTorch backends"
echo
echo "No SmartSim or SmartRedis source patching was applied."
echo
echo "Run the script again on the other architecture when both x64 and"
echo "arm64 environments are required. Use the same identity values."
