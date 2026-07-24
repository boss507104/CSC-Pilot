#!/bin/bash
# smartsim-python.sh
# Interactive installer for the unified Python 3.12 ML + SmartSim/SmartRedis
# Tykky environment, native SmartRedis library, optional OpenFOAM v2412
# integration, smartsim-update command, optional PySR/Julia runtime setup,
# and Jupyter kernel registration.
#
# Intended location:
#   /scratch/$CSC_PROJECT/$PROJECT_USER_DIR/smartsim-python.sh
#
# This installer targets CSC's Roihu supercomputer only.
#
# The target profile is selected automatically from the current Roihu node:
#   - x86_64  -> linux-x64-cpu
#   - aarch64 -> linux-arm64-gpu
#
# SmartSim and SmartRedis are installed from the unified SmartSim-CSC
# monorepo. The exact component versions, RedisAI backends, and platform
# assets are defined by the pinned SmartSim-CSC ref and its stack.toml.
#
# No post-install source patching is performed. On x86_64, the installer can
# optionally build the bundled OpenFOAM.com v2412 integration after the
# native SmartRedis library has been installed.
#
# PySR (and its Julia toolchain) is OPTIONAL and is asked about separately
# for each architecture. Answering "no" skips it entirely: it is left out of
# requirements.in, no Julia resolve/precompile step runs during the build,
# no writable Julia runtime is prepared, and the loader never sets up
# PYTHON_JULIAPKG_PROJECT / JULIA_DEPOT_PATH for that architecture.
#
# This script performs installation only. It skips the manual validation,
# dependency-workflow notes, troubleshooting, and deployment examples from
# the full guide.

set -e

echo "=================================================================="
echo " Unified ML + SmartSim Environment Installer (Roihu only)"
echo "=================================================================="
echo
echo "WARNING: This script runs the Tykky build and native SmartRedis"
echo "compilation on the current node. Use the Roihu node architecture"
echo "for which the environment is intended."
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
# Architecture and SmartSim-CSC profile detection
# ------------------------------------------------------------------
detect_architecture() {
    case "$(uname -m)" in
        x86_64)
            ENV_ARCH="x64"
            SMARTSIM_CSC_PROFILE="linux-x64-cpu"
            ;;
        aarch64)
            ENV_ARCH="arm64"
            SMARTSIM_CSC_PROFILE="linux-arm64-gpu"
            ;;
        *)
            echo "Unsupported Roihu architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
}

# ------------------------------------------------------------------
# PySR / Julia toggle prompt
# ------------------------------------------------------------------
prompt_install_pysr() {
    local value

    while true; do
        read -r -p "Install PySR (symbolic regression) with its Julia toolchain? [Y/n]: " value
        value="$(echo "$value" | tr '[:upper:]' '[:lower:]' | xargs)"

        case "$value" in
            ""|y|yes)
                INSTALL_PYSR="yes"
                return
                ;;
            n|no)
                INSTALL_PYSR="no"
                return
                ;;
            *)
                echo "Invalid choice. Enter y or n."
                echo
                ;;
        esac
    done
}

# ------------------------------------------------------------------
# OpenFOAM v2412 toggle prompt (x86_64 only)
# ------------------------------------------------------------------
prompt_build_openfoam() {
    local value

    if [ "$ENV_ARCH" != "x64" ]; then
        BUILD_OPENFOAM="no"
        return
    fi

    while true; do
        read -r -p "Build the bundled OpenFOAM v2412 integration? [Y/n]: " value
        value="$(echo "$value" | tr '[:upper:]' '[:lower:]' | xargs)"

        case "$value" in
            ""|y|yes)
                BUILD_OPENFOAM="yes"
                return
                ;;
            n|no)
                BUILD_OPENFOAM="no"
                return
                ;;
            *)
                echo "Invalid choice. Enter y or n."
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
detect_architecture
echo "Detected $(uname -m): ENV_ARCH=$ENV_ARCH, PROFILE=$SMARTSIM_CSC_PROFILE"

echo
echo "--- Optional PySR / Julia toolchain ---"
prompt_install_pysr

echo
echo "--- Optional OpenFOAM integration ---"
prompt_build_openfoam

# Roihu compiler and CUDA modules.
if [ "$ENV_ARCH" = "x64" ]; then
    GCC_MODULE="gcc/13.4.0"
    CMAKE_MODULE="cmake/3.26.5"
    CUDA_MODULE=""
else
    GCC_MODULE="gcc/14.3.0"
    CMAKE_MODULE="cmake/3.31.11"
    CUDA_MODULE="cuda/12.9.1"
fi

echo
echo "--- Configuration ---"
echo "CSC_PROJECT       = $CSC_PROJECT"
echo "PROJECT_USER_DIR  = $PROJECT_USER_DIR"
echo "ENV_NICKNAME      = $ENV_NICKNAME"
echo "ENV_ARCH          = $ENV_ARCH"
echo "GCC_MODULE        = $GCC_MODULE"
echo "CMAKE_MODULE      = $CMAKE_MODULE"
echo "CUDA_MODULE       = ${CUDA_MODULE:-none}"
echo "Python            = 3.12"
echo "SmartSim-CSC repo = https://github.com/PentagonToy/SmartSim-CSC.git"
echo "SmartSim-CSC ref  = ${SMARTSIM_CSC_REF:-3d4749d}"
echo "SmartSim profile  = $SMARTSIM_CSC_PROFILE"
if [ "$BUILD_OPENFOAM" = "yes" ]; then
    echo "OpenFOAM v2412    = BUILD on x86_64"
else
    echo "OpenFOAM v2412    = SKIPPED"
fi
if [ "$INSTALL_PYSR" = "yes" ]; then
    echo "PySR / Julia      = resolved and precompiled during build"
else
    echo "PySR / Julia      = SKIPPED (INSTALL_PYSR=no)"
fi
echo

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
export INSTALL_PYSR
export BUILD_OPENFOAM
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTSIM_CSC_REPO="${SMARTSIM_CSC_REPO:-https://github.com/PentagonToy/SmartSim-CSC.git}"
export SMARTSIM_CSC_REF="${SMARTSIM_CSC_REF:-3d4749d}"
export SMARTSIM_CSC_DIR="$PYTHON_ROOT/src/SmartSim-CSC"
export SMARTSIM_CSC_PROFILE
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export OPENFOAM_USER_DIR="$BASE_SCRATCH/OpenFOAM/OpenFOAM-v2412"
export OPENFOAM_USER_DIR="$BASE_SCRATCH/OpenFOAM/OpenFOAM-v2412"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

# Persist the PySR toggle for this architecture so later sessions
# (updates, loader, rebuilds) can recover it without re-asking.
cat <<EOF > "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
export INSTALL_PYSR="$INSTALL_PYSR"
EOF
chmod 600 "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"

echo "      ENV_ARCH=$ENV_ARCH"
echo "      PYTHON_ROOT=$PYTHON_ROOT"
echo "      ENV_PREFIX=$ENV_PREFIX"
echo "      SMARTSIM_CSC_DIR=$SMARTSIM_CSC_DIR"
echo "      SMARTSIM_CSC_REF=$SMARTSIM_CSC_REF"
echo "      SMARTSIM_CSC_PROFILE=$SMARTSIM_CSC_PROFILE"
echo "      SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "      OPENFOAM_USER_DIR=$OPENFOAM_USER_DIR"
echo "      BUILD_OPENFOAM=$BUILD_OPENFOAM"
echo "      TMP_BUILD_DIR=$TMP_BUILD_DIR"
echo "      INSTALL_PYSR=$INSTALL_PYSR"
echo "      Recorded: $PYTHON_ROOT/install-options-$ENV_ARCH.sh"
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

# pysr/julia are appended conditionally, based on INSTALL_PYSR, rather than
# always being present.
if [ "$INSTALL_PYSR" = "yes" ]; then
    cat <<'EOF' >> "$PYTHON_ROOT/requirements.in"

# --- Symbolic Regression & Julia ---
pysr
julia
EOF
    echo "      Added pysr/julia to requirements.in (INSTALL_PYSR=yes)."
else
    echo "      Skipped pysr/julia in requirements.in (INSTALL_PYSR=no)."
fi

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

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

if [ "$INSTALL_PYSR" = "yes" ]; then
    echo "INSTALL_PYSR=yes - resolving and precompiling PySR's Julia dependency..."

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
else
    echo "INSTALL_PYSR=no - skipping PySR/Julia resolve and precompile."
fi

# Install the unified SmartSim-CSC stack.
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
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

if [ "$INSTALL_PYSR" = "yes" ]; then
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
else
    echo "PySR/Julia was not installed (INSTALL_PYSR=no)." \
        > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
fi

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
module load "$GCC_MODULE"
echo "      Loaded $GCC_MODULE"

if [ -n "$CUDA_MODULE" ]; then
    module load "$CUDA_MODULE"
    echo "      Loaded $CUDA_MODULE"
fi

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
export INSTALL_PYSR
export SMARTSIM_CSC_REPO SMARTSIM_CSC_REF SMARTSIM_CSC_DIR SMARTSIM_CSC_PROFILE

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"

echo
echo "      Tykky environment built (INSTALL_PYSR=$INSTALL_PYSR):"
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
ls -lh "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
echo

# ------------------------------------------------------------------
# 5b. Prepare writable PySR / Julia runtime once (only if INSTALL_PYSR=yes)
# ------------------------------------------------------------------
if [ "$INSTALL_PYSR" = "yes" ]; then
    echo "      Preparing writable PySR / Julia runtime..."

    PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"
    JULIA_ENV_SOURCE="$PYTHON_PREFIX/julia_env"
    JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
    JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

    if [ ! -d "$JULIA_ENV_SOURCE" ]; then
        echo "ERROR: Packaged Julia environment was not found:"
        echo "       $JULIA_ENV_SOURCE"
        exit 1
    fi

    rm -rf "$JULIA_ENV_RUNTIME"
    cp -a "$JULIA_ENV_SOURCE" "$JULIA_ENV_RUNTIME"
    mkdir -p "$JULIA_DEPOT_RUNTIME"

    echo "      Julia environment: $JULIA_ENV_RUNTIME"
    echo "      Julia depot:       $JULIA_DEPOT_RUNTIME"
else
    echo "      INSTALL_PYSR=no - skipping writable Julia runtime preparation."
    rm -rf "$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH" "$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
fi
echo

# ------------------------------------------------------------------
# 6. Native SmartRedis library
# ------------------------------------------------------------------
echo "[5/10] Loading native-build modules..."

module purge
module load "$GCC_MODULE"
module load "$CMAKE_MODULE"

echo "      Loaded $GCC_MODULE"
echo "      Loaded $CMAKE_MODULE"
echo

# Record runtime modules and the PySR-enabled flag for the loader so it
# never has to guess.
cat <<EOF > "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
export SMARTSIM_GCC_MODULE="$GCC_MODULE"
export SMARTSIM_CUDA_MODULE="$CUDA_MODULE"
export SMARTSIM_PYSR_ENABLED="$INSTALL_PYSR"
export SMARTSIM_OPENFOAM_ENABLED="$BUILD_OPENFOAM"
EOF
chmod 600 "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"

echo "[6/10] Building the native SmartRedis library..."

if [ ! -d "$SMARTSIM_CSC_DIR/components/smartredis" ]; then
    echo "SmartRedis source was not found in the SmartSim-CSC checkout:"
    echo "    $SMARTSIM_CSC_DIR/components/smartredis"
    exit 1
fi

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
# 6b. Optional OpenFOAM v2412 integration (x86_64 only)
# ------------------------------------------------------------------
if [ "$BUILD_OPENFOAM" = "yes" ]; then
    echo "[8/11] Building the OpenFOAM v2412 integration..."

    if [ "$ENV_ARCH" != "x64" ]; then
        echo "ERROR: OpenFOAM integration is currently supported only on x86_64."
        exit 1
    fi

    if [ ! -x "$SMARTSIM_CSC_DIR/scripts/openfoam/build-openfoam-v2412.sh" ]; then
        echo "OpenFOAM build script was not found:"
        echo "    $SMARTSIM_CSC_DIR/scripts/openfoam/build-openfoam-v2412.sh"
        exit 1
    fi

    module --force purge
    module load gcc/15.2.0
    module load openmpi/5.0.10
    module load openfoam/2412

    export FOAM_USER_DIR="$OPENFOAM_USER_DIR"
    mkdir -p "$FOAM_USER_DIR"

    cd "$SMARTSIM_CSC_DIR"
    ./scripts/openfoam/build-openfoam-v2412.sh

    echo
    echo "      OpenFOAM integration installed:"
    echo "      FOAM_USER_APPBIN=$FOAM_USER_APPBIN"
    echo "      FOAM_USER_LIBBIN=$FOAM_USER_LIBBIN"

    for executable in foamSmartSimSvd foamSmartSimSvdDBAPI svdToFoam; do
        if [ ! -x "$FOAM_USER_APPBIN/$executable" ]; then
            echo "ERROR: Missing OpenFOAM executable: $FOAM_USER_APPBIN/$executable"
            exit 1
        fi
    done

    if ldd "$FOAM_USER_APPBIN/foamSmartSimSvdDBAPI" | grep -q "not found"; then
        echo "ERROR: OpenFOAM executable has unresolved shared libraries."
        ldd "$FOAM_USER_APPBIN/foamSmartSimSvdDBAPI"
        exit 1
    fi

    echo "      OpenFOAM v2412 integration built successfully."
else
    echo "      BUILD_OPENFOAM=no - skipping OpenFOAM integration build."
fi
echo

# Restore the SmartSim runtime compiler modules before creating and sourcing
# the normal Python loader.
module --force purge
module load "$GCC_MODULE"
module load "$CMAKE_MODULE"

# ------------------------------------------------------------------
# 7. Loader
# ------------------------------------------------------------------
echo "[9/11] Creating loader and update tooling..."

cat <<'EOF' > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash
#
# SmartSim Python environment loader (Roihu only)
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
export SMARTSIM_CSC_DIR="$PYTHON_ROOT/src/SmartSim-CSC"

case "$(uname -m)" in
    x86_64)
        export ENV_ARCH="x64"
        export KERNEL_ARCH="x86_64"
        export JAX_PLATFORMS="cpu"
        export SMARTSIM_CSC_PROFILE="linux-x64-cpu"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        export KERNEL_ARCH="aarch64"
        export JAX_PLATFORMS="cuda"
        export SMARTSIM_CSC_PROFILE="linux-arm64-gpu"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        return 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export OPENFOAM_USER_DIR="$BASE_SCRATCH/OpenFOAM/OpenFOAM-v2412"

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

# Default to "yes" for backward compatibility with environments built
# before this flag existed (they always installed PySR/Julia).
: "${SMARTSIM_PYSR_ENABLED:=yes}"
: "${SMARTSIM_OPENFOAM_ENABLED:=no}"

if [ -n "${SMARTSIM_GCC_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module is-loaded "$SMARTSIM_GCC_MODULE" 2>/dev/null ||
        module load "$SMARTSIM_GCC_MODULE"
fi

if [ -n "${SMARTSIM_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module is-loaded "$SMARTSIM_CUDA_MODULE" 2>/dev/null ||
        module load "$SMARTSIM_CUDA_MODULE"
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
    # PySR / Julia runtime paths - prepared ONCE at build time; this loader
    # only points environment variables at them.
    export JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
    export JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

    if [ ! -d "$JULIA_ENV_RUNTIME" ]; then
        echo "Writable Julia environment not found:"
        echo "    $JULIA_ENV_RUNTIME"
        echo "Run the SmartSim installer again for $ENV_ARCH."
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
    # This architecture was built with INSTALL_PYSR=no - make sure no
    # stale Julia environment variables leak in from a previous session.
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
    echo "SMARTSIM_CSC_DIR=$SMARTSIM_CSC_DIR"
    echo "SMARTSIM_CSC_PROFILE=$SMARTSIM_CSC_PROFILE"
    echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
    echo "JAX_PLATFORMS=$JAX_PLATFORMS"
    echo "SMARTSIM_PYSR_ENABLED=$SMARTSIM_PYSR_ENABLED"
echo "SMARTSIM_OPENFOAM_ENABLED=$SMARTSIM_OPENFOAM_ENABLED"
if [ "$SMARTSIM_OPENFOAM_ENABLED" = "yes" ]; then
    echo "OPENFOAM_USER_DIR=$OPENFOAM_USER_DIR"
fi
    if [ "$SMARTSIM_PYSR_ENABLED" = "yes" ]; then
        echo "PYTHON_JULIAPKG_PROJECT=$PYTHON_JULIAPKG_PROJECT"
    fi
fi

unset -f path_prepend
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

    uv pip install \
        --link-mode=copy \
        --upgrade \
        "${UPDATE_PACKAGES[@]}"
fi

if [ "$INSTALL_PYSR" = "yes" ]; then
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
else
    echo "INSTALL_PYSR=no - skipping Julia/PySR maintenance during update."
fi

# SmartSim and SmartRedis are owned by the pinned SmartSim-CSC checkout.
# Package updates must not reinstall or rebuild the SmartSim stack.
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
        export SMARTSIM_CSC_PROFILE="linux-x64-cpu"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        export SMARTSIM_CSC_PROFILE="linux-arm64-gpu"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"
export UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ -f "$PYTHON_ROOT/install-options-$ENV_ARCH.sh" ]; then
    source "$PYTHON_ROOT/install-options-$ENV_ARCH.sh"
fi
export INSTALL_PYSR="${INSTALL_PYSR:-yes}"

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
        smartsim|smartredis|jax|jaxlib|jax-cuda12-plugin|jax-cuda12-pjrt)
            echo "$package_name is managed by SmartSim-CSC and must not be updated with smartsim-update."
            exit 1
            ;;
        pysr|julia)
            if [ "$INSTALL_PYSR" != "yes" ]; then
                echo "$package_name requires INSTALL_PYSR=yes for this architecture ($ENV_ARCH)."
                echo "Edit $PYTHON_ROOT/install-options-$ENV_ARCH.sh and do a full rebuild instead."
                exit 1
            fi
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
echo "[10/11] Registering the Jupyter kernel..."

source "$BASE_SCRATCH/Python4SmartSim.sh"

# ------------------------------------------------------------------
# 10a. Kernel launcher wrapper
# ------------------------------------------------------------------
JUPYTER_KERNEL_LAUNCHER="$PYTHON_ROOT/jupyter-kernel-$ENV_ARCH.sh"

cat <<EOF > "$JUPYTER_KERNEL_LAUNCHER"
#!/bin/bash
export SMARTSIM_ENV_QUIET=1
source "$BASE_SCRATCH/Python4SmartSim.sh" || exit 1
unset SMARTSIM_ENV_QUIET
exec "$ENV_PREFIX/bin/python" -m ipykernel_launcher "\$@"
EOF

chmod +x "$JUPYTER_KERNEL_LAUNCHER"

echo "      Created $JUPYTER_KERNEL_LAUNCHER"

# ------------------------------------------------------------------
# 10b. kernel.json using the launcher
# ------------------------------------------------------------------
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

echo "      Created $JUPYTER_KERNEL_DIR/kernel.json"
echo "      Registered $JUPYTER_KERNEL_NAME"
echo

if command -v jupyter >/dev/null 2>&1; then
    jupyter kernelspec list 2>/dev/null || true
fi

echo "[11/11] Installation complete."
echo "=================================================================="
echo
echo "Load the environment:"
echo "    source \"$BASE_SCRATCH/Python4SmartSim.sh\""
echo
echo "Update or add packages:"
echo "    smartsim-update pydantic"
echo "    smartsim-update loguru pyinstrument"
echo
echo "To update the SmartSim-CSC stack itself, pin a new SMARTSIM_CSC_REF"
echo "and rerun this installer for the current architecture."
echo
if [ "$BUILD_OPENFOAM" = "yes" ]; then
    echo "To use the OpenFOAM integration in a new shell:"
    echo "    module --force purge"
    echo "    source \"$BASE_SCRATCH/Python4SmartSim.sh\""
    echo "    module load gcc/15.2.0 openmpi/5.0.10 openfoam/2412"
    echo "    export FOAM_USER_DIR=\"$OPENFOAM_USER_DIR\""
    echo "Do not source Python4SmartSim.sh again after loading OpenFOAM."
    echo
fi
echo "The environment uses Python 3.12 and includes:"
echo "    JAX + Equinox"
echo "    TensorFlow 2.18.1"
echo "    PyTorch 2.7.1"
echo "    ONNX + ONNX Runtime"
if [ "$INSTALL_PYSR" = "yes" ]; then
    echo "    PySR + JuliaCall"
else
    echo "    PySR + JuliaCall: SKIPPED (INSTALL_PYSR=no for $ENV_ARCH)"
fi
echo "    SmartSim-CSC unified stack"
if [ "$BUILD_OPENFOAM" = "yes" ]; then
    echo "    OpenFOAM.com v2412 SmartRedis integration (x86_64)"
else
    echo "    OpenFOAM.com v2412 integration: SKIPPED"
fi
echo "    RedisAI ONNX Runtime + JAX backends selected by stack.toml"
echo
echo "No SmartSim, SmartRedis, or OpenFOAM source patching was applied."
echo
echo "Run the script again on the other architecture when both x64 and"
echo "arm64 Roihu environments are required. Each architecture asks for"
echo "its own PySR/Julia choice, and it is recorded in:"
echo "    $PYTHON_ROOT/install-options-$ENV_ARCH.sh"
echo "Use the same identity values."
