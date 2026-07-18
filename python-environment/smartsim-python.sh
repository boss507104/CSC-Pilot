#!/bin/bash
# smartsim-python.sh
# Interactive installer for the SmartSim Tykky environment + native SmartRedis
# library + smartsim-update command + Jupyter kernel registration
# (Sections 0, 1, 3, 5, 6, 7, 8, 11 only).
# Intended location: /scratch/$CSC_PROJECT/$PROJECT_USER_DIR/smartsim-python.sh
# Intended to be run directly on the LOGIN NODE, per explicit request.
#
# This script performs INSTALLATION ONLY. It intentionally skips:
#   - Environment validation          (guide Section 9)
#   - Dependency file workflow notes  (guide Section 10, doc only)
#   - Rebuild / troubleshooting       (guide Sections 12-13)
#   - Deployment track notes          (guide Section 14, doc only)
# Those remain manual steps from the full guide if/when you need them.
#
# Run this once per architecture (once on a CPU/login node for x64, once
# on a Roihu GPU node for arm64) to match the guide's per-architecture
# build + native library + kernel registration flow.
#
# ARCHITECTURE SPLIT (as of this version):
#   The Redis + RedisAI Orchestrator (`smart build`) is built ONLY on x64.
#   RedisAI's dependency-fetch step for `dlpack.h` does not complete on
#   Linux ARM64 in this pipeline (confirmed: outbound network access from
#   the GPU node works fine — `curl`/`git ls-remote` both succeed — but no
#   dlpack directory is ever created under RedisAI's build tree on arm64,
#   and `make` then fails with "dlpack/dlpack.h: No such file or
#   directory"). This looks like an architecture-detection gap inside
#   RedisAI's own fetch script, separate from SmartSim's platform.py.
#
#   Rather than patch RedisAI's fetch logic, arm64 environments are built
#   as SmartRedis-client + JAX workers only: `smart build` is skipped
#   entirely there, and the environment is expected to connect to a
#   remote x64 Orchestrator over the network. The SmartSim platform.py /
#   linux-arm64-cpu.json patch used in earlier versions of this script is
#   REMOVED — its only purpose was making `smart build` accept arm64,
#   which no longer runs on arm64 at all.
#
# Build flags note (x64 only, history kept for reference):
#   Attempt 1: --skip-backends --skip-python-packages   -> not real flags, rejected
#   Attempt 2: --no_tf --no_pt                           -> not real flags, rejected
#   Attempt 3 (current): --skip-torch --skip-tensorflow --skip-onnx
#   This matches the flags documented for the CLI actually shipped with
#   smartsim==0.8.0 as installed here. If this ever errors again with
#   "unrecognized arguments", run `smart build --help` inside the build
#   environment to get the ground-truth flag list for whatever version
#   actually got installed, rather than trusting cached documentation.

set -e

echo "=================================================================="
echo " SmartSim Environment Installer (login node, installation-only)"
echo "=================================================================="
echo
echo "WARNING: This build will run directly on the LOGIN NODE."
echo "The full guide recommends compute nodes (srun/sinteractive) for"
echo "BOTH the Tykky build AND native SmartRedis compilation, to avoid"
echo "resource contention on shared login nodes. Proceeding anyway,"
echo "per your request."
echo

# ------------------------------------------------------------------
# Helper: prompt twice, require matching values, loop until they agree
# ------------------------------------------------------------------
prompt_confirmed() {
    local prompt_text="$1"
    local __resultvar="$2"
    local first second

    while true; do
        read -p "Type ${prompt_text}: " first
        read -p "Type ${prompt_text} (verification): " second

        if [ -z "$first" ]; then
            echo "Value cannot be empty. Try again."
            echo
            continue
        fi

        if [ "$first" != "$second" ]; then
            echo "Values did not match. Try again."
            echo
            continue
        fi

        printf -v "$__resultvar" '%s' "$first"
        break
    done
}

# ------------------------------------------------------------------
# Helper: architecture prompt with matching + normalisation
# ------------------------------------------------------------------
prompt_architecture() {
    local first second norm_first norm_second

    while true; do
        read -p "Type node or architecture (cpu / gpu / x64 / arm64): " first
        read -p "Type node or architecture (verification): " second

        norm_first="$(echo "$first"  | tr '[:upper:]' '[:lower:]' | xargs)"
        norm_second="$(echo "$second" | tr '[:upper:]' '[:lower:]' | xargs)"

        if [ "$norm_first" != "$norm_second" ]; then
            echo "Values did not match. Try again."
            echo
            continue
        fi

        case "$norm_first" in
            cpu|x64)
                ENV_ARCH="x64"
                return
                ;;
            gpu|arm64)
                ENV_ARCH="arm64"
                return
                ;;
            *)
                echo "Invalid choice: '$first'. Enter one of: cpu, gpu, x64, arm64."
                echo
                ;;
        esac
    done
}

# ------------------------------------------------------------------
# Helper: target system prompt (drives GCC/CMake module selection
# for the native SmartRedis build — see guide Section 6)
# ------------------------------------------------------------------
prompt_system() {
    local first second norm_first norm_second

    while true; do
        read -p "Type target system (roihu / mahti / puhti): " first
        read -p "Type target system (verification): " second

        norm_first="$(echo "$first"  | tr '[:upper:]' '[:lower:]' | xargs)"
        norm_second="$(echo "$second" | tr '[:upper:]' '[:lower:]' | xargs)"

        if [ "$norm_first" != "$norm_second" ]; then
            echo "Values did not match. Try again."
            echo
            continue
        fi

        case "$norm_first" in
            roihu|mahti|puhti)
                TARGET_SYSTEM="$norm_first"
                return
                ;;
            *)
                echo "Invalid choice: '$first'. Enter one of: roihu, mahti, puhti."
                echo
                ;;
        esac
    done
}

# ------------------------------------------------------------------
# Step 1: Collect identity values (mirrors guide Section 0)
# ------------------------------------------------------------------
echo "--- Project identity ---"
prompt_confirmed "project number" RAW_PROJECT

if [[ "$RAW_PROJECT" == project_* ]]; then
    CSC_PROJECT="$RAW_PROJECT"
else
    CSC_PROJECT="project_${RAW_PROJECT}"
fi

echo
prompt_confirmed "project user directory name" PROJECT_USER_DIR

echo
prompt_confirmed "environment nickname" ENV_NICKNAME

echo
echo "--- Target architecture ---"
prompt_architecture

echo
if [ "$ENV_ARCH" = "arm64" ]; then
    echo "NOTE: arm64 selected. This environment will be a SmartRedis-client +"
    echo "JAX worker only — 'smart build' (Redis + RedisAI Orchestrator) will"
    echo "NOT be run on this architecture. See the header comment in this"
    echo "script for why. You'll need a separately built x64 environment"
    echo "running the Orchestrator for this to connect to."
    echo
fi

echo "--- Target system (for GCC / CMake module selection) ---"
prompt_system

echo
echo "--- Summary ---"
echo "CSC_PROJECT       = $CSC_PROJECT"
echo "PROJECT_USER_DIR  = $PROJECT_USER_DIR"
echo "ENV_NICKNAME      = $ENV_NICKNAME"
echo "ENV_ARCH          = $ENV_ARCH"
echo "TARGET_SYSTEM     = $TARGET_SYSTEM"
if [ "$ENV_ARCH" = "arm64" ]; then
    echo "Orchestrator build (smart build) = SKIPPED (client-only environment)"
else
    echo "Orchestrator build (smart build) = YES (Redis + RedisAI module)"
fi
echo

# ------------------------------------------------------------------
# Step 2: Select GCC / CMake modules for the native SmartRedis build
# (per guide Section 6's Roihu / Mahti examples)
# ------------------------------------------------------------------
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
        echo "The guide does not list default GCC/CMake modules for Puhti."
        echo "Enter them manually (check 'module avail gcc' / 'module avail cmake' first)."
        echo
        prompt_confirmed "GCC module (e.g. gcc/13.1.0)" GCC_MODULE
        echo
        prompt_confirmed "CMake module (e.g. cmake/3.28.6)" CMAKE_MODULE
        LOAD_GIT_MODULE="yes"
        ;;
esac

echo
echo "Selected GCC module:   $GCC_MODULE"
echo "Selected CMake module: $CMAKE_MODULE"
echo

# ------------------------------------------------------------------
# Step 3: Architecture sanity check (login node reality check)
# ------------------------------------------------------------------
HOST_ARCH="$(uname -m)"

if [ "$ENV_ARCH" = "arm64" ] && [ "$HOST_ARCH" != "aarch64" ]; then
    echo "WARNING: You selected arm64/gpu, but this login node reports"
    echo "         architecture '$HOST_ARCH' (expected aarch64)."
    echo
    echo "The Tykky container AND the native SmartRedis library are"
    echo "architecture-specific. Building either here will very likely"
    echo "produce artefacts that do NOT run correctly on Roihu GPU nodes."
    echo
    read -p "Continue anyway? [y/N]: " CONFIRM_ARCH
    case "$CONFIRM_ARCH" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
    echo
fi

if [ "$ENV_ARCH" = "x64" ] && [ "$HOST_ARCH" = "aarch64" ]; then
    echo "WARNING: You selected cpu/x64, but this host reports architecture"
    echo "         '$HOST_ARCH' (aarch64/ARM64), not x86_64."
    echo
    echo "Both the Tykky container and the native SmartRedis library are"
    echo "built for whatever host they actually run on — building here"
    echo "would produce ARM64 artefacts mislabelled as x64, which will"
    echo "fail confusingly on a real x86_64 node later."
    echo
    read -p "Continue anyway? [y/N]: " CONFIRM_ARCH2
    case "$CONFIRM_ARCH2" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
    echo
fi

read -p "Proceed with installation using the values above? [y/N]: " CONFIRM_ALL
case "$CONFIRM_ALL" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
esac
echo

# ------------------------------------------------------------------
# Step 4: Write the shared identity file (guide Section 0)
# ------------------------------------------------------------------
echo "[1/10] Writing identity file..."

mkdir -p "$HOME/.config/csc-hpc"

if [ -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "      Identity file already exists — overwriting with the values"
    echo "      entered above. If you already set this up for the ML stack,"
    echo "      or for the OTHER architecture of this SmartSim stack, make"
    echo "      sure CSC_PROJECT/PROJECT_USER_DIR/ENV_NICKNAME still match."
fi

cat <<EOF > "$HOME/.config/csc-hpc/identity.sh"
# --- USER CONFIGURATION START ---
export CSC_PROJECT="$CSC_PROJECT"
export PROJECT_USER_DIR="$PROJECT_USER_DIR"
export ENV_NICKNAME="$ENV_NICKNAME"
# --- USER CONFIGURATION END ---
EOF

chmod 600 "$HOME/.config/csc-hpc/identity.sh"
echo "      -> $HOME/.config/csc-hpc/identity.sh"
echo

# ------------------------------------------------------------------
# Step 5: Global Configuration (guide Section 1.1 / 1.2)
# ------------------------------------------------------------------
echo "[2/10] Setting up paths..."

source "$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.11-$ENV_ARCH"
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
# Step 6: Create configuration files (guide Section 3)
# ------------------------------------------------------------------
echo "[3/10] Creating configuration files..."
cd "$PYTHON_ROOT"

cat <<'EOF' > "$PYTHON_ROOT/base4SmartSim.yml"
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
EOF

cat <<'EOF' > "$PYTHON_ROOT/requirements.in"
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

# --- Mathematical Tools ---
numba
pint
ruptures
sympy
tensorly

# --- Custom Utilities ---
DataGraph @ git+https://github.com/boss507104/DataGraph.git#subdirectory=DataGraph
eqx_io @ git+https://github.com/boss507104/CSC-HPC-Guide.git#subdirectory=utilities/eqx4smartredis

# --- Config, Logging & Profiling ---
pydantic
loguru
pyinstrument

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
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

python -m pip install --no-cache-dir uv

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# --- Patched SmartRedis Python client (both architectures) ---
rm -rf "$CW_BUILD_TMPDIR/SmartRedis"
git clone \
    https://github.com/boss507104/SmartRedis.git \
    "$CW_BUILD_TMPDIR/SmartRedis"
cd "$CW_BUILD_TMPDIR/SmartRedis"

grep -q '#include <cstdint>' src/cpp/tensorpack.cpp || \
    sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp

OLD_CFLAGS="${CFLAGS-}"; OLD_CXXFLAGS="${CXXFLAGS-}"
OLD_CPPFLAGS="${CPPFLAGS-}"; OLD_LDFLAGS="${LDFLAGS-}"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

python -m pip install --no-cache-dir .

export CFLAGS="$OLD_CFLAGS" CXXFLAGS="$OLD_CXXFLAGS"
export CPPFLAGS="$OLD_CPPFLAGS" LDFLAGS="$OLD_LDFLAGS"

# --- SmartSim, installed only after SmartRedis (both architectures) ---
uv pip install --link-mode=copy smartsim==0.8.0

# --- Build the Orchestrator (Redis + RedisAI) — x64 ONLY ---
# SmartSim's RedisAI build chain does not fetch its dlpack dependency
# correctly on Linux ARM64 in this pipeline (fatal error:
# dlpack/dlpack.h: No such file or directory during `make`). Network
# access from the arm64 build node was confirmed working separately, so
# this is treated as an architecture-detection gap in RedisAI's own
# dependency-fetch script, not a network issue. Rather than patch that,
# x64 builds the full Orchestrator; arm64 stays client-only and connects
# to a remote x64 Orchestrator over the network.
if [ "$ENV_ARCH" = "x64" ]; then
    export USE_SYSTEMD=no

    env CFLAGS="-Wno-incompatible-pointer-types" \
        CXXFLAGS="-Wno-incompatible-pointer-types" \
        USE_SYSTEMD=no \
        smart clobber

    env CFLAGS="-Wno-incompatible-pointer-types" \
        CXXFLAGS="-Wno-incompatible-pointer-types" \
        USE_SYSTEMD=no \
        smart build \
            --device cpu \
            --skip-torch \
            --skip-tensorflow \
            --skip-onnx
else
    echo "Skipping smart build (Orchestrator/RedisAI) on $ENV_ARCH."
    echo "This environment is a SmartRedis client + JAX worker only,"
    echo "and connects to an x64 SmartSim Orchestrator over the network."
fi

# Restore packages potentially disturbed by the build above
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

# Record installed versions; SmartSim/SmartRedis are installed separately
# and intentionally excluded from this replay file.
python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

rm -rf "$CW_BUILD_TMPDIR/SmartRedis"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"

echo "      -> base4SmartSim.yml, requirements.in, extra4SmartSim.sh"
echo

# ------------------------------------------------------------------
# Step 7: Build the Tykky environment (guide Section 5, on login node)
# ------------------------------------------------------------------
echo "[4/10] Building the Tykky environment on the login node..."
if [ "$ENV_ARCH" = "x64" ]; then
    echo "      (this installs a large scientific stack + SmartRedis + SmartSim"
    echo "       + the Redis/RedisAI Orchestrator, and can take a long time)"
else
    echo "      (this installs a large scientific stack + SmartRedis + SmartSim;"
    echo "       the Orchestrator build is skipped on $ENV_ARCH — see header note)"
fi
echo

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
echo "      Build finished. Checking output..."
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt" 2>/dev/null || true
echo

# ------------------------------------------------------------------
# Step 8: Build the native SmartRedis library (guide Section 6)
# ------------------------------------------------------------------
# This is a separate CMake build unrelated to smart build/RedisAI, and
# is run on BOTH architectures — it is unaffected by the ARM64 issue.
echo "[5/10] Loading compiler modules for the native SmartRedis build..."

module purge
module load "$GCC_MODULE"
module load "$CMAKE_MODULE"
if [ "$LOAD_GIT_MODULE" = "yes" ]; then
    module load git
fi

echo "      Loaded: $GCC_MODULE, $CMAKE_MODULE"
echo

echo "[6/10] Building the native SmartRedis library..."

cd "$BASE_SCRATCH"

echo "      This will remove only this native SmartRedis directory:"
echo "      $SMARTREDIS_DIR"
rm -rf "$SMARTREDIS_DIR"

git clone \
    https://github.com/boss507104/SmartRedis.git \
    "$SMARTREDIS_DIR"

cd "$SMARTREDIS_DIR"

grep -q '#include <cstdint>' src/cpp/tensorpack.cpp || \
    sed -i '30i #include <cstdint>' src/cpp/tensorpack.cpp

rm -rf build install

env \
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran

echo
echo "[7/10] Verifying the native SmartRedis library..."

find "$SMARTREDIS_DIR/install" -maxdepth 3 -type f | sort

# Detect whether this system produced lib64 or lib, so the loader
# and smartsim-update helper text below don't hardcode the wrong one.
if [ -d "$SMARTREDIS_DIR/install/lib64" ]; then
    LIB_DIR="lib64"
else
    LIB_DIR="lib"
fi
echo "      Detected library directory: install/$LIB_DIR"

ls -la "$SMARTREDIS_DIR/install/$LIB_DIR"

if [ -f "$SMARTREDIS_DIR/install/$LIB_DIR/libsmartredis-fortran.so" ]; then
    echo "      SmartRedis Fortran library installed successfully."
else
    echo "      WARNING: libsmartredis-fortran.so not found under"
    echo "      $SMARTREDIS_DIR/install/$LIB_DIR — check the build log above."
fi

ldd "$SMARTREDIS_DIR/install/$LIB_DIR/libsmartredis-fortran.so" 2>/dev/null || true
echo

# ------------------------------------------------------------------
# Step 9: Create the loader (guide Section 7), with the detected
# lib/lib64 directory and the GCC module baked in.
# ------------------------------------------------------------------
echo "[8/10] Creating loader Python4SmartSim.sh and update tooling..."

cat <<EOF > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash

if [ ! -f "\$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "Identity file not found: \$HOME/.config/csc-hpc/identity.sh"
    echo "Run smartsim-python.sh (or Section 0 of the SmartSim guide) first."
    return 1
fi

source "\$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/\$CSC_PROJECT/\$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="\$BASE_SCRATCH/Python"
export PYTHON_ROOT="\$PYTHON_BASE/PythonSmartSim"

case "\$(uname -m)" in
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
        echo "Unsupported architecture: \$(uname -m)"
        return 1
        ;;
esac

export ENV_PREFIX="\$PYTHON_ROOT/envs/\$ENV_NICKNAME-3.11-\$ENV_ARCH"
export SMARTREDIS_DIR="\$BASE_SCRATCH/SmartRedis-\$ENV_ARCH"

# GCC module used to build SmartRedis on $TARGET_SYSTEM
module load $GCC_MODULE

export PATH="\$ENV_PREFIX/bin:\$PATH"

# Detected at build time on $TARGET_SYSTEM: install/$LIB_DIR
export LD_LIBRARY_PATH="\$SMARTREDIS_DIR/install/$LIB_DIR:\${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="\$SMARTREDIS_DIR/install:\${CMAKE_PREFIX_PATH:-}"

export SMARTSIM_DB_FILE_PARSE_TRIALS=600

export JUPYTER_KERNEL_NAME="\$ENV_NICKNAME-smartsim-\$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.11 (\$ENV_NICKNAME SmartSim \$KERNEL_ARCH)"
export XDG_DATA_HOME="\${XDG_DATA_HOME:-\$HOME/.local/share/\$KERNEL_ARCH}"
export JUPYTER_KERNEL_DIR="\$XDG_DATA_HOME/jupyter/kernels/\$JUPYTER_KERNEL_NAME"

echo "ENV_ARCH=\$ENV_ARCH"
echo "PYTHON_ROOT=\$PYTHON_ROOT"
echo "ENV_PREFIX=\$ENV_PREFIX"
echo "SMARTREDIS_DIR=\$SMARTREDIS_DIR"
echo "JAX_PLATFORMS=\$JAX_PLATFORMS"

if [ "\$ENV_ARCH" = "arm64" ]; then
    echo "NOTE: This is a client-only SmartSim environment (no local"
    echo "Orchestrator). Connect to a remote x64 Orchestrator via"
    echo "smartredis.Client(address=..., cluster=False)."
fi
EOF

chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"
echo "      -> $BASE_SCRATCH/Python4SmartSim.sh"

# ------------------------------------------------------------------
# Step 9b: Create update4SmartSim.sh (guide Section 11, post-install script)
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
export UV_CONCURRENT_DOWNLOADS=4

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

python -m pip install --no-cache-dir uv

# Install the complete constrained dependency set
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# Explicitly upgrade packages requested through smartsim-update
UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ -s "$UPDATE_REQUEST" ]; then
    mapfile -t UPDATE_PACKAGES < "$UPDATE_REQUEST"

    uv pip install \
        --link-mode=copy \
        --upgrade \
        "${UPDATE_PACKAGES[@]}"
fi

# Install the patched SmartRedis Python client (both architectures)
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

# Install SmartSim only after SmartRedis (both architectures)
uv pip install \
    --link-mode=copy \
    smartsim==0.8.0

# Rebuild the Orchestrator (Redis + RedisAI) — x64 ONLY.
# See extra4SmartSim.sh for why this is skipped on arm64.
if [ "$ENV_ARCH" = "x64" ]; then
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
            --skip-tensorflow \
            --skip-onnx
else
    echo "Skipping smart build (Orchestrator/RedisAI) on $ENV_ARCH."
fi

# Restore the constrained dependency set after the build step above
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

# Record the installed package state
python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

rm -f "$UPDATE_REQUEST"
rm -rf "$CW_BUILD_TMPDIR/SmartRedis"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF

chmod +x "$PYTHON_ROOT/update4SmartSim.sh"
echo "      -> $PYTHON_ROOT/update4SmartSim.sh"

# ------------------------------------------------------------------
# Step 9c: Create the smartsim-update command (guide Section 11)
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
    echo "Run smartsim-python.sh (or Section 0 of the SmartSim guide) first."
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

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.11-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"
export UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "Environment not found:"
    echo "$ENV_PREFIX"
    exit 1
fi

if [ ! -f "$PYTHON_ROOT/requirements.in" ]; then
    echo "requirements.in not found:"
    echo "$PYTHON_ROOT/requirements.in"
    exit 1
fi

for package in "$@"; do
    package_name="$(
        printf '%s\n' "$package" |
        sed -E 's/\[.*//; s/[<>=!~].*//'
    )"

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

echo
echo "Architecture: $ENV_ARCH"
echo "Environment:  $ENV_PREFIX"
echo "Packages:     $*"
echo

conda-containerize update \
    --post-install "$PYTHON_ROOT/update4SmartSim.sh" \
    "$ENV_PREFIX"

echo
echo "Update completed."
echo "Recorded packages:"
echo "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
EOF

chmod +x "$HOME/bin/smartsim-update"
echo "      -> $HOME/bin/smartsim-update"

grep -qxF 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" || \
    echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"

echo "      -> Added \$HOME/bin to PATH in ~/.bashrc (if not already present)"
echo

# ------------------------------------------------------------------
# Step 10: Register the Jupyter kernel (guide Section 8)
# ------------------------------------------------------------------
echo "[9/10] Registering the Jupyter kernel for this architecture..."

source "$BASE_SCRATCH/Python4SmartSim.sh"

mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": ["$ENV_PREFIX/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": { "debugger": true }
}
EOF

echo "      -> $JUPYTER_KERNEL_DIR/kernel.json"
echo "      Registered kernel: $JUPYTER_KERNEL_NAME"
echo

if command -v jupyter >/dev/null 2>&1; then
    jupyter kernelspec list 2>/dev/null || true
else
    echo "      (jupyter CLI not on PATH in this shell — kernel.json was still"
    echo "       written correctly; 'jupyter kernelspec list' will show it once"
    echo "       run from inside the loaded environment.)"
fi
echo

echo "[10/10] Installation complete."
echo "=================================================================="
echo " Installation complete."
echo "=================================================================="
echo
echo "Load the environment with:"
echo "    source \"$BASE_SCRATCH/Python4SmartSim.sh\""
echo
echo "Reload your shell (or open a new one) so smartsim-update is on"
echo "PATH, then update/add packages with, e.g.:"
echo "    smartsim-update pydantic"
echo "    smartsim-update \"tensorflow>=2.20\""
echo
echo "In VS Code, after registering, reload the remote window:"
echo "    Command Palette -> Developer: Reload Window"
echo
if [ "$ENV_ARCH" = "x64" ]; then
    echo "Note: this x64 build compiles Redis + the RedisAI module (required by"
    echo "the SmartSim Orchestrator) using --skip-torch --skip-tensorflow"
    echo "--skip-onnx, excluding all three ML-execution backends. Missing"
    echo "PyTorch/TensorFlow/ONNXRuntime in 'smart validate' output is"
    echo "expected; RedisAI itself IS present."
else
    echo "Note: this arm64 build is SmartRedis-client + JAX-worker only."
    echo "'smart build' was NOT run here (RedisAI's dlpack fetch fails on"
    echo "Linux ARM64 in this pipeline). There is no local Orchestrator —"
    echo "connect this environment to a separately built x64 environment"
    echo "running the Orchestrator, e.g.:"
    echo "    from smartredis import Client"
    echo "    client = Client(address=\"X64_HOST:6379\", cluster=False)"
fi
echo
echo "If a future SmartSim patch renames the x64 build flags again, run"
echo "'smart build --help' inside the build environment to get the current"
echo "ground truth, rather than trusting this script."
echo
echo "If you're setting up BOTH architectures, run this script again on"
echo "the OTHER node type (CPU/login for x64, Roihu GPU for arm64) with"
echo "the SAME identity values."
echo
echo "Note: if this native library was built on a different node type than"
echo "you'll actually run solvers on (e.g. built here but used later on a"
echo "different Roihu partition), re-check install/lib vs install/lib64"
echo "and the GCC module — this loader currently assumes: $GCC_MODULE,"
echo "install/$LIB_DIR."
echo
echo "Skipped (not part of installation — see the full guide if needed):"
echo "  - Environment validation          (guide Section 9)"
echo "  - Dependency file workflow notes  (guide Section 10, doc only)"
echo "  - Rebuild / troubleshooting       (guide Sections 12-13)"
echo "  - Deployment track notes          (guide Section 14, doc only)"
