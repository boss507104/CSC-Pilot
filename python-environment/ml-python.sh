#!/bin/bash
# ml-python.sh
# Interactive installer for the ML Tykky environment + ml-update command +
# Jupyter kernel registration (Sections 0, 1, 3, 5, 6, 7, 10 only).
# Intended location: /scratch/$CSC_PROJECT/$PROJECT_USER_DIR/ml-python.sh
# Intended to be run directly on the LOGIN NODE, per explicit request.
#
# This script performs INSTALLATION ONLY. It intentionally skips:
#   - Environment validation         (guide Section 8)
#   - Dependency file workflow notes (guide Section 9, doc only)
#   - Rebuild / troubleshooting      (guide Sections 11-12)
# Those remain manual steps from the full guide if/when you need them.
#
# Run this once per architecture (once on a CPU/login node for x64, once
# on a Roihu GPU node for arm64) to match the guide's per-architecture
# build + kernel registration flow.

set -e

echo "=================================================================="
echo " ML Environment Installer (login node, installation-only)"
echo "=================================================================="
echo
echo "WARNING: This build will run directly on the LOGIN NODE."
echo "The full guide recommends compute nodes (srun/sinteractive) for"
echo "Tykky builds to avoid resource contention on shared login nodes."
echo "Proceeding anyway, per your request."
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
# Step 1: Collect identity values (mirrors guide Section 0)
# ------------------------------------------------------------------
echo "--- Project identity ---"
prompt_confirmed "project number" RAW_PROJECT

# Accept either "2015384" or "project_2015384"
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
echo "--- Summary ---"
echo "CSC_PROJECT       = $CSC_PROJECT"
echo "PROJECT_USER_DIR  = $PROJECT_USER_DIR"
echo "ENV_NICKNAME      = $ENV_NICKNAME"
echo "ENV_ARCH          = $ENV_ARCH"
echo

# ------------------------------------------------------------------
# Step 2: Architecture sanity check (login node reality check)
# ------------------------------------------------------------------
HOST_ARCH="$(uname -m)"

if [ "$ENV_ARCH" = "arm64" ] && [ "$HOST_ARCH" != "aarch64" ]; then
    echo "WARNING: You selected arm64/gpu, but this login node reports"
    echo "         architecture '$HOST_ARCH' (expected aarch64)."
    echo
    echo "Per the guide, Tykky containers are architecture-specific and"
    echo "ARM64 builds normally need to run ON a Roihu GPU node, not the"
    echo "login node. Building here will very likely produce a container"
    echo "that does NOT run correctly on Roihu GPU nodes."
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
    echo "Tykky builds native binaries for whatever host it actually runs"
    echo "on — building here would produce ARM64 binaries mislabelled as"
    echo "x64, which will fail confusingly on a real x86_64 node later."
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
# Step 3: Write the shared identity file (guide Section 0)
# ------------------------------------------------------------------
echo "[1/9] Writing identity file..."

mkdir -p "$HOME/.config/csc-hpc"

if [ -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "      Identity file already exists — overwriting with the values"
    echo "      entered above. If you already set this up for the SmartSim"
    echo "      stack, or for the OTHER architecture of this ML stack, make"
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
# Step 4: Global Configuration (guide Section 1.1 / 1.2)
# ------------------------------------------------------------------
echo "[2/9] Setting up paths..."

source "$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonML"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "      ENV_ARCH=$ENV_ARCH"
echo "      PYTHON_ROOT=$PYTHON_ROOT"
echo "      ENV_PREFIX=$ENV_PREFIX"
echo "      TMP_BUILD_DIR=$TMP_BUILD_DIR"
echo

# ------------------------------------------------------------------
# Step 5: Create configuration files (guide Section 3)
# ------------------------------------------------------------------
echo "[3/9] Creating configuration files..."
cd "$PYTHON_ROOT"

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
jaxopt
jaxtyping
lineax
optax
optimistix
sympy2jax

# --- ONNX conversion and Runtime
onnx
onnxruntime
jax2onnx
skl2onnx
tf2onnx

# --- Machine Learning ---
tensorflow
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

echo "      -> base4ML.yml, requirements.in, extra4ML.sh"
echo

# ------------------------------------------------------------------
# Step 6: Build the Tykky environment (guide Section 5, on login node)
# ------------------------------------------------------------------
echo "[4/9] Building the Tykky environment on the login node..."
echo "      (this can take a long time — installing a large scientific stack + Julia)"
echo

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

echo
echo "      Build finished. Checking output..."
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt" 2>/dev/null || true
ls -lh "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt" 2>/dev/null || true
echo

# ------------------------------------------------------------------
# Step 7: Create the loader (guide Section 6) so the env is usable
# ------------------------------------------------------------------
echo "[5/9] Creating loader Python4ML.sh..."

cat <<'EOF' > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

if [ ! -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "Identity file not found: $HOME/.config/csc-hpc/identity.sh"
    echo "Run ml-python.sh (or Section 0 of the ML Environment guide) first."
    return 1
fi

source "$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonML"

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
export JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
export JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

export JULIA_ENV_RUNTIME
export JULIA_DEPOT_RUNTIME

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

export JUPYTER_KERNEL_NAME="$ENV_NICKNAME-ml-$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.12 ($ENV_NICKNAME ML $KERNEL_ARCH)"
export XDG_DATA_HOME="$HOME/.local/share/$KERNEL_ARCH"
export JUPYTER_KERNEL_DIR="$XDG_DATA_HOME/jupyter/kernels/$JUPYTER_KERNEL_NAME"

echo "ENV_ARCH=$ENV_ARCH"
echo "PYTHON_ROOT=$PYTHON_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"
echo "PYTHON_JULIAPKG_PROJECT=$PYTHON_JULIAPKG_PROJECT"
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
EOF

chmod +x "$BASE_SCRATCH/Python4ML.sh"
echo "      -> $BASE_SCRATCH/Python4ML.sh"
echo

# ------------------------------------------------------------------
# Step 8: Create update4ML.sh + ml-update command (guide Section 10)
# ------------------------------------------------------------------
echo "[6/9] Creating update4ML.sh..."

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

# Install the complete direct dependency set
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.in"

# Explicitly upgrade the packages requested through ml-update
UPDATE_REQUEST="$PYTHON_ROOT/.ml-update-$ENV_ARCH.txt"

if [ -s "$UPDATE_REQUEST" ]; then
    mapfile -t UPDATE_PACKAGES < "$UPDATE_REQUEST"

    uv pip install \
        --upgrade \
        "${UPDATE_PACKAGES[@]}"
fi

# Keep the packaged Julia environment ready for PySR
python - <<'PY'
import juliapkg
import pysr

juliapkg.resolve()

print(f"PySR version:     {pysr.__version__}")
print(f"Julia executable: {juliapkg.executable()}")
print(f"Julia project:    {juliapkg.project()}")
PY

python - <<'PY'
import juliapkg
import subprocess

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

python -m pip freeze > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

rm -f "$UPDATE_REQUEST"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF

chmod +x "$PYTHON_ROOT/update4ML.sh"
echo "      -> $PYTHON_ROOT/update4ML.sh"
echo

echo "[7/9] Creating ml-update command..."

mkdir -p "$HOME/bin"

cat <<'EOF' > "$HOME/bin/ml-update"
#!/bin/bash -l
set -e

if [ "$#" -eq 0 ]; then
    echo "Usage: ml-update <package> [package ...]"
    exit 1
fi

if [ ! -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "Identity file not found: $HOME/.config/csc-hpc/identity.sh"
    echo "Run ml-python.sh (or Section 0 of the ML Environment guide) first."
    exit 1
fi

source "$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonML"

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
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_$ENV_ARCH"
export UPDATE_REQUEST="$PYTHON_ROOT/.ml-update-$ENV_ARCH.txt"

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

printf '%s\n' "$@" > "$UPDATE_REQUEST"

for package in "$@"; do
    PACKAGE_NAME="$(printf '%s\n' "$package" | sed -E 's/\[.*//; s/[<>=!~].*//')"

    if grep -Eq "^${PACKAGE_NAME}(\[.*\])?([<>=!~].*)?$" "$PYTHON_ROOT/requirements.in"; then
        sed -i -E \
            "s|^${PACKAGE_NAME}(\[.*\])?([<>=!~].*)?$|${package}|" \
            "$PYTHON_ROOT/requirements.in"

        echo "Updated requirement: $package"
    else
        echo "$package" >> "$PYTHON_ROOT/requirements.in"
        echo "Added requirement: $package"
    fi
done

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
    --post-install "$PYTHON_ROOT/update4ML.sh" \
    "$ENV_PREFIX"

echo
echo "Update completed."
echo "Recorded packages:"
echo "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
EOF

chmod +x "$HOME/bin/ml-update"
echo "      -> $HOME/bin/ml-update"

grep -qxF 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" || \
    echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"

echo "      -> Added \$HOME/bin to PATH in ~/.bashrc (if not already present)"
echo

# ------------------------------------------------------------------
# Step 9: Register the Jupyter kernel (guide Section 7)
# ------------------------------------------------------------------
echo "[8/9] Registering the Jupyter kernel for this architecture..."

# Source the loader we just wrote — this is what derives
# JUPYTER_KERNEL_DIR / JUPYTER_KERNEL_NAME / JUPYTER_KERNEL_DISPLAY /
# JAX_PLATFORMS / PYTHON_JULIAPKG_PROJECT / JULIA_DEPOT_PATH for the
# architecture we just built on THIS node.
source "$BASE_SCRATCH/Python4ML.sh"

mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": [
    "$ENV_PREFIX/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "env": {
    "JAX_PLATFORMS": "$JAX_PLATFORMS",
    "PYTHON_JULIAPKG_PROJECT": "$PYTHON_JULIAPKG_PROJECT",
    "JULIA_DEPOT_PATH": "$JULIA_DEPOT_PATH",
    "PYTHON_JULIAPKG_OFFLINE": "yes",
    "PYTHON_JULIACALL_THREADS": "auto"
  }
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

echo "=================================================================="
echo " Installation complete."
echo "=================================================================="
echo
echo "Load the environment with:"
echo "    source \"$BASE_SCRATCH/Python4ML.sh\""
echo
echo "Reload your shell (or open a new one) so ml-update is on PATH,"
echo "then update/add packages with, e.g.:"
echo "    ml-update tensorflow"
echo "    ml-update \"tensorflow>=2.20\""
echo
echo "In VS Code, after registering, reload the remote window:"
echo "    Command Palette -> Developer: Reload Window"
echo
echo "If you're setting up BOTH architectures, run this script again on"
echo "the OTHER node type (CPU/login for x64, Roihu GPU for arm64) with"
echo "the SAME identity values, to register that architecture's kernel too."
echo
echo "Skipped (not part of installation — see the full guide if needed):"
echo "  - Environment validation         (guide Section 8)"
echo "  - Dependency file workflow notes (guide Section 9, doc only)"
echo "  - Rebuild / troubleshooting      (guide Sections 11-12)"
