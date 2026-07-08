# VS Code Tunnel on a Slurm Interactive Node on CSC Roihu

This guide covers:

1. Installing the VS Code CLI
2. Allocating a Slurm interactive CPU or GPU node
3. Starting the VS Code Tunnel on the allocated node
4. Connecting from local VS Code
5. Closing the tunnel and releasing the allocation

This guide assumes that:

1. The `csc-ssh-keys` command has already been configured.
2. The `roihu-cpu` and `roihu-gpu` SSH hosts have already been configured.
3. `ssh roihu-cpu` and `ssh roihu-gpu` connect successfully.

> **Placeholder values:** `Harry` is a placeholder username inspired by Harry Potter. Replace `Harry` with your actual CSC username and `project_xxxxxxxx` with your actual CSC project number.

The VS Code Tunnel runs on the allocated Slurm compute node. Local VS Code therefore connects directly to that compute node.

---

## 1. Install the VS Code CLI

On the local workstation, renew the CSC SSH certificate:

```bash
csc-ssh-keys
```

Connect to Roihu:

```bash
ssh roihu-cpu
```

Create the installation directory:

```bash
mkdir -p ~/bin
cd ~/bin
```

Download the stable VS Code CLI:

```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
    --output vscode_cli.tar.gz
```

Extract the archive:

```bash
tar -xf vscode_cli.tar.gz
```

Verify the installation:

```bash
~/bin/code --version
```

> This installation only needs to be completed once. If `~/bin` is shared between Roihu login environments, the same installation can be used from both `roihu-cpu` and `roihu-gpu`.

---

## 2. Allocate an Interactive Node

Use the CPU or GPU login node depending on the resource type.

### 2.1 Allocate an Interactive CPU Node

From the Roihu CPU login node:

```bash
srun --account=project_xxxxxxxx \
    --partition=interactive \
    --cpus-per-task=32 \
    --mem=62G \
    --time=09:00:00 \
    --pty bash
```

Wait until Slurm grants the allocation.

Verify that the shell has moved to a compute node:

```bash
hostname
```

### 2.2 Allocate an Interactive GPU Node

From the Roihu GPU login node:

```bash
srun --account=project_xxxxxxxx \
    --partition=gpuinteractive \
    --gres=gpu:gh200:1 \
    --time=12:00:00 \
    --pty bash
```

Wait until Slurm grants the allocation.

Verify that the shell has moved to a compute node:

```bash
hostname
```

Verify that a GPU is visible:

```bash
nvidia-smi
```

> The `gpuinteractive` partition currently provides full GPUs until GPU slices are fully configured.

---

## 3. Start the VS Code Tunnel

From the allocated compute node:

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

During the first run:

1. Select **GitHub Account**.
2. Open the following page on the local workstation:

   ```
   https://github.com/login/device
   ```

3. Sign in with the same GitHub account used in local VS Code.
4. Enter the temporary device code shown in the Roihu terminal.
5. Approve access for Visual Studio Code.
6. Use one of the following tunnel names when prompted:

   ```
   roihu-cpu-interactive
   ```

   or:

   ```
   roihu-gpu-interactive
   ```

> Leave the tunnel process and SSH terminal running while using VS Code.

---

## 4. Connect from Local VS Code

On the local workstation:

1. Open VS Code.
2. Sign in with the same GitHub account.
3. Open **Remote Explorer**.
4. Select **Tunnels**.
5. Select:

   ```
   roihu-cpu-interactive
   ```

   or:

   ```
   roihu-gpu-interactive
   ```

Alternatively, open the Command Palette with **Command-Shift-P** and run:

```
Remote Tunnels: Connect to Tunnel
```

VS Code now connects directly to the allocated compute node.

Open the project directory through:

**File → Open Folder**

For example:

```
/scratch/project_xxxxxxxx/Harry
```

---

## 5. Close the Tunnel and Release the Node

Close the remote VS Code window.

Return to the SSH terminal running the tunnel and press:

```
Ctrl-C
```

Exit the interactive compute node:

```bash
exit
```

Then disconnect from Roihu:

```bash
exit
```

---

## 6. Optional: Shell Function Shortcuts

To simplify allocation and tunnel startup into a single command, create helper functions on Roihu.

Create the directory for bash includes:

```bash
mkdir -p ~/.bashrc.d
```

### 6.1 CPU Launcher Script

Create the CPU launcher script on `roihu-cpu`:

```bash
cat > ~/.bashrc.d/vscode-interactive-cpu.sh << 'EOF'
# Slurm CPU interactive allocation + VS Code tunnel launcher
vscode-interactive-cpu() {
    srun --account=project_xxxxxxxx \
        --partition=interactive \
        --cpus-per-task=32 \
        --mem=62G \
        --time=09:00:00 \
        --pty ~/bin/code tunnel --accept-server-license-terms
}
EOF
```

Verify the script contents:

```bash
cat ~/.bashrc.d/vscode-interactive-cpu.sh
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

Confirm the function is available:

```bash
type vscode-interactive-cpu
```

Once configured, allocate the CPU node and start the tunnel in one step:

```bash
vscode-interactive-cpu
```

### 6.2 GPU Launcher Script

Create the GPU launcher script on `roihu-gpu`:

```bash
cat > ~/.bashrc.d/vscode-interactive-gpu.sh << 'EOF'
# Slurm GPU interactive allocation + VS Code tunnel launcher
vscode-interactive-gpu() {
    srun --account=project_xxxxxxxx \
        --partition=gpuinteractive \
        --gres=gpu:gh200:1 \
        --time=12:00:00 \
        --pty ~/bin/code tunnel --accept-server-license-terms
}
EOF
```

Verify the script contents:

```bash
cat ~/.bashrc.d/vscode-interactive-gpu.sh
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

Confirm the function is available:

```bash
type vscode-interactive-gpu
```

Once configured, allocate the GPU node and start the tunnel in one step:

```bash
vscode-interactive-gpu
```

> Ensure `~/.bashrc` sources files from `~/.bashrc.d/`. If it does not, add a snippet to `~/.bashrc` that loops over and sources scripts in that directory.

---

## 7. Routine Workflow

After the VS Code CLI has been installed, use the following commands for each session.

### 7.1 CPU Session

**On the local workstation:**

```bash
csc-ssh-keys
ssh roihu-cpu
```

**On the Roihu CPU login node, using the shortcut:**

```bash
vscode-interactive-cpu
```

**Or, using the manual method:**

```bash
srun --account=project_xxxxxxxx \
    --partition=interactive \
    --cpus-per-task=32 \
    --mem=62G \
    --time=09:00:00 \
    --pty bash
```

**On the allocated CPU compute node, if using the manual method:**

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

**In local VS Code:**

```
Remote Explorer → Tunnels → roihu-cpu-interactive
```

### 7.2 GPU Session

**On the local workstation:**

```bash
csc-ssh-keys
ssh roihu-gpu
```

**On the Roihu GPU login node, using the shortcut:**

```bash
vscode-interactive-gpu
```

**Or, using the manual method:**

```bash
srun --account=project_xxxxxxxx \
    --partition=gpuinteractive \
    --gres=gpu:gh200:1 \
    --time=12:00:00 \
    --pty bash
```

**On the allocated GPU compute node, if using the manual method:**

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

**In local VS Code:**

```
Remote Explorer → Tunnels → roihu-gpu-interactive
```

### 7.3 When finished

Stop the tunnel:

```
Ctrl-C
```

Then:

```bash
exit
exit
```

---

## 8. Notes

- Start the VS Code Tunnel only after entering the Slurm interactive node.
- Use `roihu-cpu` for CPU interactive sessions.
- Use `roihu-gpu` for GPU interactive sessions.
- Keep the original SSH terminal open while using VS Code.
- The tunnel stops when the Slurm allocation ends.
- The maximum CPU interactive allocation used in this guide is 32 CPU cores and 62 GiB of RAM.
- The maximum GPU interactive time used in this guide is 12 hours.
- The `gpuinteractive` partition currently provides full GPUs until GPU slices are fully configured.
- Use batch jobs for long-running production workloads.
- The `vscode-interactive-cpu` and `vscode-interactive-gpu` shell functions combine Slurm allocation and tunnel startup into a single command for convenience.
