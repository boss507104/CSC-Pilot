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

### 1.1 Install the x64 VS Code CLI for Roihu CPU

Connect to the Roihu CPU login node:

```bash
ssh roihu-cpu
```

Create the installation directory:

```bash
mkdir -p ~/bin/vscode-cli-x64
cd ~/bin/vscode-cli-x64
```

Download the stable x64 VS Code CLI:

```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
    --output vscode_cli.tar.gz
```

Extract the archive:

```bash
tar -xf vscode_cli.tar.gz
```

### 1.2 Install the ARM64 VS Code CLI for Roihu GPU

Connect to the Roihu GPU login node:

```bash
ssh roihu-gpu
```

Create the installation directory:

```bash
mkdir -p ~/bin/vscode-cli-arm64
cd ~/bin/vscode-cli-arm64
```

Download the stable ARM64 VS Code CLI:

```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64' \
    --output vscode_cli.tar.gz
```

Extract the archive:

```bash
tar -xf vscode_cli.tar.gz
```

> Roihu CPU nodes use the x64 VS Code CLI. Roihu GPU nodes use the ARM64 VS Code CLI.

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

This CPU interactive allocation requests:

- 32 CPU cores
- 62 GiB of memory
- 9 hours of runtime

Wait until Slurm grants the allocation.

Verify that the shell has moved to a compute node:

```bash
hostname
```

### 2.2 Allocate an Interactive GPU Node

From the Roihu GPU login node:

```bash
sinteractive \
    --account project_xxxxxxxx \
    --gpu \
    --cores 36 \
    --time 09:00:00
```

This GPU interactive allocation requests:

- 36 CPU cores
- 1 GPU
- fixed GPU-node memory
- 9 hours of runtime

Wait until Slurm grants the allocation.

Verify that the shell has moved to a compute node:

```bash
hostname
```

Verify that a GPU is visible:

```bash
nvidia-smi
```

> The `gpuinteractive` partition should be accessed through `sinteractive` from the `roihu-gpu` login node. The partition currently provides full GPUs until GPU slices are fully configured. The GPU interactive memory is fixed by the partition; `sinteractive` may show 110000 MB, while Slurm may override it to 217086 MB.

---

## 3. Start the VS Code Tunnel

From the allocated CPU compute node:

```bash
~/bin/vscode-cli-x64/code tunnel \
    --name roihu-cpu-int \
    --accept-server-license-terms
```

From the allocated GPU compute node:

```bash
~/bin/vscode-cli-arm64/code tunnel \
    --name roihu-gpu-int \
    --accept-server-license-terms
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
   roihu-cpu-int
   ```

   or:

   ```
   roihu-gpu-int
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

Create helper functions on Roihu to simplify the routine workflow.

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
        --pty ~/bin/vscode-cli-x64/code tunnel \
            --name roihu-cpu-int \
            --accept-server-license-terms
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
    sinteractive \
        --account project_xxxxxxxx \
        --gpu \
        --cores 36 \
        --time 09:00:00 \
        ~/bin/vscode-cli-arm64/code tunnel \
            --name roihu-gpu-int \
            --accept-server-license-terms
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
~/bin/vscode-cli-x64/code tunnel \
    --name roihu-cpu-int \
    --accept-server-license-terms
```

**In local VS Code:**

```
Remote Explorer → Tunnels → roihu-cpu-int
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
sinteractive \
    --account project_xxxxxxxx \
    --gpu \
    --cores 36 \
    --time 09:00:00
```

**On the allocated GPU compute node, if using the manual method:**

```bash
~/bin/vscode-cli-arm64/code tunnel \
    --name roihu-gpu-int \
    --accept-server-license-terms
```

**In local VS Code:**

```
Remote Explorer → Tunnels → roihu-gpu-int
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
- Roihu CPU nodes require the x64 VS Code CLI.
- Roihu GPU nodes require the ARM64 VS Code CLI.
- Keep the original SSH terminal open while using VS Code.
- The tunnel stops when the Slurm allocation ends.
- The maximum CPU interactive allocation used in this guide is 32 CPU cores and 62 GiB of RAM.
- The GPU interactive allocation used in this guide requests 36 CPU cores, 1 GPU, and 9 hours of runtime.
- The GPU interactive partition uses fixed memory. `sinteractive` may show 110000 MB, while Slurm may override it to 217086 MB.
- The maximum GPU interactive runtime is 12 hours.
- The `gpuinteractive` partition should be accessed through `sinteractive` from the `roihu-gpu` login node.
- The `gpuinteractive` partition currently provides full GPUs until GPU slices are fully configured.
- Use batch jobs for long-running production workloads.
- The `vscode-interactive-cpu` shell function combines CPU allocation and tunnel startup into a single command.
- The `vscode-interactive-gpu` shell function combines GPU allocation and tunnel startup into a single command.
