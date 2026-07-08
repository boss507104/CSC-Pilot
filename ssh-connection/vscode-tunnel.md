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

### CPU interactive node

Connect to the Roihu CPU login node:

```bash
ssh roihu-cpu
```

From the Roihu CPU login node:

```bash
sinteractive --account project_xxxxxxxx --cores 32
```

Wait until Slurm grants the allocation.

Verify that the shell has moved to a compute node:

```bash
hostname
```

### GPU interactive node

Connect to the Roihu GPU login node:

```bash
ssh roihu-gpu
```

From the Roihu GPU login node:

```bash
sinteractive --account project_xxxxxxxx
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

> Request CPU interactive nodes from `roihu-cpu.csc.fi` and GPU interactive nodes from `roihu-gpu.csc.fi`.

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

   ```text
   https://github.com/login/device
   ```

3. Sign in with the same GitHub account used in local VS Code.
4. Enter the temporary device code shown in the Roihu terminal.
5. Approve access for Visual Studio Code.
6. Use one of the following tunnel names when prompted:

   ```text
   roihu-cpu-interactive
   ```

   or:

   ```text
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
5. Select the tunnel name:

   ```text
   roihu-cpu-interactive
   ```

   or:

   ```text
   roihu-gpu-interactive
   ```

Alternatively, open the Command Palette with **Command-Shift-P** and run:

```text
Remote Tunnels: Connect to Tunnel
```

VS Code now connects directly to the allocated compute node.

Open the project directory through:

**File → Open Folder**

For example:

```text
/scratch/project_xxxxxxxx/Harry
```

---

## 5. Close the Tunnel and Release the Node

Close the remote VS Code window.

Return to the SSH terminal running the tunnel and press:

```text
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

## 6. Optional: Shell Function Shortcut

To simplify allocation and tunnel startup, create a helper function on Roihu.

Create the directory for bash includes:

```bash
mkdir -p ~/.bashrc.d
```

### CPU shortcut

Create this file on `roihu-cpu`:

```bash
cat > ~/.bashrc.d/vscode-interactive.sh << 'EOF'
# Slurm CPU interactive allocation launcher
vscode-interactive() {
    sinteractive --account project_xxxxxxxx --cores 32
}
EOF
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

Confirm the function is available:

```bash
type vscode-interactive
```

Then start a CPU interactive session with:

```bash
vscode-interactive
```

After the allocation starts, run this on the allocated compute node:

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

### GPU shortcut

Create this file on `roihu-gpu`:

```bash
cat > ~/.bashrc.d/vscode-interactive.sh << 'EOF'
# Slurm GPU interactive allocation launcher
vscode-interactive() {
    sinteractive --account project_xxxxxxxx
}
EOF
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

Confirm the function is available:

```bash
type vscode-interactive
```

Then start a GPU interactive session with:

```bash
vscode-interactive
```

After the allocation starts, run this on the allocated compute node:

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

> The same function name can be used on both `roihu-cpu` and `roihu-gpu`. Each login environment has its own shell configuration.

> Ensure `~/.bashrc` sources files from `~/.bashrc.d/`. If it does not, add this to `~/.bashrc`:

```bash
for file in ~/.bashrc.d/*.sh; do
    [ -r "$file" ] && source "$file"
done
```

---

## 7. Routine Workflow

After the VS Code CLI has been installed, use one of the following workflows for each session.

### CPU workflow

On the local workstation:

```bash
csc-ssh-keys
ssh roihu-cpu
```

On the Roihu CPU login node:

```bash
vscode-interactive
```

or:

```bash
sinteractive --account project_xxxxxxxx --cores 32
```

On the allocated CPU compute node:

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

In local VS Code:

```text
Remote Explorer → Tunnels → roihu-cpu-interactive
```

### GPU workflow

On the local workstation:

```bash
csc-ssh-keys
ssh roihu-gpu
```

On the Roihu GPU login node:

```bash
vscode-interactive
```

or:

```bash
sinteractive --account project_xxxxxxxx
```

On the allocated GPU compute node:

```bash
cd ~/bin
./code tunnel --accept-server-license-terms
```

In local VS Code:

```text
Remote Explorer → Tunnels → roihu-gpu-interactive
```

### When finished

Stop the tunnel:

```text
Ctrl-C
```

Then exit the allocation and login node:

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
- The CPU interactive partition supports up to 32 CPU cores and 64 GiB of memory.
- The GPU interactive partition is intended for GPU work through `sinteractive`.
- During the current Roihu pilot, GPU slices are not yet fully configured, so GPU interactive sessions may provide full GPUs.
- Use batch jobs for long-running production workloads.
- The `vscode-interactive` shell function can use the same name on CPU and GPU login nodes.
