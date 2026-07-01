# CSC Roihu SSH and Storage Setup

This guide covers:

1. Configuring CSC SSH certificate generation on a local workstation
2. Configuring persistent SSH storage on Mahti
3. Configuring direct and interactive SSH connections to Roihu
4. Mounting Roihu storage on macOS with rclone

---

## 1. Local Workstation Setup

### 1.1 Clone the CSC Certificate Helper Tool
Run this once on the local workstation:

```bash
cd ~
git clone [https://github.com/CSCfi/certificate-helper-tool.git](https://github.com/CSCfi/certificate-helper-tool.git)
cd certificate-helper-tool

```

### 1.2 Test SSH Certificate Generation

The command opens a browser for CSC authentication:

```bash
python3 csc_cert.py -u kanghans ~/.ssh/id_ed25519.pub

```

### 1.3 Add a Persistent Zsh Command

Add the following function to `~/.zshrc`:

```bash
cat >> ~/.zshrc <<'EOF'
# Generate a CSC SSH certificate
csc-ssh-keys() {
    (
        cd ~/certificate-helper-tool || return 1
        python3 csc_cert.py -u kanghans ~/.ssh/id_ed25519.pub
    )
}
EOF

```

Reload the Zsh configuration:

```bash
source ~/.zshrc

```

Verify the command:

```bash
csc-ssh-keys

```

---

## 2. Mahti SSH Setup

Store the SSH configuration under the persistent project scratch directory and link `~/.ssh` to it.

### 2.1 Create the Persistent SSH Directory

Check whether the directory already exists:

```bash
ls -la /scratch/project_2015384/Hanseul/SSH

```

Create it when necessary:

```bash
mkdir -p /scratch/project_2015384/Hanseul/SSH

```

### 2.2 Back Up the Existing SSH Directory

Run this only once:

```bash
mv ~/.ssh ~/.ssh_backup 2>/dev/null

```

### 2.3 Create the Symbolic Link

```bash
ln -s /scratch/project_2015384/Hanseul/SSH ~/.ssh

```

Verify the link:

```bash
ls -ld ~/.ssh

```

### 2.4 Set SSH Permissions

```bash
chmod 700 /scratch/project_2015384/Hanseul/SSH
chmod 600 /scratch/project_2015384/Hanseul/SSH/id_ed25519
chmod 644 /scratch/project_2015384/Hanseul/SSH/id_ed25519.pub

```

When an SSH configuration file exists:

```bash
chmod 600 /scratch/project_2015384/Hanseul/SSH/config

```

### 2.5 Configure CSC Certificate Generation on Mahti

Add the following function to `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'
# Generate a CSC SSH certificate
csc-ssh-keys() {
    python3.9 ~/certificate-helper-tool/csc_cert.py \
        -u kanghans \
        /scratch/project_2015384/Hanseul/SSH/id_ed25519.pub
}
EOF

```

Reload the Bash configuration:

```bash
source ~/.bashrc

```

Verify the command:

```bash
csc-ssh-keys

```

> **Note:** The `certificate-helper-tool` repository must also exist at `~/certificate-helper-tool` on Mahti.

---

## 3. Roihu SSH Configuration

Add the following entries to `~/.ssh/config`:

```ssh
Host roihu-cpu
    HostName roihu-cpu.csc.fi
    User kanghans
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host roihu-gpu
    HostName roihu-gpu.csc.fi
    User kanghans
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host roihu-cpu-interactive
    HostName roihu-cpu.csc.fi
    User kanghans
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ConnectTimeout 120
    ServerAliveInterval 30
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand ssh roihu-cpu "srun --account=project_2015384 --partition=interactive --cpus-per-task=32 --mem=64G --time=09:00:00 --unbuffered nc localhost 22"

Host roihu-gpu-interactive
    HostName roihu-gpu.csc.fi
    User kanghans
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ConnectTimeout 120
    ServerAliveInterval 30
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand ssh roihu-gpu "srun --account=project_2015384 --partition=gpuinteractive --gres=gpu:1 --time=09:00:00 --unbuffered nc localhost 22"

```

Set the correct permissions:

```bash
chmod 600 ~/.ssh/config

```

### 3.1 Test Direct Connections

```bash
ssh roihu-cpu
ssh roihu-gpu

```

### 3.2 Test Interactive Connections

The interactive hosts request a Slurm allocation before establishing the final SSH connection.

```bash
ssh roihu-cpu-interactive
ssh roihu-gpu-interactive

```

> `StrictHostKeyChecking no` and `UserKnownHostsFile /dev/null` suppress host-key verification for dynamically allocated interactive nodes.

---

## 4. Mount Roihu Storage on macOS

The following configuration mounts the Roihu project directory locally through rclone.

### 4.1 Paths

* **Remote directory:** `Roihu:/scratch/project_2015384/Hanseul`
* **Local mount point:** `/Users/kangh3/ROIHU`
* **Log file:** `/Users/kangh3/Rclone/rclone-roihu.log`

Create the required local directories once:

```bash
mkdir -p /Users/kangh3/ROIHU
mkdir -p /Users/kangh3/Rclone

```

### 4.2 Add Mount Functions to .zshrc

Open the configuration file:

```bash
nano ~/.zshrc

```

Add the following functions:

```bash
# Mount Roihu project storage
mount-roihu() {
    csc-ssh-keys || return 1
    ssh-add ~/.ssh/id_ed25519 || return 1
    mkdir -p /Users/kangh3/ROIHU
    mkdir -p /Users/kangh3/Rclone
    rclone mount \
        Roihu:/scratch/project_2015384/Hanseul \
        /Users/kangh3/ROIHU \
        --vfs-cache-mode full \
        --vfs-cache-max-size 10G \
        --vfs-read-chunk-size 32M \
        --buffer-size 64M \
        --vfs-cache-max-age 24h \
        --no-modtime \
        --timeout 30m \
        --attr-timeout 5s \
        --dir-cache-time 5m \
        --tpslimit 10 \
        --log-level INFO \
        --log-file /Users/kangh3/Rclone/rclone-roihu.log \
        --daemon
}

# Unmount Roihu project storage
unmount-roihu() {
    pkill -SIGTERM -f "rclone mount.*Roihu:" 2>/dev/null
    sleep 2
    diskutil unmount force /Users/kangh3/ROIHU 2>/dev/null || \
        umount -f /Users/kangh3/ROIHU 2>/dev/null
}

```

Reload the configuration:

```bash
source ~/.zshrc

```

### 4.3 Mount Roihu

```bash
mount-roihu

```

Verify the mount:

```bash
mount | grep ROIHU

```

Alternatively:

```bash
ls -la /Users/kangh3/ROIHU

```

### 4.4 Unmount Roihu

```bash
unmount-roihu

```

```bash
tail -f /Users/kangh3/Rclone/rclone-roihu.log

```

---

## 5. Routine Commands

### Local Workstation

* **Generate or renew the CSC SSH certificate:** `csc-ssh-keys`
* **Connect to Roihu:** `ssh roihu-cpu`
* **Mount Roihu storage:** `mount-roihu`
* **Unmount Roihu storage:** `unmount-roihu`

### Mahti

* **Generate or renew the CSC SSH certificate:** `csc-ssh-keys`
* **Connect to Roihu:** `ssh roihu-cpu`
* **Connect to an interactive CPU node:** `ssh roihu-cpu-interactive`
* **Connect to an interactive GPU node:** `ssh roihu-gpu-interactive`

---

## 6. Troubleshooting

### SSH reports incorrect key permissions

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/config

```

### The CSC certificate has expired

Generate a new certificate:

```bash
csc-ssh-keys

```

Then reload the key:

```bash
ssh-add ~/.ssh/id_ed25519

```

### ssh-add reports that no authentication agent exists

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

```

### The Roihu mount already exists

```bash
unmount-roihu
mount-roihu

```

### The mount directory appears frozen

```bash
pkill -SIGTERM -f "rclone mount.*Roihu:"
sleep 2
diskutil unmount force /Users/kangh3/ROIHU

```

Fallback:

```bash
umount -f /Users/kangh3/ROIHU

```

### Check whether rclone is running

```bash
pgrep -af rclone

```

### Inspect recent rclone errors

```bash
tail -n 100 /Users/kangh3/Rclone/rclone-roihu.log

```

---

## 7. Notes

* CSC SSH certificates expire and must periodically be regenerated with `csc-ssh-keys`.
* Keep private keys restricted to the owner with permission mode `600`.
* Prefer shell functions over multiline aliases for operations containing several commands.
* Avoid a general `pkill rclone` command when multiple independent rclone processes may be running.
* The interactive SSH hosts allocate compute resources through Slurm before opening the final connection.
