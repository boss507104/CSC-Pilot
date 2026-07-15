# Transfer File From Mahti to Roihu

This workflow outlines the exact procedure for synchronising heavy computational datasets from Mahti to Roihu using CSC certificate-based authentication and a background `tmux` session.

---

## 1. Environment Preparation on Mahti

Mahti uses a symbolic link for the `.ssh` directory mapping to the persistent scratch path to retain the authentication configurations. Ensure the cryptographic permissions are correctly enforced.

```bash
# Verify symbolic link and restrict permissions
chmod 700 /scratch/$CSC_PROJECT/$CSC_USER/System/.ssh
chmod 600 /scratch/$CSC_PROJECT/$CSC_USER/System/.ssh/id_ed25519

```

To enable the short-term certificate signing mechanism on the Mahti login node, append the registration alias to your shell profile.

```bash
# Append persistent shortcut to bash profile
echo "alias csc-ssh-keys='python3.9 ~/certificate-helper-tool/csc_cert.py -u $CSC_USER_ID /scratch/$CSC_PROJECT/$CSC_USER/SSH/id_ed25519.pub'" >> ~/.bashrc
source ~/.bashrc

```

---

## 2. Certificate Renewal

CSC utilizes token-based ephemeral short-term certificates. You must request authentication signing immediately prior to establishing an outbound connection to Roihu.

```bash
# Trigger the browser/PIN authentication link
csc-ssh-keys

```

---

## 3. Background Migration Layer via Tmux

Heavy datasets require persistent execution decoupling to protect the process from local network dropouts or workstation terminal closure.

```bash
# Initialize a dedicated background transfer layer
tmux new -s dataset_migration

```

---

## 4. Execution of Data Synchronization

Execute the `rsync` transaction by explicitly injecting the certified cryptographic key path into the secure shell environment block.

```bash
rsync -avhi --progress \
  -e "ssh -i /scratch/$CSC_PROJECT/$CSC_USER/SSH/id_ed25519 -o StrictHostKeyChecking=no" \
  /scratch/$CSC_PROJECT/$CSC_USER/$DATASET_DIR/ \
  $CSC_USER_ID@roihu-cpu.csc.fi:/scratch/$CSC_PROJECT/$CSC_USER/$DATASET_DIR/

```

> **Important structural note:** Leaving the trailing slash on the source directory path `/$DATASET_DIR/` processes only the constituent contents into the destination tree without generating redundant nested folders.

---

## 5. Process Monitoring and Session Management

To exit the foreground screen and leave the migration running in the background safely:

* Press `Ctrl + B`, release, and then press `D` (Detach).

To inspect the transfer velocity or verify completion at a later stage:

```bash
# Reattach to the live migration sequence
tmux a -t dataset_migration

```

When the transfer report lists `Done`, close the persistent framework safely by typing `exit`.
