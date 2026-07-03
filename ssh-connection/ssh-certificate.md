# CSC SSH Certificate Setup

This guide covers:

1. Installing the CSC certificate helper tool
2. Configuring the CSC username
3. Generating and renewing an SSH certificate with `csc-ssh-keys`

---

## 1. Clone the CSC Certificate Helper Tool

Run this once on your local workstation:

```bash
cd ~
git clone https://github.com/CSCfi/certificate-helper-tool.git
```

---

## 2. Configure the CSC SSH Certificate Helper

Replace `Harry` with your own CSC username, then append the following configuration to `~/.zshrc`:

```bash
cat >> ~/.zshrc <<'EOF'

# CSC SSH certificate configuration
export CSC_USER="Harry"

# Generate or renew a CSC SSH certificate
csc-ssh-keys() {
    (
        cd ~/certificate-helper-tool || return 1
        python3 csc_cert.py -u "${CSC_USER}" ~/.ssh/id_ed25519.pub
    )
}
EOF
```

The CSC username now appears in only one place. If you later use a different CSC account, simply change the value of `CSC_USER`.

Reload the Zsh configuration:

```bash
source ~/.zshrc
```

---

## 3. Test SSH Certificate Generation

First, test the certificate helper directly:

```bash
python3 ~/certificate-helper-tool/csc_cert.py \
    -u "${CSC_USER}" \
    ~/.ssh/id_ed25519.pub
```

The command opens a browser for CSC authentication and signs your existing SSH public key, generating a new CSC SSH certificate.

Next, verify that the helper function works:

```bash
csc-ssh-keys
```

Whenever the CSC SSH certificate expires, simply run:

```bash
csc-ssh-keys
```

to generate a new certificate.
