# CSC HPC Guide

**Last updated:** 20 July 2026

**Written by:**
Aalto University
Department of Energy and Mechanical Engineering
Energy Conversion and Systems Team

---

## Overview & Motivation

This repository provides a practical setup guide for using CSC high-performance computing systems, with particular emphasis on the newly introduced **Roihu** environment. The workflow covers secure SSH authentication, remote development, cloud-storage mounting, and reproducible Python environments using Tykky.

*Most procedures also apply to **Puhti** and **Mahti** with minor modifications to hostnames, partitions, and modules.*

---

## Repository Structure

```text
CSC-HPC-Guide/
├── file-transfer/          # Data movement workflows
├── python-environment/     # Tykky/SmartSim/ML environment builds
├── rclone-mount-unmount/   # Cloud storage integration
├── ssh-connection/         # SSH, Certs, and VS Code Tunnels
└── utilities/              # Helper scripts
```

---

## Recommended Setup Workflow

### 1. SSH & Connection

* **[SSH Certificate](https://www.google.com/search?q=ssh-connection/ssh-certificate.md):** Foundation for all authentication.
* **[SSH Connection](https://www.google.com/search?q=ssh-connection/ssh-connection.md):** Managing login node access.
* **[VS Code Tunnel](https://www.google.com/search?q=ssh-connection/vscode-tunnel.md):** Remote development on interactive compute nodes.

### 2. Data & Storage

* **[rclone Mount](https://www.google.com/search?q=rclone-mount-unmount/rclone-mount-unmount.md):** Mounting cloud-hosted datasets.
* **[File Transfer](https://www.google.com/search?q=file-transfer/file-transfer.md):** Best practices for local <-> CSC data movement.

### 3. Python Environment Configuration

The Python environments are packaged with **Tykky** to minimise small-file I/O overhead on Lustre parallel filesystems.

#### Unified SmartSim and Machine-Learning Environment

* **[SmartSim Environment Configuration Guide](python-environment/smartsim-environment.md)**
* **Purpose:** SmartSim `1.0.3+csc`, SmartRedis `1.0.0+csc`, RedisAI, JAX, Equinox, TensorFlow, PyTorch, ONNX, PySR, and JuliaCall workflows.
* **Python:** 3.12
* **NumPy:** `>= 2.0`
* **TensorFlow:** `2.18.1`
* **PyTorch:** `2.7.1`
* **Architecture support:** x86_64 and ARM64/aarch64
* **RedisAI backends:** TensorFlow, ONNX Runtime, LibTorch, and JAX

This unified environment replaces the previously separate SmartSim and machine-learning environments. A standalone `PythonML` environment is not required when using this stack.

SmartSim and SmartRedis are installed from the CSC-maintained releases:

* SmartSim: `v1.0.3-csc`
* SmartRedis: `v1.0.0-csc`
* RedisAI: `v1.0.0-csc`

The Tykky environment and native SmartRedis library must be built separately for each architecture.

---

## Quick Start Links

1. [SSH Certificate Configuration](https://github.com/boss507104/CSC-HPC-Guide/blob/main/ssh-connection/ssh-certificate.md)
2. [SSH Connection to CSC Login Nodes](https://github.com/boss507104/CSC-HPC-Guide/blob/main/ssh-connection/ssh-connection.md)
3. [VS Code Tunnel to an Interactive Compute Node](https://github.com/boss507104/CSC-HPC-Guide/blob/main/ssh-connection/vscode-tunnel.md)
4. [rclone Mount and Unmount Guide](https://github.com/boss507104/CSC-HPC-Guide/blob/main/rclone-mount-unmount/rclone-mount-unmount.md)
5. [File Transfer Best Practices](https://github.com/boss507104/CSC-HPC-Guide/blob/main/file-transfer/file-transfer.md)
6. [Unified SmartSim and Machine-Learning Environment](https://github.com/boss507104/CSC-HPC-Guide/blob/main/python-environment/smartsim-environment.md)

---

## Recommended Usage Principles

| Resource | Best Practice |
| --- | --- |
| **Login Nodes** | SSH access, file management, job submission, lightweight editing. |
| **Interactive Compute Nodes** | Compilation, package installation, notebooks, debugging, environment builds. |
| **Batch Jobs** | Production simulations, large data processing, long-running workloads. |
| **Project Scratch** | Active datasets, software environments, temporary build data. |
| **Home Directory** | Only for lightweight configuration files, such as `.bashrc`. |

---

## System Compatibility

* **Targets:** Roihu, Puhti, Mahti.
* **Architectures:** x86_64 and ARM64/aarch64.
* **Key Considerations:** Always use the specific module versions, Slurm partitions, compiler versions, and GPU hardware configured for your target cluster.
* **Note:** Large builds, including Tykky containerization and SmartRedis compilation, must be executed on **compute nodes** through interactive allocations to avoid resource contention on shared login nodes.
