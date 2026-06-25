# CSC Environment Helpers

Last updated: 25 June 2026

---

## Overview & Motivation

This repository provides an automated, robust framework to streamline environment deployment and data management across CSC supercomputers (**Puhti / Mahti / Roihu**). Efficient distributed computing requires strict version control; this project provides standardised configurations to eliminate software deployment overheads.

A major driver for this architecture is the rigid dependency structure of machine learning and numerical tools. Specifically, **SmartSim 0.8.0** imposes strict upper bounds on **NumPy** versions ($< 2.0.0$). Conversely, modern machine learning, statistics, and chemical kinetics frameworks increasingly require **NumPy 2.0.0** or later. To resolve these conflicting constraints, this repository explicitly separates the execution stacks into distinct, isolated environments rather than forcing a single, unstable deployment.

Furthermore, this toolkit provides dedicated utilities to optimise high-throughput file transfers via **Allas**, minimizing data latency when interfacing with object storage.

---

## Repository Structure

The repository is organised into three functional directories:

```plaintext
CSCEnvironmentHelpers/
├── SmartSimEnvironment/       # Configurations for SmartSim–JAX–OpenFOAM workflows (NumPy < 2.0.0)
├── MLEnvironment/             # Configurations for modern ML and chemical kinetics (NumPy >= 2.0.0)
└── UsefulCommands/            # Production-ready SLURM templates and Allas data transfer scripts
└── Utilities/ 

```

---

## Directory Modules

### 1. SmartSimEnvironment

This module manages the deployment of **SmartSim 0.8.0 + SmartRedis 0.6.1** coupled with **JAX** and **Equinox** on CSC systems. To bypass the performance penalties associated with reading millions of small files across the Lustre parallel filesystem, we leverage **Tykky** to containerize the Python stack into a single-file image.

* **Target Workflows:** CFD–ML coupling, in-situ data processing, and non-premixed combustion modeling.
* **Core Constraint:** Explicitly locks NumPy to version `1.26.4` to prevent API breakage within the SmartSim orchestrator.

### 2. MLEnvironment

This module hosts the environment profiles for standard machine learning, advanced statistical modelling, and detailed chemical kinetics analysis.

* **Target Workflows:** Heavy training workloads, high-volume dimensional reduction, and standalone deep learning evaluations.
* **Core Strategy:** Utilises **NumPy 2.0.0** or higher to exploit modern vectorisation optimisations and ensure full compatibility with the latest data-science libraries.

### 3. UsefulCommands

A curated collection of highly optimised shell scripts and templates designed to maximise throughput on CSC infrastructure.

* **SLURM Templates:** Production-ready batch scripts configured for optimal core binding, memory allocation, and GPU job configurations across different partitions.
* **Allas Transfer Tools:** Scripted wrappers using the `s3cmd` and `swift` clients to achieve fast, parallelised data staging between computing nodes and object storage.

---

## Getting Started

### Prerequisites

Before initiating builds, you must define your global environment variables. Update your local configuration script as follows:

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"          # Your CSC project ID
export CSC_USER="USERNAME"                  # Your CSC username
export ENV_NICKNAME="NICKNAME"              # Desired environment name
# --- USER CONFIGURATION END ---

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$CSC_USER/Utilities"
mkdir -p "$BASE_SCRATCH"

```

> [!TIP]
> Move the deployment folders into **your own Utilities directory** on the scratch parallel filesystem.
> Confirm that the target paths exist before running the building toolchains.

---

## Usage & Environment Activation

To prevent environment pollution, clear your module path and load only the required software stacks.

### Activating the SmartSim Stack

```bash
source $BASE_SCRATCH/CSCEnvironmentHelpers/SmartSimEnvironment/load_env.sh

```

### Activating the Modern ML Stack

```bash
source $BASE_SCRATCH/CSCEnvironmentHelpers/MLEnvironment/load_env.sh

```
