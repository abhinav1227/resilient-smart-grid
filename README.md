# Resilient Grid: MLSecOps for AC Power Systems ⚡🛡️

## Overview
This project is an end-to-end Machine Learning Security Operations (MLSecOps) pipeline designed to protect AC power grids from False Data Injection Attacks (FDIA). 

Using a Graph Convolutional Network (GCN) trained on IEEE 14-bus topological data, the system acts as a "Digital Twin" of the grid. It learns the complex, non-linear AC power flow physics to predict both **Voltage Magnitude (|V|)** and **Voltage Angle (δ)**. It leverages **Projected Gradient Descent (PGD) Adversarial Training** to harden the AI against malicious noise, deploying it as an active Intrusion Detection System (IDS) that cross-checks physical SCADA sensors in real-time.

## Core Features
* **Full AC Steady-State Engine:** Ingests Active (P) and Reactive (Q) loads to calculate complete steady-state phasors, acting as a neural Newton-Raphson solver.
* **White-Box Vulnerability Discovery:** Utilizes FGSM and iterative PGD algorithms to dynamically scale physical attacks, proving baseline model vulnerability.
* **Minimax Adversarial Defense:** Employs randomized adversarial training loops to drop the Adversarial Success Rate (ASR) of targeted attacks from ~100% to near 0%.
* **Dynamic Digital Twin IDS:** Features a purely functional detection script that calculates per-node 3-Sigma statistical thresholds to catch sophisticated cyber-attacks while eliminating false positives.

## Architecture Pipeline
1. `data/generate.py`: Simulates grid load fluctuations and extracts topological edge indices and AC steady-state targets.
2. `models/gcn.py`: A PyTorch Geometric GCN utilizing message passing to learn electrical coupling.
3. `attacks/pgd.py(fgsm.py)`: A highly optimized, iterative attack generator that navigates the model's loss landscape to find worst-case physical perturbations.
4. `main.py`: The orchestration engine handling data loading, standard training, PGD adversarial training, and evaluation.
5. `detector.py`: The control room simulation script that dynamically generates sophisticated FDIA attacks and catches them using the hardened GNN.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/abhinav1227/resilient-smart-grid.git](https://github.com/abhinav1227/resilient-smart-grid.git)
   cd resilient-grid