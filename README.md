# Resilient Grid MLSecOps Pipeline
**Automated Digital Twin for AC Power System Security**



## Project Overview
This project represents an industry-grade evolution of the Resilient Grid IDS. It transitions from monolithic research scripts to a **fully orchestrated MLSecOps pipeline**. By leveraging **Spatio-Temporal Graph Attention Networks (STGAT)** and **Prefect Orchestration**, the system acts as a continuous, physics-informed Digital Twin capable of intercepting surgical Advanced Persistent Threats (APTs) on massive topologies like the **IEEE-118 national grid**.

## The Engineering Pipeline (Orchestration)
To ensure production-grade observability and reproducibility, the project is structured as a modular Directed Acyclic Graph (DAG):

* **Step 1: Extract (Physics Simulation):** High-fidelity AC Power Flow generation via `pandapower`, simulating thousands of timesteps of SCADA logs.
* **Step 2: Transform (Graph Engineering):** Automated construction of 3D temporal tensors ($Nodes \times Time \times Features$) with sliding-window logic to capture grid inertia.
* **Step 3: Train (Adversarial ML):** Robust training of the STGAT using **Topological DropEdge** and **PGD Adversarial Training** to ensure N-1 Contingency awareness.
* **Step 4: Audit (Security Detection):** A live Digital Twin control room that calibrates **4-Sigma dynamic thresholds** to detect FDIA and Breaker attacks.



## Core Scientific Innovations

### 1. Spatio-Temporal Physics (Deep STGAT)
The architecture combines spatial and temporal modeling to obey Kirchhoff’s Laws and generator momentum:
* **LSTM Encoder:** Captures the physical inertia of the grid over a 5-timestep window, preventing attackers from "rewriting" system history.
* **Deep GAT with Jumping Knowledge (JK):** A 5-layer attention mechanism that embeds physical Resistance ($R$) and Reactance ($X$) as edge attributes, allowing the AI to scale to 118+ buses without mathematical oversmoothing.



### 2. Multi-Vector APT Simulation
* **Surgically Targeted FDIA:** A White-Box attacker that uses a Spatio-Temporal mask to spoof a single SCADA sensor at the live timestep ($t_0$), mimicking a Stuxnet-style strike.
* **Topological Breaker Attacks:** Simulates the digital severing of transmission lines in the graph topology to test the AI's N-1 Contingency awareness.

## Security & Mitigation Logic
The `detector.py` engine operates as a live, self-healing loop:
* **Dynamic Calibration:** Establish per-node anomaly thresholds based on statistical noise floors.
* **Quarantine & Heal:** Upon detecting a physics discrepancy, the system isolates compromised sensors and overwrites them with the AI's trusted physics predictions.
* **Resilience Insight:** The pipeline distinguishes between "Security Failures" and "Physical Resilience," acknowledging when the grid's N-1 redundancy absorbs a strike without voltage collapse.

## Security Audit Performance
| Metric | Performance | Engineering Logic |
| :--- | :--- | :--- |
| **Node Detection (FDIA)** | **~100%** | Caught via physical discrepancy vs. Digital Twin. |
| **Edge Detection (Breaker)**| **Variable** | Only flags attacks causing physical shifts; ignores redundant rerouting. |
| **False Alarm Rate** | **< 1%** | Minimized via calibrated 4-Sigma dynamic thresholding. |

## Getting Started

1.  **Start the Orchestration UI:**
    ```bash
    prefect server start
    ```
2.  **Run the End-to-End Pipeline:**
    ```bash
    python pipeline.py
    ```
    *Monitor logs, execution graphs, and security artifacts at `http://127.0.0.1:4200`.*

---

## Project Structure
* `pipeline.py`: The Prefect orchestrator and entry point.
* `detector.py`: The Digital Twin security auditor and mitigation engine.
* `data/`: Modular scripts for physics simulation and tensor preprocessing.
* `models/`: `DeepPowerSTGAT` architecture with Jumping Knowledge.
* `attacks/`: White-box PGD and APT simulation algorithms.
* `explanation/`: Integrated Gradients (Captum) for attack attribution.
