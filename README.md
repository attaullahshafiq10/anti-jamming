# Autonomous UAV Anti-Jamming Framework

This repository contains the PyTorch implementation of the methodology described in the research paper:
**"Energy Efficient Deep Reinforcement Learning for Autonomous UAV Anti Jamming"**

## Overview
The framework provides an energy-efficient, autonomous anti-jamming solution for UAVs operating in dynamic adversarial RF environments. It is structured into two primary tiers:
- **Tier 1 (Spectrum Perception):** A lightweight 1D-CNN–Bi-LSTM module that directly processes raw RF telemetry to classify five distinct spectral jamming states.
- **Tier 2 (Autonomous Mitigation Engine):** A Double Dueling Deep Q-Network enhanced with Prioritized Experience Replay (3DQN-PER) that dynamically selects optimal, battery-preserving evasion strategies (e.g., localized channel switching, wideband frequency hopping).

## Directory Structure
```text
code/
├── configs/
│   └── config.py               # Centralized hyperparameters, reward weights, and paths
├── data/
│   ├── raw_data.csv            # Raw UAV telemetry CSV file
│   └── preprocess_data.csv     # Processed sequences and scaled features file
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion and structural cleaning
│   ├── preprocessing.py        # Train/test split, Min-Max scaling, SMOTE, and temporal windowing
│   ├── dataset.py              # PyTorch Dataset and DataLoader wrappers
│   ├── model.py                # Tier 1 1D-CNN-Bi-LSTM architecture
│   ├── dqn_model.py            # Tier 2 Dueling Double DQN architecture
│   ├── replay_buffer.py        # Prioritized Experience Replay (PER) buffer
│   ├── environment.py          # Configurable RF wireless simulation interface
│   ├── reward.py               # Multi-objective reward calculation formulation
│   ├── train_perception.py     # Tier 1 supervised training loop
│   ├── evaluate_perception.py  # Tier 1 classification metrics (Accuracy, F1, ROC-AUC)
│   ├── train_agent.py          # Tier 2 reinforcement learning episode loop
│   ├── evaluate_agent.py       # Tier 2 policy evaluation (Convergence, PDR, Energy)
│   ├── explainability.py       # Trustworthy AI integrations (SHAP global, LIME local)
│   ├── metrics.py              # Algorithmic complexity and MFLOPs calculation
│   ├── visualization.py        # Plotting utilities for t-SNE, confusion matrices, and curves
│   └── main.py                 # Core entry point orchestrating the end-to-end pipeline
└── requirements.txt            # Python environment dependencies
```

## Installation
Ensure you have Python 3.8+ installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Execute the full end-to-end pipeline:
```bash
python src/main.py
```
