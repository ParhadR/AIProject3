# AIProject3
01:198:440:G6 INTRO ARTIFIC INTELL - Project 3: MDPs and ML

## Project Overview

You are a bot tasked with removing a rat from a maze-like ship. Both you and the rat take turns moving. The rat moves randomly to a neighboring open cell, while the bot's goal is to catch it as efficiently as possible.

This project has two main phases:
1. **MDP Value Iteration** — Compute the optimal expected number of moves (T) required to catch the rat using dynamic programming.
2. **Machine Learning** — Train a neural network to approximate the function T(bx, by, rx, ry) from data generated in Phase 1.

---

## Project Structure
- `main_driver_project3.py`: Generates ship, runs value iteration, and exports T-values.
- `src/`: Core logic for MDP solver, model training, and map generation.
  - `map_generator.py`: Generates random maze-like ship maps.
  - `mdp_solver.py`: Value iteration engine and T export.
  - `train_model.py`: Trains ML model on T(bx, by, rx, ry) data.
- `models/`: Stores trained PyTorch models.
- `data/`: Stores generated CSV datasets.
- `run_model.py`: Loads and evaluates trained models on test cases.

---

## Setup Instructions

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
