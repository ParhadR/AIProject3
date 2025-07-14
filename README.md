# AIProject3
01:198:440:G6 INTRO ARTIFIC INTELL - Project 3: MDPs and ML

---

## Project Structure
- `main_driver_project3.py`: Generates ship, runs value iteration, and exports T-values.
- `src/`: MDP solver, model training, and map generation.
  - `map_generator.py`: Generates random maze-like ship maps.
  - `mdp_solver.py`: Value iteration engine and T export.
  - `train_model.py`: Trains ML model on T(bx, by, rx, ry) data.
- `models/`: Stores trained PyTorch models.
- `data/`: Stores generated CSV datasets.
- `run_model.py`: Loads and evaluates trained models on test cases.
- `multi-ship/`: Folder for running against mulitple ships, will update rest later

---

## Setup Instructions

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
