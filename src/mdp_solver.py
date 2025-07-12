import numpy as np
from collections import deque
from src.map_generator import OPEN

# Directions for movement (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def get_neighbors(x, y, ship_map):
    """Return valid neighboring OPEN cells."""
    D = len(ship_map)
    return [
        (nx, ny)
        for dx, dy in DIRECTIONS
        if 0 <= (nx := x + dx) < D and 0 <= (ny := y + dy) < D and ship_map[nx][ny] == OPEN
    ]

def initialize_T(ship_map):
    """Initialize T with base cases and invalid positions."""
    D = len(ship_map)
    T = np.full((D, D, D, D), np.inf)

    for bx in range(D):
        for by in range(D):
            for rx in range(D):
                for ry in range(D):
                    if ship_map[bx][by] != OPEN or ship_map[rx][ry] != OPEN:
                        continue
                    if (bx, by) == (rx, ry):
                        T[bx][by][rx][ry] = 0.0  # Caught immediately
    return T

def value_iteration(ship_map, max_iters=1000, tolerance=1e-3):
    """Compute T(bx, by, rx, ry) via value iteration."""
    D = len(ship_map)
    T = initialize_T(ship_map)

    for iteration in range(max_iters):
        delta = 0.0
        new_T = T.copy()

        for bx in range(D):
            for by in range(D):
                if ship_map[bx][by] != OPEN:
                    continue
                for rx in range(D):
                    for ry in range(D):
                        if ship_map[rx][ry] != OPEN:
                            continue
                        if (bx, by) == (rx, ry):
                            continue  # Already 0

                        best_value = np.inf
                        bot_neighbors = get_neighbors(bx, by, ship_map)
                        if not bot_neighbors:
                            continue

                        for nbx, nby in bot_neighbors:
                            rat_neighbors = get_neighbors(rx, ry, ship_map)
                            if not rat_neighbors:
                                continue

                            expected = 0.0
                            for nrx, nry in rat_neighbors:
                                if (nbx, nby) == (nrx, nry):
                                    expected += 0  # Rat caught next turn
                                else:
                                    expected += T[nbx][nby][nrx][nry]

                            expected /= len(rat_neighbors)
                            best_value = min(best_value, 1 + expected)

                        new_T[bx][by][rx][ry] = best_value
                        delta = max(delta, abs(T[bx][by][rx][ry] - best_value))

        T = new_T
        print(f"Iteration {iteration + 1}: Δ = {delta:.4f}")
        if delta < tolerance:
            break

    return T

def extract_optimal_policy(T, ship_map):
    """
    For each (bx, by, rx, ry), return the optimal move for the bot as a direction tuple (dx, dy).
    Skip invalid configurations (walls).
    """
    D = len(ship_map)
    policy = {}

    for bx in range(D):
        for by in range(D):
            if ship_map[bx][by] != OPEN:
                continue
            for rx in range(D):
                for ry in range(D):
                    if ship_map[rx][ry] != OPEN:
                        continue
                    if (bx, by) == (rx, ry):
                        continue  # Terminal state — no move needed

                    best_action = None
                    best_value = float('inf')

                    for dx, dy in DIRECTIONS:
                        nbx, nby = bx + dx, by + dy
                        if not (0 <= nbx < D and 0 <= nby < D):
                            continue
                        if ship_map[nbx][nby] != OPEN:
                            continue

                        rat_neighbors = get_neighbors(rx, ry, ship_map)
                        if not rat_neighbors:
                            continue

                        expected = 0.0
                        for nrx, nry in rat_neighbors:
                            if (nbx, nby) == (nrx, nry):
                                expected += 0  # Immediate capture
                            else:
                                expected += T[nbx][nby][nrx][nry]

                        expected /= len(rat_neighbors)
                        total_cost = 1 + expected

                        if total_cost < best_value:
                            best_value = total_cost
                            best_action = (dx, dy)

                    if best_action is not None:
                        policy[(bx, by, rx, ry)] = best_action

    return policy

def find_max_T_state(T, ship_map):
    """
    Find the (bx, by, rx, ry) configuration with the maximum expected time T.
    Returns: ((bx, by, rx, ry), max_value)
    """
    D = len(ship_map)
    max_val = -1
    max_state = None

    for bx in range(D):
        for by in range(D):
            if ship_map[bx][by] != OPEN:
                continue
            for rx in range(D):
                for ry in range(D):
                    if ship_map[rx][ry] != OPEN:
                        continue
                    if (bx, by) == (rx, ry):
                        continue  # T = 0 in this case

                    val = T[bx][by][rx][ry]
                    if val > max_val:
                        max_val = val
                        max_state = (bx, by, rx, ry)

    return max_state, max_val

import csv

def export_T_to_csv(T, ship_map, filepath="T_dataset.csv"):
    """
    Export all valid (bx, by, rx, ry) → T mappings to a CSV file.
    Columns: bx, by, rx, ry, T
    """
    D = len(ship_map)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bx", "by", "rx", "ry", "T"])

        for bx in range(D):
            for by in range(D):
                if ship_map[bx][by] != OPEN:
                    continue
                for rx in range(D):
                    for ry in range(D):
                        if ship_map[rx][ry] != OPEN:
                            continue
                        val = T[bx][by][rx][ry]
                        if not np.isinf(val):  # Skip unreachable
                            writer.writerow([bx, by, rx, ry, val])

    print(f"Exported T dataset to: {filepath}")
