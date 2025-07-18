import numpy as np
import csv
from src.map_generator import OPEN

# Directions for movement (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def get_neighbors(x, y, ship_map):
    D = len(ship_map)
    return [
        (nx, ny)
        for dx, dy in DIRECTIONS
        if 0 <= (nx := x + dx) < D and 0 <= (ny := y + dy) < D and ship_map[nx][ny] == OPEN
    ]

# sets up the initial T array with all values 1000 except caught cases (T = 0)
def initialize_T(ship_map):
    D = len(ship_map)
    T = np.full((D, D, D, D), 1000.0) # high default value as per Jack (TA)'s feedback

    for bx in range(D):
        for by in range(D):
            for rx in range(D):
                for ry in range(D):
                    # skip if either bot or rat starts on a wall
                    if ship_map[bx][by] != OPEN or ship_map[rx][ry] != OPEN:
                        continue
                    if (bx, by) == (rx, ry):
                        T[bx][by][rx][ry] = 0.0  # already caught
    return T

# runs value iteration to fill in T values with expected steps until capture
def value_iteration(ship_map, max_iters=1000, tolerance=1e-3):
    D = len(ship_map)
    T = initialize_T(ship_map)

    for iteration in range(max_iters):
        delta = 0.0
        num_updates = 0
        num_discoveries = 0
        new_T = T.copy()

        for bx in range(D):
            for by in range(D):
                if ship_map[bx][by] != OPEN:
                    continue
                for rx in range(D):
                    for ry in range(D):
                        if ship_map[rx][ry] != OPEN or (bx, by) == (rx, ry):
                            continue

                        bot_neighbors = get_neighbors(bx, by, ship_map)
                        if not bot_neighbors:
                            continue

                        best_value = 1000.0
                        for nbx, nby in bot_neighbors:
                            rat_neighbors = get_neighbors(rx, ry, ship_map)
                            if not rat_neighbors:
                                continue

                            expected = 0.0
                            for nrx, nry in rat_neighbors:
                                if (nbx, nby) == (nrx, nry):
                                    continue  # caught here
                                v = T[nbx][nby][nrx][nry]
                                expected += v

                            expected /= len(rat_neighbors)
                            total_cost = 1 + expected
                            best_value = min(best_value, total_cost)

                        old_val = T[bx][by][rx][ry]
                        new_T[bx][by][rx][ry] = best_value

                        if old_val == 1000.0 and best_value < 1000.0:
                            num_discoveries += 1
                            delta = float("inf") # new reachable state found
                        else:
                            change = abs(old_val - best_value)
                            if change > delta:
                                delta = change
                            if change > 0:
                                num_updates += 1

        T = new_T
        print(
            f"Iteration {iteration + 1}: Δ = {delta:.4f}, updates = {num_updates}, new discoveries = {num_discoveries}"
        )
        
        # stop if changes are small enough (and not still discovering new states)
        if delta != float("inf") and delta < tolerance:
            break

    return T

def extract_optimal_policy(T, ship_map):
    D = len(ship_map)
    policy = {}

    for bx in range(D):
        for by in range(D):
            if ship_map[bx][by] != OPEN:
                continue
            for rx in range(D):
                for ry in range(D):
                    if ship_map[rx][ry] != OPEN or (bx, by) == (rx, ry):
                        continue

                    best_action = None
                    best_value = float("inf")

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
                                continue
                            expected += T[nbx][nby][nrx][nry]

                        expected /= len(rat_neighbors)
                        total_cost = 1 + expected

                        if total_cost < best_value:
                            best_value = total_cost
                            best_action = (dx, dy)

                    if best_action is not None:
                        policy[(bx, by, rx, ry)] = best_action

    return policy

# finds the state (bx,by,rx,ry) with the highest non-infinite T value
def find_max_T_state(T, ship_map):
    D = len(ship_map)
    max_val = -1
    max_state = None

    for bx in range(D):
        for by in range(D):
            if ship_map[bx][by] != OPEN:
                continue
            for rx in range(D):
                for ry in range(D):
                    if ship_map[rx][ry] != OPEN or (bx, by) == (rx, ry):
                        continue

                    val = T[bx][by][rx][ry]
                    if val < 1000.0 and val > max_val:
                        max_val = val
                        max_state = (bx, by, rx, ry)

    return max_state, max_val

def export_T_to_csv(T, ship_map, filepath="T_dataset.csv"):
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
                        if val < 1000.0:
                            writer.writerow([bx, by, rx, ry, val])

    print(f"Exported T dataset to: {filepath}")
