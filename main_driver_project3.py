import os
import csv
from src.map_generator import generate_ship
from src.mdp_solver import (
    value_iteration,
    export_T_to_csv,
    find_max_T_state,
    extract_optimal_policy
)

# Parameters
D = 30
p_factor = 0.5
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

def save_policy(policy, filepath="data/policy_ship1.csv"):
    """Save the extracted optimal policy to a CSV file."""
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bx", "by", "rx", "ry", "dx", "dy"])
        for (bx, by, rx, ry), (dx, dy) in policy.items():
            writer.writerow([bx, by, rx, ry, dx, dy])
    print(f"Policy saved to: {filepath}")

def main():
    print("Generating shi: ")
    ship_map = generate_ship(D, p_factor)

    print("Running value iteration: ")
    T = value_iteration(ship_map)

    print("Exporting T-values to CSV: ")
    export_T_to_csv(T, ship_map, filepath=f"{output_dir}/T_ship1.csv")

    print("Finding worst-case configuration: ")
    max_state, max_val = find_max_T_state(T, ship_map)
    print(f"\nWorst-case (slowest) configuration:")
    print(f"  Bot: ({max_state[0]}, {max_state[1]}), Rat: ({max_state[2]}, {max_state[3]})")
    print(f"  Expected time to catch: {max_val:.2f}")

    print("Extracting optimal policy: ")
    policy = extract_optimal_policy(T, ship_map)
    print(f"  Extracted {len(policy)} optimal actions.")

    print("Saving policy to CSV: ")
    save_policy(policy, filepath=f"{output_dir}/policy_ship1.csv")

if __name__ == "__main__":
    main()
