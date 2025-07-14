import pandas as pd
import numpy as np

policy_df = pd.read_csv("data/policy_ship1.csv")


# Define the greedy direction (shortest-path) function
def greedy_direction(bx, by, rx, ry):
    dx = rx - bx
    dy = ry - by
    if abs(dx) > abs(dy):
        return (np.sign(dx), 0)
    elif abs(dy) > abs(dx):
        return (0, np.sign(dy))
    elif abs(dx) == abs(dy) and dx != 0:
        return (np.sign(dx), 0)  # prefer horizontal if equal
    return (0, 0)  # same cell


# Compare greedy vs optimal actions
total = 0
matches = 0
diffs = []

for _, row in policy_df.iterrows():
    bx, by, rx, ry = int(row["bx"]), int(row["by"]), int(row["rx"]), int(row["ry"])
    dx_opt, dy_opt = int(row["dx"]), int(row["dy"])
    dx_greedy, dy_greedy = greedy_direction(bx, by, rx, ry)

    total += 1
    if (dx_opt, dy_opt) == (dx_greedy, dy_greedy):
        matches += 1
    else:
        diffs.append(
            {
                "bx": bx,
                "by": by,
                "rx": rx,
                "ry": ry,
                "optimal_action": (dx_opt, dy_opt),
                "greedy_action": (dx_greedy, dy_greedy),
            }
        )

match_percent = (matches / total) * 100
print(f"Greedy matches optimal in {match_percent:.2f}% of configurations.")

# May delete this:
# Save mismatches for inspection (we will circle back to this later)
if diffs:
    pd.DataFrame(diffs).to_csv("data/mismatched_policy_rows.csv", index=False)
    print(f" Any mismatches saved to: data/mismatched_policy_rows.csv")
