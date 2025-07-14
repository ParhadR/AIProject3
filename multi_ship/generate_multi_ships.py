import os
import csv
from src.map_generator import generate_ship
from src.mdp_solver import value_iteration, export_T_to_csv


NUM_SHIPS = 5  
D = 30
P_FACTOR = 0.5
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate multiple ships and export T-values

def generate_multiple_T_datasets():
    for i in range(1, NUM_SHIPS + 1):
        print(f"\n🚢 Generating ship {i}/{NUM_SHIPS}...")
        ship_map = generate_ship(D, P_FACTOR)

        print(f"🔄 Running value iteration on ship {i}...")
        T = value_iteration(ship_map)

        csv_filename = f"T_ship{i}.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)

        print(f"💾 Exporting T-values to {csv_filename}...")
        export_T_to_csv(T, ship_map, filepath=csv_path)

    print(f"\nGenerated and saved T-values for {NUM_SHIPS} ships.")

if __name__ == "__main__":
    generate_multiple_T_datasets()
