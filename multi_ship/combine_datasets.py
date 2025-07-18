import os
import pandas as pd

# Config
INPUT_DIR = "data"
OUTPUT_FILE = "data/combined_T.csv"
PREFIX = "T_ship"
EXT = ".csv"


# Combine all T_ship*.csv into one
def combine_T_datasets():
    all_dfs = []
    for file in sorted(os.listdir(INPUT_DIR)):
        # only look at files that match the pattern
        if file.startswith(PREFIX) and file.endswith(EXT):
            print(f"Reading: {file}")
            df = pd.read_csv(os.path.join(INPUT_DIR, file))
            # keep track of which file each row came from
            df["ship_id"] = file
            all_dfs.append(df)

    if not all_dfs:
        print("No ship datasets found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCombined dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    combine_T_datasets()
