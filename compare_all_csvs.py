import os
import pandas as pd
import glob
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def search_frame(df, w, s, c):
    a = df["width"] == w
    b = df["samples"] == s
    c = df["complexity"] == c

    return df[a & b & c]


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None  # default='warn'

    # Read all csvs into pandas data frame
    file_name = "*output*.csv"

    all_csvs = glob.glob(file_name)

    all_frames = []

    print(all_csvs)

    for csv in all_csvs:
        data_type = csv.split("_")[0]
        frame = pd.read_csv(csv, names=["width", "samples", "complexity", "time"])
        frame["type"] =[data_type] * len(frame["width"])
        all_frames.append(frame)

    result = pd.concat(all_frames, ignore_index=True)

    result["time"][result["type"] == "mega"] = result["time"][result["type"] == "mega"] * 1.5

    result.to_csv("mid_csv.csv")

    # result = pd.read_csv("final_results.csv", header=0)

    new_data = []

    for width in range(5, 12, 2):
        for sample_dec in range(1, 10, 2):
            for complexity in range(1, 9):
                search_res = search_frame(result, 2 ** width, sample_dec * 8, complexity)
                best_types = search_res["type"][search_res["time"].idxmin()]

                if "queue" in best_types:
                    best_type = "queue"
                elif "bitmasked" in best_types:
                    best_type = "bitmasked"
                else:
                    best_type = "mega"

                next_row = {"width": 2 ** width, "samples": sample_dec * 8, "complexity": complexity, "type": best_type}

                new_data.append(next_row)

    out_df = pd.DataFrame(new_data)

    out_df.to_csv("type_data_final.csv")
