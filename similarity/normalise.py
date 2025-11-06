import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", type=str,            required=True)
parser.add_argument("--target", "-t", type=str,            required=False)
parser.add_argument("--hourly",       action="store_true", default=False)
args = parser.parse_args()

path_to_source = Path(args.source)
df = pd.read_csv(path_to_source)

mean_sim = df["similarity"].mean()
std_sim  = df["similarity"].std()
df["similarity"] = (df["similarity"] - mean_sim) / std_sim

if args.hourly:
    df["total"] = df.groupby(["day1", "month1", "year1", "day2", "month2", "year2"])[["total1", "total2"]].transform("sum").sum(axis=1)
    df["weight"] = (df["total1"] + df["total2"]) / df["total"]

    df_p0 = df[df["period"] == 0]

    ndf = pd.DataFrame()
    ndf["day1"]   = df_p0["day1"]
    ndf["month1"] = df_p0["month1"]
    ndf["year1"]  = df_p0["year1"]
    ndf["day2"]   = df_p0["day2"]
    ndf["month2"] = df_p0["month2"]
    ndf["year2"]  = df_p0["year2"]
    with tqdm(total=ndf.shape[0], desc="Gathering ODMs into daily data") as pbar:
        def sim_pbar(row):
            s = df[
                (df["day1"] == row["day1"]) & (df["month1"] == row["month1"]) & (df["year1"] == row["year1"]) &
                (df["day2"] == row["day2"]) & (df["month2"] == row["month2"]) & (df["year2"] == row["year2"])
            ][["similarity", "weight"]].prod(axis=1).sum(axis=0)
            pbar.update(1)
            return s
        ndf["similarity"] = ndf.apply(sim_pbar, axis=1)
else:
    ndf = df

if args.target is None:
    path_to_target = path_to_source.parent / f"norm_{path_to_source.name}"
else:
    path_to_target = Path(args.target)
ndf.to_csv(path_to_target)