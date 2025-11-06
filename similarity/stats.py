import argparse
import datetime
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

YEAR = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "Obtober", "November", "December"]
WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
date2day = lambda d, m, y: (datetime.datetime(int(y), int(m), int(d)).weekday() + 1) % 7

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", type=str,            required=True)
parser.add_argument("--target", "-t", type=str,            required=False)
parser.add_argument("--year",         action="store_true", default=False)
parser.add_argument("--fontsize",     type=int,            default=18)
parser.add_argument("--markersize",   type=float,          default=10)
parser.add_argument("--linewidth",    type=float,          default=3)
parser.add_argument("--stdalpha",     type=float,          default=0.3)
args = parser.parse_args()

path_to_source = Path(args.source)
with open(path_to_source, "r") as file:
    algs = json.load(file)

n_elem = len(YEAR) if args.year else len(WEEK)
x_axis = list(range(n_elem))
fig, axs = plt.subplots(n_elem, 1, sharex=True, figsize=(n_elem, n_elem * 2))

min_y = 1.0
for alg in algs:
    path_to_data = Path(alg["location"])
    df = pd.read_csv(path_to_data)
    
    min_sim = df["similarity"].min()
    max_sim = df["similarity"].max()
    if alg["dissimilarity"]:
        df["similarity"] = (max_sim - df["similarity"]) / (max_sim - min_sim)
    else:
        df["similarity"] = (df["similarity"] - min_sim) / (max_sim - min_sim)
    
    if args.year:
        s1 = "month1"
        s2 = "month2"
    else:
        s1 = "dayweek1"
        s2 = "dayweek2"
        df[s1] = df.apply(lambda row: date2day(row["day1"], row["month1"], row["year1"]), axis=1)
        df[s2] = df.apply(lambda row: date2day(row["day2"], row["month2"], row["year2"]), axis=1)
    data_all = [
        [
            df[(df[s1] == i1) & (df[s2] == i2)]["similarity"]
            for i2 in x_axis
        ]
        for i1 in x_axis
    ]
    data_mean = np.asarray([[data_all[i1][i2].mean() for i2 in x_axis] for i1 in x_axis])
    data_std  = np.asarray([[data_all[i1][i2].std()  for i2 in x_axis] for i1 in x_axis])
    min_y = min(min_y, np.min(data_mean - data_std))
    
    for i, ax in enumerate(axs):
        p = ax.plot(x_axis, data_mean[i], "-o",
                    markersize=args.markersize,
                    linewidth=args.linewidth,
                    zorder=2,
                    label=alg["name"])
        ax.fill_between(x_axis, data_mean[i] - data_std[i], data_mean[i] + data_std[i],
                        color=p[0].get_color(),
                        alpha=args.stdalpha,
                        zorder=1)

for i, ax in enumerate(axs):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(min_y, 1.1)
    if args.year:
        ax.set_ylabel(YEAR[i], fontsize=args.fontsize)
        ax.set_xticks(x_axis, [m[:3] for m in YEAR], fontsize=args.fontsize)
    else:
        ax.set_ylabel(WEEK[i], fontsize=args.fontsize)
        ax.set_xticks(x_axis, [d[:3] for d in WEEK], fontsize=args.fontsize)
    ax.xaxis.set_ticks_position("none")
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc="lower center", ncol=len(algs), fontsize=args.fontsize, bbox_to_anchor=(0.5, 1))
fig.tight_layout()

if args.target is None:
    alg_names = [re.compile("[\W_]+").sub("", a["name"]).lower() for a in algs]
    path_to_target = path_to_source.parent / f"stats_{'_'.join(alg_names)}_{'year' if args.year else 'week'}.png"
else:
    path_to_target = Path(args.target)
fig.savefig(path_to_target, bbox_extra_artists=[legend], bbox_inches="tight")
