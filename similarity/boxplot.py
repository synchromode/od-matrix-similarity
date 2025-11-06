import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
date2day = lambda d, m, y: WEEK[(datetime.datetime(int(y), int(m), int(d)).weekday() + 1) % 7]

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s",  type=str,            required=True)
parser.add_argument("--target", "-t",  type=str,            required=False)
parser.add_argument("--dissimilarity", action="store_true", default=False)
parser.add_argument("--fontsize",      type=int,            default=18)
args = parser.parse_args()

path_to_source = Path(args.source)
df = pd.read_csv(path_to_source)

df["dayweek1"] = df.apply(lambda row: date2day(row["day1"], row["month1"], row["year1"]), axis=1)
df["dayweek2"] = df.apply(lambda row: date2day(row["day2"], row["month2"], row["year2"]), axis=1)

min_sim = df["similarity"].min()
max_sim = df["similarity"].max()
if args.dissimilarity:
    df["similarity"] = (max_sim - df["similarity"]) / (max_sim - min_sim)
else:
    df["similarity"] = (df["similarity"] - min_sim) / (max_sim - min_sim)

data = [
    [
        df[(df["dayweek1"] == day1) & (df["dayweek2"] == day2)]["similarity"]
        for day2 in WEEK
    ]
    for day1 in WEEK
]

n_elem = len(WEEK)
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

data_q1 = np.asarray([[data[i][j].quantile(0.25) for i in range(n_elem)] for j in range(n_elem)])
data_q3 = np.asarray([[data[i][j].quantile(0.75) for i in range(n_elem)] for j in range(n_elem)])
data_iqr = data_q3 - data_q1

fig, axs = plt.subplots(n_elem, 1, sharex=True, figsize=(n_elem * 0.5, n_elem * 2))
for i, ax in enumerate(axs):
    bplot = ax.boxplot(data[i], notch=True, flierprops={"markersize": 0}, patch_artist=True)
    for patch, colour in zip(bplot["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(colours[i])
    ax.set_ylim(np.min(data_q1[i] - 1.5 * data_iqr[i]), np.max(data_q3[i] + 1.5 * data_iqr[i]))
    ax.set_ylabel(WEEK[i], fontsize=args.fontsize, color=colours[i])
    ax.set_xticks(range(1, n_elem + 1), WEEK, rotation=90, fontsize=args.fontsize)
    ax.tick_params(colors=colours[i], which="major")
    ax.xaxis.set_ticks_position("none")
    for tick, colour in zip(ax.get_xticklabels(), colours):
        tick.set_color(colour)
fig.tight_layout()

if args.target is None:
    path_to_target = path_to_source.parent / f"{path_to_source.stem.replace('norm_sim', 'boxplot')}.png"
else:
    path_to_target = Path(args.target)
fig.savefig(path_to_target)
plt.show()
