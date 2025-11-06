import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

YEAR = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "Obtober", "November", "December"]
WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
date2day = lambda y, m, d: WEEK[(datetime.datetime(int(y), int(m), int(d)).weekday() + 1) % 7]
date2str = lambda y, m, d: f"{YEAR[m-1][:3]} {d}, {y} ({date2day(y, m, d)[:3]})"

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s",  type=str,            required=True)
parser.add_argument("--target", "-t",  type=str,            required=False)
parser.add_argument("--dissimilarity", action="store_true", default=False)
args = parser.parse_args()

path_to_source = Path(args.source)
df = pd.read_csv(path_to_source)

min_sim = df["similarity"].min()
max_sim = df["similarity"].max()
if args.dissimilarity:
    df["similarity"] = (max_sim - df["similarity"]) / (max_sim - min_sim)
else:
    df["similarity"] = (df["similarity"] - min_sim) / (max_sim - min_sim)

df["x"] = df[["year1", "month1", "day1"]].apply(lambda x: tuple(x), axis=1)
df["y"] = df[["year2", "month2", "day2"]].apply(lambda y: tuple(y), axis=1)
df = df[["x", "y", "similarity"]]

x2i = {x: i for i, x in enumerate(sorted(df["x"].unique()))}
y2j = {y: j for j, y in enumerate(sorted(df["y"].unique()))}

holiday = {
    (2024,  1,  1),
    (2024,  3, 29),
    (2024,  3, 31),
    (2024,  4,  1),
    (2024,  4, 27),
    (2024,  5,  9),
    (2024,  5, 19),
    (2024,  5, 20),
    (2024, 12, 25),
    (2024, 12, 26),
}

xh = sorted(holiday & x2i.keys())
yh = sorted(holiday & y2j.keys())

a = np.empty([len(x2i), len(y2j)])
for row in df.itertuples():
    a[x2i[row.x], y2j[row.y]] = row.similarity


plt.imshow(a, cmap="viridis")
plt.gca().invert_yaxis()
plt.xticks([x2i[x] for x in xh], [])
plt.yticks([y2j[y] for y in yh], [])
plt.tick_params(color="red", length=5)
plt.xlabel("public holidays", color="red")
plt.ylabel("public holidays", color="red")
plt.tight_layout()

if args.target is None:
    path_to_target = path_to_source.parent / f"{path_to_source.stem.replace('norm_sim', 'heatmap')}.png"
else:
    path_to_target = Path(args.target)
plt.savefig(path_to_target)
plt.show()
