import argparse
import json
import math
from functools import partial
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fprint = partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--source",   type=str, required=True)
parser.add_argument("--days",     type=str, required=True, action="store", nargs="+", metavar="DD/MM/YYYY")
parser.add_argument("--target",   type=str, default="target")
parser.add_argument("--geometry", type=str, default="georef-netherlands-postcode-pc4.geojson")
parser.add_argument("--areas",    type=str, default="area2index.json")
parser.add_argument("--events",   type=str, default="events.json")
_direction = parser.add_mutually_exclusive_group()
_direction.add_argument("--origins",      action="store_true", default=False)
_direction.add_argument("--destinations", action="store_true", default=False)
_event = parser.add_mutually_exclusive_group()
_event.add_argument("--event-beach",     action="store_true", default=False)
_event.add_argument("--event-keukenhof", action="store_true", default=False)
_plot = parser.add_argument_group()
_plot.add_argument("--nrow",      type=int)
_plot.add_argument("--ncol",      type=int)
_plot.add_argument("--factor",    type=float,          default=3.0)
_plot.add_argument("--lgd-width", type=float,          default=0.15)
_plot.add_argument("--log-scale", action="store_true", default=False)
args = parser.parse_args()

if (args.event_beach or args.event_keukenhof) and args.events is None:
    parser.error("--event-beach and --event-keukenhof require --events")

path_to_source = Path(args.source)

fprint("Loading area mapping...", end=" ")
df: gpd.GeoDataFrame = gpd.read_file(path_to_source / args.geometry)
fprint("OK")

odms = []
days = []
for day in args.days:
    d, m, y = day.split("/")
    path_to_file = path_to_source / "odm" / f"{m}_{y}" / f"{y}-{m}-{d}.npy"
    if not path_to_file.exists():
        print(f"File {path_to_file.as_posix()} does not exist")
        continue
    odms.append(np.sum(np.load(path_to_file), axis=0))
    days.append(day)
odms = np.stack(odms, axis=0)

with open(path_to_source / args.areas, "r") as file:
    a2i: dict = json.load(file)

if args.event_beach or args.event_keukenhof:
    with open(path_to_source / args.events, "r") as file:
        event = json.load(file)

if args.event_beach:
    idx = event["beach"]["index"]
elif args.event_keukenhof:
    idx = event["keukenhof"]["index"]
else:
    idx = list(range(odms.shape[-1]))
i2x = np.sum(odms[:, :, idx], axis=2)
x2i = np.sum(odms[:, idx, :], axis=1)

o_max = np.max(i2x)
d_max = np.max(x2i)
vmax = o_max + d_max
if args.origins:
    vmax -= d_max
if args.destinations:
    vmax -= o_max
if args.log_scale:
    vmax = np.log1p(vmax)

def color_fct(row, i):
    o = i2x[i, a2i[row.pc4_code]]
    d = x2i[i, a2i[row.pc4_code]]
    c = o + d
    if args.origins:
        c -= d
    if args.destinations:
        c -= o
    if args.log_scale:
        c = np.log1p(c)
    return c

df = df[df["pc4_code"].isin(a2i.keys())]

n_odm = odms.shape[0]
if args.nrow is None:
    n_row = int(math.sqrt(n_odm))
else:
    n_row = args.nrow
if args.ncol is None:
    n_col = (n_odm // n_row) + (n_odm % n_row > 0)
else:
    n_col = args.ncol
    if args.nrow is None:
        n_row = (n_odm // n_col) + (n_odm % n_col > 0)

fig, axs = plt.subplots(n_row, n_col, figsize=(args.factor * n_col / (1 - args.lgd_width), args.factor * n_row))
for i, (day, ax) in enumerate(zip(days, axs.flatten())):
    df["color"] = df.apply(lambda row: color_fct(row, i), axis=1)
    df.plot(column="color", ax=ax, vmax=vmax)
    ax.set_title(day)
    ax.set_xticks([])
    ax.set_yticks([])
for _, ax in zip(range(n_odm, n_row * n_col), reversed(axs.flatten())):
    ax.remove()
fig.tight_layout()
fig.subplots_adjust(right=(1 - args.lgd_width)) 
ax = fig.add_axes([(1 - args.lgd_width), 0.15, 0.05, 0.7])
lgd_label = "Number of trips" + " (log)" if args.log_scale else ""
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=vmax)),
             cax=ax, orientation="vertical", label=lgd_label)
fig.savefig(Path(args.target, "map_odms.png"))
