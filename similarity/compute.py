import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from measure import gssi, measure, nlod
from tqdm import tqdm

fprint = partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data",            type=str,            required=True,    metavar="PATH_TO_FILE")
parser.add_argument("--target",          type=str,            default="target", metavar="PATH_TO_DIR")
parser.add_argument("--hourly",          action="store_true", default=False)
parser.add_argument("--zero-diag",       action="store_true", default=False)
sota = parser.add_mutually_exclusive_group(required=False)
sota.add_argument("--nlod", action="store_true", default=False)
sota.add_argument("--gssi", action="store_true", default=False)
misc = parser.add_argument_group()
misc.add_argument("--event-beach",     action="store_true", default=False)
misc.add_argument("--event-keukenhof", action="store_true", default=False)
misc.add_argument("--event-weight",    type=float,          default=2.0)
misc.add_argument("--event-indices",   type=str,                                metavar="PATH_TO_FILE")
misc.add_argument("--gssi-clusters",   type=str,                                metavar="PATH_TO_FILE")
args = parser.parse_args()

if (args.event_beach or args.event_keukenhof) and args.event_indices is None:
    parser.error("--event-beach and --event-keukenhof require --event-indices")
if args.gssi and args.gssi_clusters is None:
    parser.error("--gssi requires --gssi-clusters")

path_to_target = Path(args.target)
path_to_target.mkdir(parents=True, exist_ok=True)

od_matrix = {}
n_area = 0
fprint("Loading OD matrices...", end=" ")
for path_to_dir in Path(args.data).iterdir():
    month, year = map(int, path_to_dir.name.split("_"))
    for day in range(31):
        path_to_file = path_to_dir / f"{year}-{month:02d}-{day+1:02d}.npy"
        if not path_to_file.exists():
            continue
        m = np.load(path_to_file)
        if n_area == 0:
            n_area = m.shape[-1]
        elif n_area != m.shape[-1]:
            fprint("ERROR (different dimensions)")
            exit(1)
        if not args.hourly:
            m = np.sum(m, axis=0, keepdims=True)
        if args.zero_diag:
            m[:, np.arange(n_area), np.arange(n_area)] = 0
        od_matrix[(day + 1, month, year)] = m
if n_area > 0:
    fprint("OK")
else:
    fprint("ERROR (no matrix found)")
    exit(1)

if args.event_beach or args.event_keukenhof:
    with open(Path(args.event_indices), "r") as file:
        event = json.load(file)
idx_event = []
if args.event_beach:
    idx_event += event["beach"]["index"]
if args.event_keukenhof:
    idx_event += event["keukenhof"]["index"]
idx_event = jnp.asarray(idx_event)
event_weight = jnp.ones((n_area,))
if len(idx_event) > 0:
    event_weight = event_weight / args.event_weight
    event_weight = event_weight.at[idx_event].mul(2 * args.event_weight)

if args.nlod:
    sim_fct = jax.jit(nlod)
elif args.gssi:
    with open(Path("data", "south_holland", "gssi_clusters.json"), "r") as file:
        communes = json.load(file)
    clusters = [jnp.asarray(i) for i in communes.values()]
    sim_fct = jax.jit(partial(gssi, clusters=clusters))
    sim_fct(jnp.zeros([500, 500]), jnp.zeros([500, 500]))
else:
    sim_fct = jax.jit(partial(measure, weight=event_weight))
    sim_fct(jnp.zeros([1, 1]), jnp.zeros([1, 1]))

alg = "coretrip"
if args.nlod:
    alg = "nlod"
if args.gssi:
    alg = "gssi"
filename = "sim"
if args.zero_diag:
    filename += "0"
filename += f"_{alg}"
if args.hourly:
    filename += "_hourly"
else:
    filename += "_daily"
if args.event_beach:
    filename += "_beach"
if args.event_keukenhof:
    filename += "_keukenhof"
filename += ".csv"

path_to_csv = path_to_target / filename
if args.hourly:
    output_df = pd.DataFrame(columns=(
        "day1", "month1", "year1",
        "day2", "month2", "year2",
        "period", "similarity", "total1", "total2",
    ))
else:
    output_df = pd.DataFrame(columns=(
        "day1", "month1", "year1",
        "day2", "month2", "year2",
        "similarity",
    ))
output_df.to_csv(path_to_csv, index=False)

for pi in range(24 if args.hourly else 1):
    output_df = pd.DataFrame()
    tqdm_desc = "Computing similarities"
    if args.hourly:
        tqdm_desc += f" ({pi + 1:02d}/24)"
    for (day1, month1, year1), od1 in tqdm(od_matrix.items(), desc=tqdm_desc):
        output = []
        tens1 = jnp.asarray(od1[pi], dtype=jnp.float32)
        for (day2, month2, year2), od2 in od_matrix.items():
            tens2 = jnp.asarray(od2[pi], dtype=jnp.float32)
            similarity = sim_fct(tens1, tens2)
            output.append({
                "day1": day1,
                "month1": month1,
                "year1": year1,
                "day2": day2,
                "month2": month2,
                "year2": year2,
                "period": pi,
                "similarity": similarity.item(),
                "total1": jnp.sum(tens1).item(),
                "total2": jnp.sum(tens2).item(),
            } if args.hourly else {
                "day1": day1,
                "month1": month1,
                "year1": year1,
                "day2": day2,
                "month2": month2,
                "year2": year2,
                "similarity": similarity.item(),
            })
        output_df = pd.concat([output_df, pd.DataFrame(output)], ignore_index=True)
    output_df.to_csv(path_to_target / filename, mode="a", header=False, index=False)
