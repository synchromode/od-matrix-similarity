import argparse
import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from similarity.measure import measure

fprint = partial(print, flush=True)


def download_google_drive_file(url_share: str, destination: str | Path) -> None:
    file_id = re.search(r"https://drive.google.com/file/d/(.+?)(?:/.*|$)", url_share).group(1)
    
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    response = session.get(base_url, params={"id": file_id}, stream=True)
    
    m = re.findall(r"<input\s+type=\"hidden\"\s+name=\"((?:(?!\").)+)\"\s+value=\"((?:(?!\").)+)\">", response.text)
    
    base_url = "https://drive.usercontent.google.com/download"
    response = session.get(base_url, params={k: v for k, v in m}, stream=True)
    
    with open(destination, "wb") as file:
        for chunk in response.iter_content(8192):
            file.write(chunk)


def download_file(url: str, destination: str | Path) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as file:
            for chunk in response.iter_content(8192):
                file.write(chunk)


parser = argparse.ArgumentParser()
parser.add_argument("--source",       type=str,            required=True)
parser.add_argument("--target",       type=str,            default="target")
parser.add_argument("--filename",     type=str,            default="sim.json")
parser.add_argument("--google-drive", action="store_true", default=False)
parser.add_argument("--hourly",       action="store_true", default=False)
parser.add_argument("--zero-diag",    action="store_true", default=False)
args = parser.parse_args()

path_to_target = Path(args.target)
path_to_target.mkdir(parents=True, exist_ok=True)

fprint("Downloading ZIP file...", end=" ")
path_to_zip = path_to_target / "odm.zip"
if args.google_drive:
    # args.source should be the link obtained by sharing the file
    download_google_drive_file(args.source, path_to_zip)
else:
    # args.source should be the link to the file to download
    download_file(args.source, path_to_zip)
fprint("OK")

fprint("Loading OD matrices...", end=" ")
pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})$")
od_matrix = {}
odm_shape = None
archive = np.load(path_to_zip)
for path_to_odm, odm in archive.items():
    if path_to_odm.endswith("/"):
        continue
    m = re.search(pattern, path_to_odm)
    year  = int(m.group(1))
    month = int(m.group(2))
    day   = int(m.group(3))
    if odm_shape is None:
        odm_shape = odm.shape
    elif odm_shape != odm.shape:
        fprint("ERROR (different dimensions)")
        exit(1)
    if not args.hourly:
        odm = np.sum(odm, axis=0, keepdims=True)
    if args.zero_diag:
        odm[:, np.arange(odm_shape[-1]), np.arange(odm_shape[-1])] = 0
    od_matrix[(day, month, year)] = odm
if odm_shape is None:
    fprint("ERROR (no matrix found)")
    exit(1)
fprint("OK")

event_weight = jnp.ones([odm_shape[-1]])
sim_fct = jax.jit(partial(measure, weight=event_weight))
sim_fct(jnp.zeros((1, 1)), jnp.zeros((1, 1)))

df = None
for pi in range(24 if args.hourly else 1):
    tqdm_desc = "Computing similarities"
    if args.hourly:
        tqdm_desc += f" ({pi + 1:02d}/24)"
    output = []
    for (day1, month1, year1), od1 in tqdm(od_matrix.items(), desc=tqdm_desc):
        tens1 = jnp.asarray(od1[pi], dtype=jnp.float32)
        for (day2, month2, year2), od2 in od_matrix.items():
            tens2 = jnp.asarray(od2[pi], dtype=jnp.float32)
            similarity = sim_fct(tens1, tens2)
            output.append({
                "day1":       day1,
                "month1":     month1,
                "year1":      year1,
                "day2":       day2,
                "month2":     month2,
                "year2":      year2,
                "period":     pi,
                "similarity": similarity.item(),
                "total":      jnp.sum(tens1 + tens2).item(),
            } if args.hourly else {
                "day1":       day1,
                "month1":     month1,
                "year1":      year1,
                "day2":       day2,
                "month2":     month2,
                "year2":      year2,
                "similarity": similarity.item(),
            })
    if df is None:
        df = pd.DataFrame(output)
    else:
        df = pd.concat([df, pd.DataFrame(output)], ignore_index=True)
    
mean_sim = df["similarity"].mean()
std_sim  = df["similarity"].std()
df["similarity"] = (df["similarity"] - mean_sim) / std_sim

if args.hourly:
    df["tmp"]    = df.groupby(["day1", "month1", "year1", "day2", "month2", "year2"])["total"].transform("sum")
    df["weight"] = df["total"] / df["tmp"]

    df_p0 = df[df["period"] == 0]

    df_day = pd.DataFrame()
    df_day["day1"]   = df_p0["day1"]
    df_day["month1"] = df_p0["month1"]
    df_day["year1"]  = df_p0["year1"]
    df_day["day2"]   = df_p0["day2"]
    df_day["month2"] = df_p0["month2"]
    df_day["year2"]  = df_p0["year2"]
    with tqdm(total=df_day.shape[0], desc="Gathering ODMs into daily data") as pbar:
        def sim_pbar(row):
            s = df[
                (df["day1"] == row["day1"]) & (df["month1"] == row["month1"]) & (df["year1"] == row["year1"]) &
                (df["day2"] == row["day2"]) & (df["month2"] == row["month2"]) & (df["year2"] == row["year2"])
            ][["similarity", "weight"]].prod(axis=1).sum(axis=0)
            pbar.update(1)
            return s
        df_day["similarity"] = df_day.apply(sim_pbar, axis=1)
else:
    df_day = df

filename = args.filename
if args.hourly:
    filename = filename.replace(".json", "_hourly.json")
path_to_json = path_to_target / filename
df_day.to_json(path_to_json)
