import json
from functools import partial
from pathlib import Path

import geopandas as gpd

fprint = partial(print, flush=True)

path_to_dir = Path("data", "south_holland")

fprint("Loading area mapping...", end=" ")
df: gpd.GeoDataFrame = gpd.read_file(path_to_dir / "georef-netherlands-postcode-pc4.geojson")
fprint("OK")

with open(path_to_dir / "area2index.json", "r") as file:
    a2i = json.load(file)

clusters = {}
for row in df.itertuples():
    if row.pc4_code not in a2i:
        continue
    clusters.setdefault(row.gem_name, [])
    clusters[row.gem_name].append(a2i[row.pc4_code])

with open(path_to_dir / "gssi_clusters.json", "w") as file:
    json.dump(clusters, file)
