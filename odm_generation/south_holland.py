import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from tqdm import tqdm

AREAS = [
    "Woerden",
    "Nieuwkoop",
    "Katwijk",
    "Voorschoten",
    "Leidschendam-Voorburg",
    "Leiden",
    "Zoetermeer",
    "Bodegraven-Reeuwijk",
    "Haarlemmermeer",
    "Alphen aan den Rijn",
    "Lisse",
    "Teylingen",
    "Noordwijk",
    "Kaag en Braassem",
    "Zoeterwoude",
    "Oegstgeest",
    "Leiderdorp",
    "Hillegom",
    "Wassenaar",
]

path_to_dir = Path("data", "south_holland")

print("Loading segment mapping...", end=" ")
segment_to_coord: gpd.GeoDataFrame = gpd.read_file(path_to_dir / "basemap_2024_pzh.gpkg")
segment_to_coord.sort_values(by=["segmentID"], inplace=True)
print("OK")

print("Loading area mapping...", end=" ")
coord_to_area: gpd.GeoDataFrame = gpd.read_file(path_to_dir / "georef-netherlands-postcode-pc4.geojson")
print("OK")

transformer = Transformer.from_crs("epsg:3857", "epsg:4326")
def get_lat_lon(segment_id: int, last: bool=False) -> Tuple[float, float]:
    if segment_id not in segment_to_coord["segmentID"].values:
        print(segment_id, type(segment_id))
        return 0., 0.
    line = segment_to_coord[segment_to_coord["segmentID"] == segment_id]["geometry"].iloc[0]
    easting, northing = line.coords[int(last)]
    return transformer.transform(easting, northing)

coord_to_area = coord_to_area[coord_to_area["gem_name"].isin(AREAS)]
areas = coord_to_area["pc4_code"].tolist()
areas.append("none")
n_area = len(areas)
a2i = {a: i for i, a in enumerate(areas)}

with open(path_to_dir / "area2index.json", "w") as file:
    json.dump(a2i, file)

def get_area_idx(lat: float, lon: float) -> int:
    point = Point(lon, lat)
    for row in coord_to_area.itertuples():
        if row.geometry.contains(point):
            return a2i[row.pc4_code]
    return a2i["none"]

def get_period(timestamp: str) -> int:
    return int(timestamp.split()[1].split(":")[0])

@dataclass
class TripData:
    period: int
    origin: int
    destination: int
    complete: bool = False


def compute_od_matrix(trips: pd.DataFrame) -> np.ndarray:
    od_matrix = np.zeros((24, n_area, n_area), dtype=int)
    trip_data = {}
    last_id = -1
    for i in tqdm(range(trips.shape[0]), desc="Computing OD matrices"):
        trip_id = trips["TripID"].iloc[i]
        if trip_id == last_id:
            continue
        
        if i > 0:
            latitude, longitude = get_lat_lon(trips["SegmentId"].iloc[i-1], True)
            area_idx = get_area_idx(latitude, longitude)
            trip = trip_data[last_id]
            od_matrix[trip.period, trip.origin, trip.destination] -= 1
            trip.destination = area_idx
            od_matrix[trip.period, trip.origin, trip.destination] += 1
        
        latitude, longitude = get_lat_lon(trips["SegmentId"].iloc[i])
        area_idx = get_area_idx(latitude, longitude)
        period = get_period(trips["LocalTimeStamp"].iloc[i])
        trip_data.setdefault(trip_id, TripData(period, area_idx, area_idx))
        trip = trip_data[trip_id]
        if trip.complete:
            od_matrix[trip.period, trip.origin, trip.destination] -= 1
            trip.destination = area_idx
        else:
            trip.complete = True
        od_matrix[trip.period, trip.origin, trip.destination] += 1
        
        last_id = trip_id
    
    return od_matrix

def odm_to_df(odm: np.ndarray) -> pd.DataFrame:
    data = []
    idx = np.where(odm)
    for p, o, d in zip(*idx):
        data.append({
            "period": p,
            "origin": o,
            "destination": d,
            "trips": odm[p, o, d],
        })
    return pd.DataFrame(data)

for path_to_datadir in (path_to_dir / "raw").iterdir():
    month, year = map(int, path_to_datadir.name.split("_"))
    path_to_target = path_to_dir / Path("odm", f"{month:02d}_{year}")
    path_to_target.mkdir(parents=True, exist_ok=True)
    for path_to_data in path_to_datadir.iterdir():
        day = int(path_to_data.stem.split("-")[-1])
        path_to_file = path_to_target / f"{year}-{month:02d}-{day:02d}.npy"
        if path_to_file.exists():
            continue
        print(f"Loading trips from {path_to_data.as_posix()}...", end=" ")
        trips = pd.read_csv(path_to_data, sep=";")
        print("OK")
        print("Sorting trips...", end=" ")
        trips.sort_values(by=["TripID", "LocalTimeStamp"], inplace=True)
        print("OK")
        od_matrix = compute_od_matrix(trips)
        print(f"Saving OD matrix to {path_to_file.as_posix()}...", end=" ")
        np.save(path_to_file, od_matrix)
        print("OK")
