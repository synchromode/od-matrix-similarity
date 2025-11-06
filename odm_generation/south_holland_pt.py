from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

path_to_dir = Path("data", "south_holland")

path_to_data   = path_to_dir / "public_transport.csv"
path_to_target = path_to_dir / "odm_pt"
path_to_target.mkdir(parents=True, exist_ok=True)

pt_df = pd.read_csv(path_to_data, sep=";")
for row in tqdm(pt_df.itertuples(), desc="Computing OD matrices"):
    d, m, y = map(int, row.date.split("/"))
    path_to_target_dir = path_to_target / f"{m:02d}_{y:04d}"
    path_to_target_dir.mkdir(parents=True, exist_ok=True)
    
    odm = np.zeros([1, 5, 5])
    odm[0, 0, 1] = odm[0, 1, 0] = row.amsterdam
    odm[0, 0, 2] = odm[0, 2, 0] = row.schiphol_airport
    odm[0, 0, 3] = odm[0, 3, 0] = row.leiden_central_station
    odm[0, 0, 4] = odm[0, 4, 0] = row.haarlem_station
    
    path_to_file = path_to_target_dir / f"{y:04d}-{m:02d}-{d:02d}.npy"
    np.save(path_to_file, odm)
