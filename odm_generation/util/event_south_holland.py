from pathlib import Path
import json

path_to_dir = Path("data", "south_holland")

beach = [
    "2554", "2566", "2583", "2584", "2586", # Den Haag
    "2681", "2684", "2691",                 # Westland
    "3151",                                 # Rotterdam
    "2242",                                 # Wassenaar
    "2221", "2225",                         # Katwijk
    "2202", "2204",                         # Noordwijk
]
keukenhof = ["2161", "2163"]

with open(Path(path_to_dir / "area2index.json"), "r") as file:
    a2i = json.load(file)

events = {
    "beach": {
        "area":  [a      for a in beach if a in a2i],
        "index": [a2i[a] for a in beach if a in a2i],
    },
    "keukenhof": {
        "area":  [a      for a in keukenhof if a in a2i],
        "index": [a2i[a] for a in keukenhof if a in a2i],
    },
}

with open(Path(path_to_dir / "events.json"), "w") as file:
    json.dump(events, file)