import argparse
import datetime
from pathlib import Path

import networkx as nx
import pandas as pd

YEAR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
date2day = lambda d, m, y: WEEK[(datetime.datetime(int(y), int(m), int(d)).weekday() + 1) % 7]

KEUKENHOF_OPEN = {
    2023: ((3, 23), (5, 14)),
    2024: ((3, 21), (5, 12)),
    2025: ((3, 20), (5, 11)),
}

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s",  type=str,            required=True)
parser.add_argument("--target", "-t",  type=str,            required=False)
parser.add_argument("--weather",       type=str)
parser.add_argument("--visitors",      type=str)
parser.add_argument("--keukenhof",     action="store_true", default=False)
parser.add_argument("--prune",         action="store_true", default=False)
parser.add_argument("--dissimilarity", action="store_true", default=False)
args = parser.parse_args()

path_to_source = Path(args.source)
df = pd.read_csv(path_to_source)

df["dayweek1"] = df.apply(lambda row: date2day(row["day1"], row["month1"], row["year1"]), axis=1)
df["dayweek2"] = df.apply(lambda row: date2day(row["day2"], row["month2"], row["year2"]), axis=1)

min_sim = df["similarity"].min()
max_sim = df["similarity"].max()
if args.dissimilarity:
    df["similarity"] = (df["similarity"] - min_sim) / (max_sim - min_sim)
else:
    df["similarity"] = (max_sim - df["similarity"]) / (max_sim - min_sim)

lower_bound = 0.0
if args.prune:
    q1 = df["similarity"].quantile(0.25)
    q3 = df["similarity"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
else:
    upper_bound = 1.0

date2node = {}
node2date = {}
graph = nx.Graph()
for row in df.itertuples():
    date2node.setdefault((row.day1, row.month1, row.year1), len(date2node))
    u = date2node[(row.day1, row.month1, row.year1)]
    node2date.setdefault(u, (row.day1, row.month1, row.year1))
    date2node.setdefault((row.day2, row.month2, row.year2), len(date2node))
    v = date2node[(row.day2, row.month2, row.year2)]
    node2date.setdefault(v, (row.day2, row.month2, row.year2))
    graph.add_node(
        u,
        label=f"{row.day1:02d}/{row.month1:02d}/{row.year1}",
        day=date2day(row.day1, row.month1, row.year1),
        month=YEAR[row.month1 - 1],
    )
    graph.add_node(
        v,
        label=f"{row.day2:02d}/{row.month2:02d}/{row.year2}",
        day=date2day(row.day2, row.month2, row.year2),
        month=YEAR[row.month2 - 1],
    )
    if u == v or row.similarity < lower_bound or row.similarity > upper_bound:
        continue
    graph.add_edge(u, v, weight=1/row.similarity)
num_nodes = len(date2node)

if args.weather is not None:
    weither_df = pd.read_csv(args.weather, sep=";")
    attr_weather = {}
    attr_temperature = {}
    for row in weither_df.itertuples():
        d, m, y = map(int, row.date.split("/"))
        n = date2node[(d, m, y)]
        attr_weather[n]     = row.weather
        attr_temperature[n] = row.temperature
    nx.set_node_attributes(graph, attr_weather,     "weather")
    nx.set_node_attributes(graph, attr_temperature, "temperature")

if args.visitors is not None:
    visitors_df = pd.read_csv(args.visitors, sep=";")
    d2v = {}
    for row in visitors_df.itertuples():
        d, m, y = map(int, row.date.split("/"))
        d2v[(d, m, y)] = row.number
    min_v = min(d2v.values())
    attr_visitors = {}
    for n, (d, m, y) in node2date.items():
        attr_visitors[n] = d2v.get((d, m, y), 0)
    nx.set_node_attributes(graph, attr_visitors, "visitors")

if args.keukenhof:
    attr_keukenhof = {}
    for n, (d, m, y) in node2date.items():
        attr_keukenhof[n] = KEUKENHOF_OPEN[y][0] <= (m, d) <= KEUKENHOF_OPEN[y][1]
    nx.set_node_attributes(graph, attr_keukenhof, "keukenhof")

if args.target is None:
    target_stem = path_to_source.stem.replace("norm_sim", "graph")
    if args.prune:
        target_stem += "_pruned"
    path_to_target = path_to_source.parent / f"{target_stem}.gexf"
else:
    path_to_target = Path(args.target)
nx.write_gexf(graph, path_to_target)
