import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--hourly", action="store_true", default=False)
args = parser.parse_args()

od_matrix: np.ndarray = np.load(args.file)

if args.hourly:
    figure, axis = plt.subplots(4, 6)
    for p, ax in enumerate(axis.flatten()):
        ax.imshow(od_matrix[p])
else:
    plt.imshow(np.sum(od_matrix, axis=0))
plt.colorbar()
plt.show()
