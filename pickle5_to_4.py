import pandas as pd
import os

for root, dirs, files in os.walk("data"):
    for filename in files:
        if not filename.endswith(".pkl.compressed"):
            continue
        file_path = os.path.join(root, filename)
        data = pd.read_pickle(file_path, compression='gzip')
        data.to_pickle(file_path + "_v4", protocol=4, compression="gzip")
