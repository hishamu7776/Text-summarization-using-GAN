import os
import glob
import pandas as pd


def get_file_paths(folder):
    file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
