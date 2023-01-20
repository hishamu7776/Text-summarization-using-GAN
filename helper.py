import os
import glob
import pandas as pd


def read_data(text_path, summary_path):
    # Create an empty list to store the examples
    summary_data = []
    for text_folder, summary_folder in zip(os.listdir(text_path), os.listdir(summary_path)):
        for text_file, summary_file in zip(os.listdir(os.path.join(text_path, text_folder)), os.listdir(os.path.join(summary_path, summary_folder))):
            with open(os.path.join(text_path, text_folder, text_file), 'r') as text_f:
                text = text_f.read()
            with open(os.path.join(summary_path, summary_folder, summary_file), 'r') as summary_f:
                summary = summary_f.read()
            # Create a new example and add it to the examples list
            summary_data.append((text, summary))
    return summary_data
        
