from torch.utils.data import Dataset
from helper import get_file_paths

class SummaryDataset(Dataset):
    def __init__(self, text_folder, summary_folder):
        self.text_folder = text_folder
        self.summary_folder = summary_folder
        self.text_files = get_file_paths(text_folder)
        self.summary_files = get_file_paths(summary_folder)
        self.data = []
        for text_file, summary_file in zip(self.text_files, self.summary_files):
            with open(text_file, 'r') as f:
                text = f.read()
            with open(summary_file, 'r') as f:
                summary = f.read()
            self.data.append({'text': text, 'summary': summary})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]