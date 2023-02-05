import configparser
import os
from torchdata.datapipes.iter import IterableWrapper, FileLister, FileOpener, StreamReader



class Dataset:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.cfg')
        self.dataset = self.read_data()
        print(list(self.dataset))



    def read_data(self):
        if self.config['DEFAULT']['type'] == 'TEXT':
            summary_path = os.path.join(self.config['DEFAULT']['root'],self.config['DEFAULT']['summary_folder'])
            text_path = os.path.join(self.config['DEFAULT']['root'],self.config['DEFAULT']['document_folder'])
            documents = FileLister(root=text_path, recursive=True)
            documents = IterableWrapper(documents)
            documents = FileOpener(documents, mode='t', encoding = self.config['DEFAULT']['encoding'])
            documents = StreamReader(documents)
            documents = documents.map(self.select_text)                
            
            summaries = FileLister(root=summary_path, recursive=True)
            summaries = IterableWrapper(summaries)
            summaries = FileOpener(summaries, mode='t', encoding = self.config['DEFAULT']['encoding'])
            summaries = StreamReader(summaries)
            summaries = summaries.map(self.select_text)  
            
            return documents.zip(summaries)
        elif self.config['DEFAULT']['type'] == 'CSV':
            datapipe = FileLister(root=self.config['DEFAULT']['csv_folder'])
            datapipe = IterableWrapper(datapipe)
            datapipe = FileOpener(datapipe, mode='t', encoding = self.config['DEFAULT']['encoding'])
            summary_dp = datapipe.parse_csv(skip_lines=0)
            if self.config['DEFAULT']['header'] == 'yes':
                summary_dp = summary_dp.drop(0)
            return summary_dp
        else:
            raise TypeError("Data type should be TEXT or CSV.")
    @staticmethod
    def select_text(x):
        return x[1]
