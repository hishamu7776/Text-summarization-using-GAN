import configparser
import os
import spacy
from helper import replace_contraction
from torchdata.datapipes.iter import IterableWrapper, FileLister, FileOpener, StreamReader



class Dataset:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.cfg')
        self.spacy = spacy.load("en_core_web_sm")
        self.stop_words = list(self.spacy.Defaults.stop_words)  
        self.read_data()
        self.preprocess()

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
            self.dataset = documents.zip(summaries)
        
        elif self.config['DEFAULT']['type'] == 'CSV':
            datapipe = FileLister(root=self.config['DEFAULT']['csv_folder'])
            datapipe = IterableWrapper(datapipe)
            datapipe = FileOpener(datapipe, mode='t', encoding = self.config['DEFAULT']['encoding'])
            summary_dp = datapipe.parse_csv(skip_lines=0)
            if self.config['DEFAULT']['header'] == 'yes':
                summary_dp = summary_dp.drop(0)
            self.dataset = summary_dp
        
        else:
            raise TypeError("Data type should be TEXT or CSV.")
        
    def preprocess(self):
        self.summary_data = []
        for text,summary in list(self.dataset)[:10]:
            text = self.replace_contraction(text)
            summary = self.replace_contraction(summary)
            text = self.spacy(text)
            summary = self.spacy(summary)
            text = ' '.join([token.text for token in text if not (token.is_punct or token.is_stop)])
            summary = ' '.join([token.text for token in summary if not (token.is_punct or token.is_stop)])
            self.summary_data.append((text,summary))
    
    @staticmethod
    def select_text(x):
        return x[1]
    
    @staticmethod
    def replace_contraction(doc):
        contractions_map = {"ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'll": 'I will', "I'm": 'I am', "I've": 'I have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it will', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "must've": 'must have', "mustn't": 'must not', "needn't": 'need not', "oughtn't": 'ought not', "shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "that'd": 'that would', "that's": 'that is', "there'd": 'there would', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "wasn't": 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where'd": 'where did', "where's": 'where is', "who'll": 'who will', "who's": 'who is', "won't": 'will not', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have', "how'd'y": 'how do you', "I'd've": 'I would have', "I'll've": 'I will have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "it'd've": 'it would have', "it'll've": 'it will have', "mightn't've": 'might not have', "mustn't've": 'must not have', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't've": 'ought not have', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd've": 'she would have', "she'll've": 'she will have', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "this's": 'this is', "that'd've": 'that would have', "there'd've": 'there would have', "here's": 'here is', "they'd've": 'they would have', "they'll've": 'they will have', "to've": 'to have', "we'd've": 'we would have', "we'll've": 'we will have', "what'll've": 'what will have', "when's": 'when is', "when've": 'when have', "where've": 'where have', "who'll've": 'who will have', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't've": 'will not have', "would've": 'would have', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd've": 'you would have', "you'll've": 'you will have'}
        for c_text in contractions_map.keys():
            if c_text in doc:
                doc.replace(c_text,contractions_map[c_text])
        return doc
