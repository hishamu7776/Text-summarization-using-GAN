import os
import re
import spacy
spacy_en = spacy.load("en_core_web_sm")

contractions_map = {"ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'll": 'I will', "I'm": 'I am', "I've": 'I have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it will', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "must've": 'must have', "mustn't": 'must not', "needn't": 'need not', "oughtn't": 'ought not', "shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "that'd": 'that would', "that's": 'that is', "there'd": 'there would', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "wasn't": 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where'd": 'where did', "where's": 'where is', "who'll": 'who will', "who's": 'who is', "won't": 'will not', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have', "how'd'y": 'how do you', "I'd've": 'I would have', "I'll've": 'I will have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "it'd've": 'it would have', "it'll've": 'it will have', "mightn't've": 'might not have', "mustn't've": 'must not have', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't've": 'ought not have', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd've": 'she would have', "she'll've": 'she will have', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "this's": 'this is', "that'd've": 'that would have', "there'd've": 'there would have', "here's": 'here is', "they'd've": 'they would have', "they'll've": 'they will have', "to've": 'to have', "we'd've": 'we would have', "we'll've": 'we will have', "what'll've": 'what will have', "when's": 'when is', "when've": 'when have', "where've": 'where have', "who'll've": 'who will have', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't've": 'will not have', "would've": 'would have', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd've": 'you would have', "you'll've": 'you will have'}
def read_data(text_path, summary_path):
    # Create a list to store the summary data
    summary_data = list()
    for text_folder, summary_folder in zip(os.listdir(text_path), os.listdir(summary_path)):
        for text_file, summary_file in zip(os.listdir(os.path.join(text_path, text_folder)), os.listdir(os.path.join(summary_path, summary_folder))):
            with open(os.path.join(text_path, text_folder, text_file), 'r') as text_f:
                text = text_f.read()
            with open(os.path.join(summary_path, summary_folder, summary_file), 'r') as summary_f:
                summary = summary_f.read()
            # Add summary data to the list
            summary_data.append((text, summary))
    return summary_data
        
def preprocess(text):
    #Convert to lower case
    text = text.lower()
    for contraction, expansion in contractions_map.items():
        text = text.replace(contraction, expansion)
    # Tokenize the text using SpaCy
    tokens = spacy_en(text)
    # Remove stopwords from the text
    text =  " ".join([token.text for token in tokens if not token.is_stop])
    text = re.sub(r'[^\w\s\.]', '', text)
    text = re.sub("\n", " ", text)
    text = re.sub(' +', ' ', text)
    return text