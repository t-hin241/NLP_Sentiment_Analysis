import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re

#load data
df = pd.read_csv('./all-data.csv',delimiter=',',encoding='latin-1')
df.columns = ['sentiment', 'Message']

#Convert sting to numeric
sentiment  = {'positive': 0,'neutral': 1,'negative':2}
df.sentiment = [sentiment[item] for item in df.sentiment] 

#print(df.head(5))
#print(df.shape)

def print_message(index):
    example = df[df.index == index][['Message', 'sentiment']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Message:', example[1])


#clean message data
def cleanText(text):
    #normalize with bs4
    text = BeautifulSoup(text, "lxml").text
    text = text.replace('x', 'X')
    text = bytes([c if c <= 0x7f else c-33 for c in text.encode('latin-1')])
    #hyperlink is no need for train
    text = re.sub(r'\|\|\|', r' ', str(text)) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.replace('x', '')
    #lowercase
    text = text.lower()
    return text

df['Message'] = df['Message'].apply(cleanText)

#print_message(13)
df.to_csv('./pre_data.csv', index=None, header=True)