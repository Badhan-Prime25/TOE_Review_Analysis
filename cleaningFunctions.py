import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import porter
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt_tab')

def check_null (data):
    if data.isnull().any().any():
        return data.dropna()
    else:
        return data

def tokenize(text):
    return word_tokenize(str(text))

def open_file():
    with open("customStopWords.txt", "r") as file:
        lines = file.read().splitlines()
    return lines

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    customWords = open_file()
    custom_stopwords = stop.union(customWords)
    return [word for word in text if word.lower() not in custom_stopwords]

def remove_punctuation(text):
    punctuation = string.punctuation
    return [word for word in text if word not in punctuation]

def stemming(text):
    stemmer = porter.PorterStemmer()
    return [stemmer.stem(word) for word in text]

def sentiment(text):
    sentence = ""
    for i in range(len(text)):
        sentence = sentence + text[i] + " "
    return TextBlob(sentence).sentiment

def polarity(text):
    sentence = ""
    for i in range(len(text)):
        sentence = sentence + text[i] + " "
    pol = TextBlob(sentence).polarity
    return pol

def combine(data):
    result = ((data['Rating'] / 5) * 0.5) + (data['polarity'] * 0.5)
    return result

def positive(text):
    if text <= 0.3:
        return False
    else:
        return True

def spelling_correct(text):
    sentence = ""
    for i in range(len(text)):
        sentence = sentence + text[i] + " "
    fixed = TextBlob(sentence).correct()
    return word_tokenize(str(fixed))