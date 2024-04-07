import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
def preprocess(text):

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords_set = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords_set])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text