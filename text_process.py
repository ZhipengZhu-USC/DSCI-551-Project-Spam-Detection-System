import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import re

#nltk.download('wordnet')

def process_text(text):
    #the function takes a string and return a list of cleaned words.

    #tokenize
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub("``",'',text)
    tokenized = nltk.word_tokenize(text)
    #print("after tokenization: \n ", tokenized)

    #stopwords & punc
    stop_words = set(stopwords.words('english'))
    nopunc = [char for char in tokenized if char not in string.punctuation]
    nopunc = ' '.join(nopunc)
    filtered_text = [word for word in nopunc.split() if len(word) >= 2 if not word.lower() in stop_words]
    #print("after stop words filtering: \n ", filtered_text)

    #stemming
    stemmer = SnowballStemmer(language='english')
    lemmatizer = WordNetLemmatizer()
    result = []
    for i in filtered_text:
        stemmed_word = stemmer.stem(i)
        lemmatized = lemmatizer.lemmatize(stemmed_word)
        result.append(i)
        #stemmed.append(stemmer.stem(i))
    #print("after stemming: \n ", stemmed)

    return result
