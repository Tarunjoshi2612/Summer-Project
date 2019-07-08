# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 00:15:42 2019

@author: Tarun Joshi
"""
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('hotel-reviews.csv')
dataset=dataset.iloc[:,[1,4]]

dataset=dataset.iloc[0:38900,:]

# Cleaning the texts
# Noise removal
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

#Stemming:  Stemming is a rudimentary rule-based process 
# of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
from nltk.stem.porter import PorterStemmer

corpus = []

#perform row wise noise removal and stemming for every row in dataset. 
for i in range(0, 38900):
    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i][0])
    review = review.lower()
    review = review.split()
    not_stopwords={'not'}
    stwords=set(stopwords.words('english'))-not_stopwords
    review = [word for word in review if not word in stwords]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# Also known as the vector space model
# Text to Features (Feature Engineering on text data)
  
    
    '''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()
labels = dataset.iloc[:1000, 1]
'''
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 3500)
features = tv.fit_transform(corpus).toarray()

def nlp(str1):
    import pickle
    with open('model_pickle','rb')as f:
        classifier=pickle.load(f)
        corpus = []
        review = re.sub('[^a-zA-Z]', ' ', str1)
        review = review.lower()
        review = review.split()
        review = [word for word in review 
            if not word 
            in stwords]
    
#lem = WordNetLemmatizer() #Another way of finding root word
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review]
#review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
        features = tv.transform(corpus).toarray()



# Predicting the Test set results
        labels_pred = classifier.predict(features)[0]
        if labels_pred=="happy":
            return "You seem to be happy from the Hotel"
        elif labels_pred=="not happy":
            return "You seem to be unhappy from the Hotel"

