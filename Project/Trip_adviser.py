# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 01:03:19 2019

@author: Tarun Joshi
"""


# Summer Project

# Trip Adviser


import pandas as pd

# Importing the dataset
dataset = pd.read_csv('hotel-reviews.csv')
dataset=dataset.iloc[:,[1,4]]

dataset=dataset.iloc[0:10000,:]

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
for i in range(0, 10000):
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
tv = TfidfVectorizer(max_features = 1500)
features = tv.fit_transform(corpus).toarray()
labels = dataset.iloc[:10000, 1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
model=classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
print(cm_nb)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())


import pickle
with open('model_pickle','wb')as f:
    pickle.dump(model,f)

with open('model_pickle','rb')as f:
    classifier=pickle.load(f)
    classifier.predict(features)

"""
import speech_recognition as sr
r=sr.Recognizer()
with sr.Microphone() as source:
    print("SAY SOMETHING")
    audio= r.listen(source)
    print("TIME OVER")
    
print("TEXT: "+ r.recognize_google(audio))


str1=r.recognize_google(audio)
"""




str1="good"
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
print(labels_pred)

