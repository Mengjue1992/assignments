# coding: utf-8

# In[2]:

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    listTokens = []
    for genre in movies['genres']:
        listTokens.append(tokenize_string(genre))
    
    movies['tokens'] = pd.Series(listTokens, index = movies.index)
    #print(movies['tokens'].tolist())
    return movies


def featurize(movies):
    vocab = {}    
    listVocab = []
    for tok in movies['tokens']:
        for t in tok:
            listVocab.append(t)
    
    uniqueGenres = set(listVocab)    
    counter = 0
    uniqueGenres = sorted(uniqueGenres)
    for genres in uniqueGenres:
        vocab[genres] = counter
        counter += 1
    uniqueDocs = {}
    for toks in movies['tokens']:
        flag = {}
        for t in toks:
            flag[t] = True
        for t in toks:
            if t in uniqueDocs.keys() and flag:
                uniqueDocs[t] += 1
            else:
                uniqueDocs[t] = 1
                flag[t] = False   
    numMovies = len(movies)
    num_features = len(vocab)
    listFeatures = []
    for toks in movies['tokens']:
        tokenFreq = {}
        for t in toks:
            if t in tokenFreq.keys():
                tokenFreq[t] += 1
            else:
                tokenFreq[t] = 1
        row = []
        column = []
        data = []
        maxFreq = tokenFreq[max(tokenFreq)]
        for key, values in tokenFreq.items():
            temp1 = values/maxFreq
            temp2 = np.log10(numMovies/uniqueDocs[key])
            data.append(temp1 * temp2)
            row.append(0)
            column.append(vocab[key])
        listFeatures.append(csr_matrix((data, (row, column)), shape=(1, num_features)).toarray())
    
    movies['features'] = pd.Series(listFeatures, index = movies.index)
    return movies, vocab
        
    


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    temp = a.dot(b.T)    
    normA = 0.0
    for x in a[0]:
        normA += x*x
    norm1 = np.sqrt(normA)
    normB = 0.00    
    for y in b[0]:
        normB += y*y
    norm2 = np.sqrt(normB)   
    norm = norm1 * norm2
    cosine_sim = temp[0][0]/norm
    return cosine_sim


def make_predictions(movies, ratings_train, ratings_test):
    predList = []
    for i in range(len(ratings_test)):
        movieIdTest = ratings_test['movieId'].iloc[i]
        userTest = ratings_test['userId'].iloc[i]
        ratings = ratings_train[ratings_train.userId == userTest]
        movieIdTrain = ratings_train[ratings_train.userId == userTest].movieId        
        wSum = 0.00
        wValue = 0.00
        for movieB in movieIdTrain:
            A = movies[movies.movieId == movieIdTest].features.iloc[0]
            B = movies[movies.movieId == movieB].features.iloc[0]
            rating = ratings[ratings_train.movieId == movieB].rating.iloc[0]            
            weighted = cosine_sim(A, B)                
            if(weighted > 0.00):
                wValue += weighted * rating
                wSum += weighted
            else:
                wValue += 0.00
                wSum += 0.00       
        if(wSum <= 0.00):
            wAvg = sum(ratings.rating)/len(ratings)
        else:
            wAvg = wValue/wSum
        predList.append(wAvg)
    predArray = np.array(predList)
    return predArray
                
def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()

