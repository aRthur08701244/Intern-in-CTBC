#!/usr/bin/python
import json
import os
# import en
import datetime
import nltk
import numpy as np
import pandas as pd
from wakepy import set_keepawake, unset_keepawake
import time
from sklearn.decomposition import PCA



##### !!! NOTICE: input1為初版沒有考慮company embedding !!! #####

set_keepawake(keep_screen_awake=True)

def dateGenerator(numdays): # generate N days until now, eg [20151231, 20151230]
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
    return set(date_list)

def unify_word(word): # went -> go, apples -> apple, BIG -> big
    # try: word = en.verb.present(word) # unify tense
    # except: pass
    # try: word = en.noun.singular(word) # unify noun
    # except: pass
    return word.lower()

def readGlove(we_file):
    wordVec = np.zeros([0,100])
    with open(we_file) as file:
        for line in file:
            line = line.strip().split()
            line = list(map(float,line))
            wordVec = np.vstack((wordVec,np.array(line).flatten()))
    return wordVec

def padding(sentencesVec, keepNum, ticker, wordEmbedding, word2idx):

    keepNum -= 1 # 為了concat vector of company

    shape = sentencesVec.shape[0] # 100
    ownLen = sentencesVec.shape[1] # n words

    if ownLen < keepNum:
        try:
            sentencesVec = np.hstack((sentencesVec, np.array(wordEmbedding[word2idx[ticker]]).reshape(shape, 1)))
            sentencesVec = np.hstack((np.zeros([shape, keepNum-ownLen]), sentencesVec))
            return sentencesVec.flatten()
        except:
            print('Fail')
    else:
        pca = PCA(n_components=keepNum)
        sentencesVec = pca.fit_transform(sentencesVec)

        sentencesVec = np.hstack((sentencesVec, np.array(wordEmbedding[word2idx[ticker]]).reshape(shape, 1)))

        return sentencesVec.flatten()

def gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words=60, mtype="test"):
    set_keepawake(keep_screen_awake=True)
    # step 2, build feature matrix for training data
    loc = './input2/WB_TransNews/'
    if mtype == 'train': loc += 'train/'
    if mtype == 'validation': loc += 'valid/'
    if mtype == 'test': loc += 'test/'
    input_files = [f for f in os.listdir(loc) if f.endswith('cnyesnews.csv')]
    #print(input_files)
    current_idx = 2
    dp = {} # only consider one news for a company everyday
    cnt = 0
    #testDates = dateGenerator(100)
    shape = wordEmbedding.shape[1]
    features = np.zeros([0, max_words * shape]) # 60*100
    # print(features)
    
    # print(max_words, shape, features.shape)
    labels = []
    for i in range(len(input_files)):
        file = input_files[i]
        
        this_year = file.split('_')[0][0:4]

        news = pd.read_csv(loc+file)
        # print(news)

        for i in news.index:

            if len(news.iloc[i].values) != 4: continue

            day, all, tokens, ticker = news.iloc[i].values
            day = str(day)

            if ticker not in priceDt: continue # skip if no corresponding company found
            # print(day, priceDt[ticker].keys())
            # break
            if day not in priceDt[ticker].keys(): continue # skip if no corresponding date found
            # print('y')
            # print(cnt)

            sentencesVec = np.zeros([shape, 0])
            for t in tokens:
                if t not in word2idx: continue
                sentencesVec = np.hstack((sentencesVec, np.array(wordEmbedding[word2idx[t]]).reshape(100, 1)))
            if sentencesVec.shape[1] < max_words: 
                print('Less')
                continue
            cnt += 1

            if cnt <= 393000:continue

            features  = np.vstack((features, padding(sentencesVec, max_words, ticker, wordEmbedding, word2idx)))
            labels.append(round(priceDt[ticker][day], 6))

            if (cnt%1000) == 0:
                features = np.array(features)
                labels = np.matrix(labels)
                featureMatrix = np.concatenate((features, labels.T), axis=1)
                # fileName = './input2/stockFeatures/featureMatrix_' + str(cnt//1000) + "_" + mtype + '.csv'
                # np.savetxt(fileName, featureMatrix, fmt="%s")
                fileName = './input2/stockFeatures_ForCNN/' + this_year + '/featureMatrix_' + str(cnt//1000) + "_" + mtype + '.csv'
                print(fileName)
                pd.DataFrame(featureMatrix).to_csv(fileName)

                features = np.zeros([0, max_words * shape])
                labels = []

    features = np.array(features)
    labels = np.matrix(labels)
    featureMatrix = np.concatenate((features, labels.T), axis=1)
    # fileName = './input2/stockFeatures/featureMatrix_' + str(cnt//1000+1) + "_" + mtype + '.csv'
    # np.savetxt(fileName, featureMatrix, fmt="%s")
    fileName = './input2/stockFeatures_ForCNN/' + this_year + '/featureMatrix_' + str(cnt//1000 + 1) + "_" + mtype + '.csv'
    print(fileName)
    pd.DataFrame(featureMatrix).to_csv(fileName)

    set_keepawake(keep_screen_awake=False)

def build(wordEmbedding, w2i_file, max_words=60):
    with open('./input2/stockReturns.json') as data_file:
        priceDt = json.load(data_file)
    with open(w2i_file) as data_file:
        word2idx = json.load(data_file)
    # print(word2idx)

    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "train")
    # gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "validation")
    # gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "test")
    # Making Additional Features if required
#    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "train",1)
#    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "validation",1)
#    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "test",1)

def main(we, w2i_file):
    wordEmbedding = readGlove(we)
    # print(wordEmbedding, wordEmbedding.shape)
    build(wordEmbedding, w2i_file, 30)


if __name__ == "__main__":
    # we = './input/wordEmbeddingsVocab.csv'
    # w2i_file = "./input/word2idx.json"
    we = './input2/wordEmbeddingsVocab.csv'
    w2i_file = "./input2/word2idx.json"
    main(we, w2i_file)
    set_keepawake(keep_screen_awake=False)


