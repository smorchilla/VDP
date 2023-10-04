import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from clean_gadget import clean_gadget
from blstm import BLSTM
from keras.models import load_model
import keras
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax
from loguru import logger
from sklearn.model_selection import train_test_split
import numpy as np  

@logger.catch
def read_file(path):
    with open(path) as f:
        lines = f.readlines()
    df = pd.DataFrame(lines,columns=['Text'])
    return(df)
@logger.catch
def vectorize(list_of_docs, model):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features



if __name__ == "__main__":
    df = read_file('text_to_predict.txt')
    prep_data = df['Text'].apply(lambda x: clean_gadget(x))
    modelW2v = Word2Vec.load("w2vmodel_1.model")
    df['tokens'] = prep_data
    vectorized_docs = vectorize(df['tokens'], model=modelW2v)
    df['vectors'] = vectorized_docs

    if df.shape[0] < 50:
        zeros = list(np.zeros(100, dtype=int))
        for i in range(df.shape[0],50):
            df.loc[len(df.index)]=[zeros,zeros, zeros]
        # for i in range(df.shape[0],50):
    
        #     df.loc[i]=np.zeros(100)
    
    # выводим массив
    arr = np.zeros((50, 50))
# Заполняем массив значениями из датафрейма

    for i in range(50):
        arr[i] = df['vectors'][i][:50]
    
    arr = np.array([arr])
   
    logger.debug(arr)
    model = Sequential()
    model.add(Bidirectional(LSTM(300), input_shape=( 50, 50)))
    model.add(Dense(300))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(300))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # Lower learning rate to prevent divergence
    adamax = Adamax(lr=0.002)
    model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('cwe119_cgd_model.h5')
    predictions = model.predict(arr)
    print(f'Код уязвим с вероятностью: {predictions[0][0]}')
