# this python module is to define and train 
# a neuron network of machine deep learning

#Reading in files as a string text
def read_file(filepath):   
    with open(filepath) as f:
        str_text = f.read()
    return str_text

#Tokenize and Clean Text
import spacy
nlp = spacy.load('en',disable=['parser', 'tagger','ner'])
​nlp.max_length = 1198623

# remove punctuations
def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

# read text file of original writing 
d = read_file('dick_chapters.txt')
# convert text string to tokens
tokens = separate_punc(d)

# organize tokens into sequences of 51 tokens
train_len = 50+1 # 50 training words , then one target word
text_sequences = []
for i in range(train_len, len(tokens)):
    # Grab train_len# amount of tokens
    seq = tokens[i-train_len:i]
    # Add to list of sequences
    text_sequences.append(seq)

#Keras Tokenization
from keras.preprocessing.text import Tokenizer
# encode sequences of tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
vocabulary_size = len(tokenizer.word_counts)

#Convert to Numpy Matrix
import numpy as np
sequences = np.array(sequences)

#Creating an LSTM based model of neuron networks
# one input layer, one output layer and 2 hidden layers 
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len)) 
    model.add(LSTM(150, return_sequences=True)) 
    model.add(LSTM(150)) 
    model.add(Dense(vocabulary_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.summary()    
    return model 
​
       
#Train / target Split
from keras.utils import to_categorical
# first 50 tokens for training
X = sequences[:,:-1]
# last token to target
y = sequences[:,-1]

# convert target list to matrix
z = to_categorical(y, num_classes=vocabulary_size+1)
seq_len = X.shape[1]

# create model
model = create_model(vocabulary_size+1, seq_len)

from pickle import dump,load
# train model, epoch define cycling
# for a good result, set epoch to 300-500
model.fit(X, z, batch_size=128, epochs=30,verbose=1)

# save the model to file
model.save('model_30.h5')
# save the tokenizer
dump(tokenizer, open('model_30', 'wb'))


