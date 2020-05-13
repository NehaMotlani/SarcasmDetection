import json
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


pd.options.mode.chained_assignment = None

df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json',lines=True,encoding='utf-8')
df = df.drop('article_link', 1)
# df = df.head()
n = len(df)

print('Removing noise and puntuations')
for i in range(n):
	df['headline'][i] = df['headline'][i].lower()
	df['headline'][i] = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", df['headline'][i])

print('Removing Stopwords')
stop_words = set(stopwords.words('english'))
for i in range(n):
	df['headline'][i] = [w for w in df['headline'][i].split() if not w in stop_words]
	df['headline'][i] = ' '.join(map(str, df['headline'][i])) 

print('Tokenizing the text')
for i in range(n):
	df['headline'][i] = word_tokenize(df['headline'][i])	

print('Converting sentences to vectors')
model = word2vec.Word2Vec(df['headline'], workers = 1, size = 25, min_count = 1, window = 3, sg = 1)
# print(model['work'])

for i in range(n):
	mat = []
	for j in df['headline'][i]:
		mat.append(model[j])
	df['headline'][i] = mat
# print(df.head())

print('Padding the vectors')
x = tf.keras.preprocessing.sequence.pad_sequences(df['headline'],padding='pre')
y = []
for i in df['is_sarcastic']: 
	y.append(i)
y = np.array(y) 

print('Splitting the dataset')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print('Building the model')
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
#create model
model = Sequential()
#add model layers
model.add(Conv1D(128, kernel_size=3, activation='relu',input_shape=(106, 25)))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
model.summary()
print(y_train)
#train the model
# model.fit(X_train[0, :, :], y_train[0], validation_data=(X_test, y_test), epochs=3,batch_size=1)
model.fit(X_train, y_train, batch_size=16, epochs=6)
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)