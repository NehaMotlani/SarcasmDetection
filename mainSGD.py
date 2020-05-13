import json
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


pd.options.mode.chained_assignment = None

df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json',lines=True,encoding='utf-8')
df = df.drop('article_link', 1)
n = len(df)

for i in range(n):
	df['headline'][i] = df['headline'][i].lower()
	df['headline'][i] = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", df['headline'][i])

pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', SGDClassifier()),
])

X_train, X_test, y_train, y_test = train_test_split(df['headline'],df['is_sarcastic'],random_state = 0)

model = pipeline_sgd.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(f1_score(y_test, y_predict))