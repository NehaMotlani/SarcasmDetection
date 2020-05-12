import json

data = [json.loads(line) for line in open('Sarcasm_Headlines_Dataset_v2.json', 'r')]

for element in data:
    element.pop('article_link', None)

tweets = []

for i in data:
	tweets.append(i['headline'].encode("utf-8"))

print(tweets)
