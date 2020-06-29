import json

# just a mapping between url, a title, and whether or not the title is sarcastic
with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index # lots of data

sequences = tokenizer.texts_to_sequences(sentences)
# pad with 0's after, not before
padded = pad_sequences(sequences, padding='post')

print(sentences[0])
print(padded[0]) # will pad after the last number to make it 40 length
print(padded.shape) # (26709, 40) we have 26709 sentences
print(word_index)