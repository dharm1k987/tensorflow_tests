import json
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
# the sarcasm.json has about 27000 entries, so we will test on 20000 and validate on rest
training_size = 20000

# read the data
with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# split up training and validation
training_sentences = sentences[0: training_size]
testing_sentences = sentences[training_size:]
training_labels = np.array(labels[0: training_size])
testing_labels = np.array(labels[training_size:])

# create the tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# create NN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(), # could also use Flatten
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile it
# we will either classify as a sarcastic or not sarcastic review
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit it based on the padded data and the expected label
num_epochs = 30
history = model.fit(
    training_padded, # x
    training_labels, # y
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels)
)

print("Predictions: ")
sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night",
"mom starting to fear son's web series closest thing she will have to grandchild"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
results = model.predict(padded)
for i in results:
    print(np.format_float_positional(i, trim='-'))