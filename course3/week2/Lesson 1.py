import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# contains 25000 training and 25000 testing
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# now we can tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'

# training tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

# testing sequences is what we will use validation
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# define the NN
model = tf.keras.Sequential([
    # each word will have a 16 dim vector associated with it
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(), # could also use Flatten
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

'''
Idea of Embedding
- Words that have similar meaning are close to eachother
- In the movie review, it might say movie was dull and boring
- What if we could pick a vector in a higher dim space, ie, 16, and words
that are similar are given similar vectors
- Overtime, words can begin to cluster together
- Meaning of the word can come from the labelling of the dataset (good or bad review)
'''

# compile it
# we will either classify as a negative or positive review
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit it based on the padded data and the expected label
num_epochs = 10
model.fit(
    padded, # x
    training_labels_final, # y
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels_final)
)

#########################
print("Generating .tsv file")
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

import io

e = model.layers[0]
weights = e.get_weights()[0]
out_v = io.open('./vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('./meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
#########################

print("Predictions:")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
training_sentences = [
    'This is an amazing movie that is really great and fun to watch',
    'The movie is terrible and sad and lacks taste. Terrible storytelling and unfavourable acting'
]
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

print(model.predict(padded))