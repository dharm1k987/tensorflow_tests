# instead of training our own tokenizers, some datasets offer pre-trained ones
# with subwords
# subwords are not the entire word, but part of it

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

# we can directly get access to the tokenizer
tokenizer = info.features['text'].encoder

print(tokenizer.subwords[0:10])

# see how it encodes and decodes string
sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string is {}'.format(original_string))

for ts in tokenized_string:
    print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

# format the data
max_length = 200
trunc_type = 'post'

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for s,l in train_data:
    training_sentences.append(s.numpy())
    training_labels.append(l.numpy())
  
for s,l in test_data:
    testing_sentences.append(s.numpy())
    testing_labels.append(l.numpy())

training_tokens_padded = np.array(pad_sequences(training_sentences, maxlen=max_length, truncating=trunc_type))
training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)
test_tokens_padded = np.array(pad_sequences(testing_sentences, maxlen=max_length, truncating=trunc_type))
testing_sentences = np.array(testing_sentences)
testing_labels = np.array(testing_labels)

# define the NN
embedding_dim = 64
model = tf.keras.Sequential([
    # each word will have a 16 dim vector associated with it
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim, input_length=training_tokens_padded.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(), # could also use Flatten
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile it
# we will either classify as a negative or positive review
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit it based on the padded data and the expected label
num_epochs = 10
model.fit(
    training_tokens_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(test_tokens_padded, testing_labels)
)