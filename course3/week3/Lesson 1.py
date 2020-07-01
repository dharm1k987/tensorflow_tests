"""
https://www.youtube.com/watch?v=LHXXI4-IEns
RNN are types of neural networks designed for capturing information from sequences.

Audio sequences, sentences, are examples. We introduce a hidden state which can allow
information from previous states to flow forward.

As the RNN processes more steps, it has trouble retaining information from much
previous steps. This is short term memory.

The problem with the regular NN is that they do not store
any sort of memory.

LSTM are basically the same as RNN, but it has long term memory. using something called
a cell state. Cell states are bidirectional, meaning that later contexts can
impact earlier ones.
"""

# same as subwords of last week, but this time introduce 2 LSTMs
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

# we can directly get access to the tokenizer
tokenizer = info.features['text'].encoder

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
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    # LSTM layer, if we have more than one we need the return_sequences
    # notice how we do not need the Flatten or GlobalAveragePooling1D layers
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    # noticce how we have 32 outputs, but since its bidirectional, we really have 64
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""
We can also use a GRU (another type of RNN) instead of the LSTM.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

We are even able to use the Conv 1D layer we used on images.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""

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