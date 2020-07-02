import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 

# here we will do text prediction
# given a sentence, we can predict what the next word might be

# create our tokenizer and a set of sentences
tokenizer = Tokenizer()

data = open('./irish-lyrics-eof.txt').read()
corpus = data.lower().split("\n")

# generate the dictionary of words
tokenizer.fit_on_texts(corpus)
# total word is total amount of words + 1 for out of vocabularly
total_words = len(tokenizer.word_index) + 1

# given the sentence above, we will create our x and y mappings
# the x will be a sequence of words, and the y will be the next word
input_sequences = []
for line in corpus:
    # convert text to integer sequence
    token_list = tokenizer.texts_to_sequences([line])[0]
    # split this sequence in all the ways possible
    for i in range(1, len(token_list)):
        # if the sequence is something like [23, 53, 54, 76, 45], then our n gram seq
        # will be 23 when i = 1, [23, 53] when i = 2, [23, 53, 54] when i = 3, etc
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# we need to pad each sequence to make it the same length
# for us to do that, we need to know the length of the longest sequence
max_sequence_len = max([len(x) for x in input_sequences])
# pad it
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create our x and y mapping here
xs = input_sequences[:, :-1] # all but the last item
labels = input_sequences[:, -1]
'''
If our array was as follows:
array([[0, 2, 5, 4],
       [0, 0, 1, 3]])
xs = array([[0, 2, 5],
       [0, 0, 1]])
labels = array([4, 3])
'''

# one-hot encode the labels
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# create our model now
model = Sequential()
# our sequence length is max length - 1 because the last element is the label
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
# our output word could be any of the total words
model.add(Dense(total_words, activation='softmax'))

# compile our model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

# fit our model
history = model.fit(xs, ys, epochs=500)

# predict
seed_text = "Laurence went to dublin"
# predict the next 100 words
next_words = 100

for _ in range(next_words):
    # convert text to numbers and pad it
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    # predict the next word
    predicted = model.predict_classes(token_list)
    # find the index that this corresponds to
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)