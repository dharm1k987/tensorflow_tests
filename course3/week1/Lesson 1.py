# NATURAL LANGUAGE PROCESSING

import tensorflow as tf
from tensorflow import keras

# will turn our text into streams of tokens
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# Tokenizer will take the top 100 words by volume and just encode those
# oov_token is what will be printed if the word is not in the vocabulary
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# encodes data
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index # {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5 ... }

# just turn each sentence into a list of their indexes
sequences = tokenizer.texts_to_sequences(sentences) # [[5, 3, 2, 4], [5, 3, 2, 7], ...]
# pad the sequences based on a max length of 5 from the end (leading words are stripped)
padded = pad_sequences(sequences, maxlen=5)
'''
[[ 0  5  3  2  4]
 [ 0  5  3  2  7]
 [ 0  6  3  2  4]
 [ 9  2  4 10 11]]
'''

print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)

# what if we try to encode a word that is not in the vocabularly
test_data = [
    'i really love my dog',
    'my dog loves my manatee alot'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)